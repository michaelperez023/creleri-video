"""CReLeRI object-proposal service.

POST /process_video with {"video_path": "<local video file>"}. Objects are detected with D-FINE
(Objects365 classes) and tracked with StrongSORT. Each allowed object that is tracked through
enough of the clip is boxed in a copy of the video, which the Qwen2.5-VL service captions.
The response lists the objects and, aligned with them, three captions per object. Tracks whose
captioning fails are left out; if captioning fails for every track, the service answers 502.

Environment: DFINE_CKPT (D-FINE checkpoint), QWEN_API_URL (captioning service base URL),
CRELERI_HOST (bind address).
"""

import argparse
import json
import logging
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import requests
import torch
import torch.nn as nn
import torchvision.transforms as T
import yaml
from boxmot import StrongSort
from flask import Flask, jsonify, request
from PIL import Image
from torch import amp

HERE = Path(__file__).resolve().parent
DFINE_ROOT = HERE.parents[1]

sys.path.append(str(DFINE_ROOT))
from src.core import YAMLConfig  # noqa: E402

logger = logging.getLogger(__name__)

CONFIG_PATH = DFINE_ROOT / "configs/dfine/objects365/dfine_hgnetv2_x_obj365.yml"
CHECKPOINT_PATH = Path(os.environ.get("DFINE_CKPT") or DFINE_ROOT / "dfine_x_obj365.pth")
LABEL_YAML_PATH = HERE / "Objects365.yaml"
# boxmot downloads these ReID weights to this path the first time a tracker is built.
REID_WEIGHTS = HERE / "osnet_ain_x1_0_msmt17.pt"
QWEN_API_URL = (os.environ.get("QWEN_API_URL") or "http://localhost:17259").rstrip("/")
DEVICE = "cuda:0"

# Objects365 classes proposed for captioning; tracks of any other class are dropped.
ALLOWED_LABELS = {
    "Person",
    "Car",
    "Boat",
    "SUV",
    "Van",
    "Bus",
    "Motorcycle",
    "Truck",
    "Sailboat",
    "Pickup Truck",
    "Airplane",
    "Train",
    "Sports Car",
    "Scooter",
    "Stroller",
    "Crane",
    "Fire Truck",
    "Ship",
    "Hot-air balloon",
    "Helicopter",
    "Rickshaw",
    "Barrel/bucket",
    "Pen/Pencil",
    "Knife",
    "Hockey Stick",
    "Paddle",
    "Fork",
    "Spoon",
    "Hanger",
    "Keyboard",
    "Mouse",
    "Nightstand",
    "Remote",
    "Shovel",
    "Cutting/chopping Board",
    "Scissors",
    "Ladder",
    "Converter",
    "Paint Brush",
    "Tong",
    "Tennis Racket",
    "Tape",
    "Router/modem",
    "Pliers",
    "Hammer",
    "Screwdriver",
    "Tape Measure/Ruler",
    "Stapler",
    "Electric Drill",
    "Mop",
    "Chainsaw",
    "Cow",
    "Sheep",
    "Duck",
    "Cat",
    "Elephant",
    "Zebra",
    "Giraffe",
    "Deer",
    "Goose",
    "Penguin",
    "Swan",
    "Goldfish",
    "Bear",
    "Pig",
    "Camel",
    "Antelope",
    "Parrot",
    "Seal",
    "Butterfly",
    "Donkey",
    "Lion",
    "Dolphin",
    "Jellyfish",
    "Monkey",
    "Rabbit",
    "Yak",
    "Gun",
    "Dishwasher",
    "Washing Machine/Drying Machine",
    "Blender",
    "Coffee Machine",
}

CAPTION_PROMPTS = [
    "What is the {object} doing in the video? Please only provide a caption if there is a clear action.",
    "Describe what the {object} is doing in the video. If there is no action, type 'no action'.",
    "What is the {object} doing in the video? If no action is observed, type 'no action'.",
]

app = Flask(__name__)
model = label_mapping = None  # set by load_model()


class Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.model = cfg.model.deploy()
        self.postprocessor = cfg.postprocessor.deploy()

    def forward(self, images, orig_target_sizes):
        outputs = self.model(images)
        return self.postprocessor(outputs, orig_target_sizes)


def load_model():
    """Load the D-FINE detector onto DEVICE and the Objects365 label names."""
    global model, label_mapping
    cfg = YAMLConfig(str(CONFIG_PATH))
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    state = checkpoint["ema"]["module"] if "ema" in checkpoint else checkpoint["model"]
    cfg.model.load_state_dict(state)
    logger.info("Loaded D-FINE weights from %s", CHECKPOINT_PATH)

    model = Model(cfg).to(DEVICE)
    model.eval()

    with open(LABEL_YAML_PATH) as f:
        label_mapping = yaml.safe_load(f)["names"]

    unknown_labels = ALLOWED_LABELS - set(label_mapping.values())
    if unknown_labels:
        logger.warning("ALLOWED_LABELS that are not Objects365 classes: %s", sorted(unknown_labels))


def compute_iou(box1, box2):
    """Compute IoU between two bounding boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0


def generate_video(input_file, boxes_by_frame, output_file, frame_width, frame_height, fps):
    """Copy the input video with the track's box drawn on every frame it was tracked in."""
    out = cv2.VideoWriter(
        output_file, cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height)
    )
    if not out.isOpened():
        raise RuntimeError(f"Cannot write video: {output_file}")

    cap = cv2.VideoCapture(input_file)
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        bbox = boxes_by_frame.get(frame_count)
        if bbox is not None:
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue box (BGR format)

        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()


def process_video(model, device, video_path, out_dir, threshold=0.5):
    """Detect, track and label the objects in a video, writing one clip per allowed track.

    Returns (label, clip_path) pairs in the order the tracks first appeared.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    tracker = StrongSort(
        reid_weights=REID_WEIGHTS,
        device=device,
        half=device != "cpu",
        max_cos_dist=0.7,
        max_iou_dist=0.7,
        max_age=30,
        n_init=3,
        nn_budget=200,
        mc_lambda=0.98,
        ema_alpha=0.9,
    )

    transforms = T.Compose(
        [
            T.Resize((640, 640)),
            T.ToTensor(),
        ]
    )
    track_records = {}

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        w, h = frame_pil.size
        orig_size = torch.tensor([[w, h]]).to(device)
        im_data = transforms(frame_pil).unsqueeze(0).to(device)

        with torch.inference_mode(), amp.autocast("cuda"):
            labels, boxes, scores = model(im_data, orig_size)

        valid_indices = scores[0] > threshold
        filtered_boxes = boxes[0][valid_indices].detach().cpu()
        filtered_labels = labels[0][valid_indices].detach().cpu()
        filtered_scores = scores[0][valid_indices].detach().cpu()

        detection_array = np.array(
            [
                [float(x1), float(y1), float(x2), float(y2), float(score), int(label)]
                for box, score, label in zip(filtered_boxes, filtered_scores, filtered_labels)
                for x1, y1, x2, y2 in [box.tolist()]
            ],
            dtype=float,
        )

        for x1, y1, x2, y2, track_id, *_ in tracker.update(detection_array, frame):
            bbox = [x1, y1, x2, y2]

            if track_id not in track_records:
                track_records[track_id] = {"labels": [], "boxes": {}}

            # Match detection to track label
            matched_label = None
            max_iou = 0
            for det_box, label in zip(filtered_boxes, filtered_labels):
                iou = compute_iou(bbox, det_box)
                if iou > max_iou and iou > 0.5:  # IoU threshold
                    max_iou = iou
                    matched_label = int(label)

            if matched_label is not None:
                track_records[track_id]["labels"].append(matched_label)

            track_records[track_id]["boxes"][frame_count] = bbox

        frame_count += 1
        if frame_count % 10 == 0:
            logger.debug("Processed %d/%d frames", frame_count, total_frames)

    cap.release()

    coverage_threshold = 0.2
    tracks = []
    disallowed_labels = set()
    for track_id, record in track_records.items():
        if len(record["boxes"]) / total_frames < coverage_threshold or not record["labels"]:
            continue

        # Majority vote; the model's labels are 1-based Objects365 ids, the YAML is 0-based.
        most_common_label, _ = Counter(record["labels"]).most_common(1)[0]
        label = label_mapping[most_common_label - 1]
        if label not in ALLOWED_LABELS:
            disallowed_labels.add(label)
            continue

        clip_path = os.path.join(out_dir, f"track_{int(track_id)}.mp4")
        generate_video(video_path, record["boxes"], clip_path, frame_width, frame_height, fps)
        tracks.append((label, clip_path))

    if disallowed_labels:
        logger.info("Dropped tracks with disallowed labels: %s", sorted(disallowed_labels))
    logger.info(
        "Kept %d of %d tracks from %d frames of %s",
        len(tracks),
        len(track_records),
        frame_count,
        video_path,
    )
    return tracks


def get_qwen_captions(video_path, label):
    """Ask the Qwen2.5-VL service what `label` is doing in the clip, once per prompt."""
    # One single-turn conversation per prompt, sent as one batch. The service substitutes the
    # uploaded clip for the empty "file://" video URL.
    messages = [
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": "file://",
                        "max_pixels": 360 * 420,
                        "fps": 1.0,
                    },
                    {"type": "text", "text": prompt.format(object=label)},
                ],
            }
        ]
        for prompt in CAPTION_PROMPTS
    ]

    with open(video_path, "rb") as f:
        resp = requests.post(
            f"{QWEN_API_URL}/qwen",
            data={"messages": json.dumps(messages), "is_batch": "true"},
            files={"video": (os.path.basename(video_path), f, "video/mp4")},
            timeout=120,
        )
    resp.raise_for_status()
    return resp.json()["response"]


def caption_tracks(tracks):
    """Caption each (label, clip_path) pair; a track whose captioning fails is left out."""
    objects, captions = [], []
    for label, clip_path in tracks:
        try:
            track_captions = get_qwen_captions(clip_path, label)
        except (requests.RequestException, KeyError, ValueError) as e:
            logger.warning(
                "Captioning failed for %s (%s): %r", label, os.path.basename(clip_path), e
            )
            continue
        objects.append(label)
        captions.append(track_captions)
    return objects, captions


@app.route("/process_video", methods=["POST"])
def process_video_api():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 400
    video_path = data.get("video_path")
    if not video_path:
        return jsonify({"error": "Missing 'video_path' in request data."}), 400
    if not isinstance(video_path, str) or not os.path.isfile(video_path):
        return jsonify({"error": "'video_path' must be an existing local file."}), 400

    logger.info("Processing %s", video_path)
    try:
        with tempfile.TemporaryDirectory(prefix="objdet_") as out_dir:
            tracks = process_video(model, DEVICE, video_path, out_dir)
            objects, captions = caption_tracks(tracks)
    except Exception as e:
        logger.exception("Failed to process %s", video_path)
        return jsonify({"error": str(e)}), 500
    if tracks and not objects:
        return jsonify({"error": "Captioning failed for every track."}), 502

    return jsonify(
        {
            "message": "Video processing complete.",
            "objects": objects,
            "captions": captions,
            # The per-track clips only live for the duration of the request.
            "videos": [],
        }
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("CRELERI_HOST", "127.0.0.1"),
        help="interface to bind (default: $CRELERI_HOST or 127.0.0.1)",
    )
    parser.add_argument("--port", type=int, default=17266, help="port (default: %(default)s)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    load_model()
    # Requests share one model on one GPU, so serve them one at a time.
    app.run(host=args.host, port=args.port, debug=False, threaded=False)
