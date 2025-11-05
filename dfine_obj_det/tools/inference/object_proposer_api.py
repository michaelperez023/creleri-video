import torch
import torch.nn as nn
import torchvision.transforms as T
import numpy as np
from PIL import Image
import cv2
import os
import sys
import json
from torch import amp
from pathlib import Path
from collections import Counter
from flask import Flask, request, jsonify
import requests

from strong_sort import StrongSort

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.core import YAMLConfig

app = Flask(__name__)

QWEN_PORT = 17259

# Initialize model once
CONFIG_PATH = "../../configs/dfine/objects365/dfine_hgnetv2_x_obj365.yml"
RESUME_PATH = "../../dfine_x_obj365.pth"
LABEL_YAML_PATH = "Objects365.yaml"

ALLOWED_LABELS = {
    "Person",
    "Car", "Boat", "SUV", "Van", "Bus", "Motorcycle", "Truck", "Sailboat", "Pickup Truck", "Airplane", "Train",
    "Sports Car", "Scooter", "Stroller", "Crane", "Fire Truck", "Ship", "Hot-air balloon", "Helicopter", "Rickshaw",
    "Barrel/bucket", "Pen/Pencil", "Knife", "Hockey Stick", "Paddle", "Fork", "Spoon", "Hanger", "Keyboard",
    "Mouse", "Nightstand", "Remote", "Shovel", "Cutting/chopping Board", "Scissors", "Ladder", "Converter",
    "Paint Brush", "Tong", "Tennis Racket", "Tape", "Router/modem", "Pliers", "Hammer", "Screwdriver",
    "Tape Measure/Ruler", "Stapler", "Electric Drill", "Mop", "Chainsaw",
    "Cow", "Sheep", "Duck", "Cat", "Elephant", "Zebra", "Giraffe", "Deer", "Goose", "Penguin", "Swan", "Goldfish",
    "Bear", "Pig", "Camel", "Antelope", "Parrot", "Seal", "Butterfly", "Donkey", "Lion", "Dolphin", "Jellyfish",
    "Monkey", "Rabbit", "Yak",
    "Gun", "Bow", "Arrow", "Sword", "Shield",
    "Robot", "Drone", "Excavator", "Bulldozer", "Crane", "Helicopter", "Factory Machine", "Forklift",
    "Autonomous Vehicle", "Roomba", "Dishwasher", "Washing Machine/Drying Machine", "Blender", "Coffee Machine",
    "3D Printer", "Industrial Robot", "Vacuum Cleaner", "Lawn Mower", "Autonomous Tractor"
}

def load_label_mapping(label_yaml_path):
    import yaml
    with open(label_yaml_path, 'r') as file:
        yaml_data = yaml.safe_load(file)
    return yaml_data['names']

cfg = YAMLConfig(CONFIG_PATH, resume=RESUME_PATH)
if 'HGNetv2' in cfg.yaml_cfg:
    cfg.yaml_cfg['HGNetv2']['pretrained'] = False

if RESUME_PATH:
    checkpoint = torch.load(RESUME_PATH, map_location='cpu')
    if 'ema' in checkpoint:
        state = checkpoint['ema']['module']
    else:
        state = checkpoint['model']
else:
    raise AttributeError('Only support resume to load model.state_dict by now.')

cfg.model.load_state_dict(state)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = cfg.model.deploy()
        self.postprocessor = cfg.postprocessor.deploy()

    def forward(self, images, orig_target_sizes):
        outputs = self.model(images)
        outputs = self.postprocessor(outputs, orig_target_sizes)
        return outputs

model = Model().to("cuda:0")
model.eval()
label_mapping = load_label_mapping(LABEL_YAML_PATH)


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


def generate_video(input_file, track_data, track_id, label_name, output_file, frame_width, frame_height, fps):
    cap = cv2.VideoCapture(input_file)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None

    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Draw blue bounding boxes for the track on the frame
        for bbox in track_data["bboxes"]:
            if bbox["frame"] == frame_count:
                x1, y1, x2, y2 = map(int, bbox["bbox"])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue box (BGR format)

        if out is None:
            out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

        out.write(frame)
        frame_count += 1

    cap.release()
    if out:
        out.release()

def process_video(model, device, file_path, threshold=0.5, output_dir="./tmp2"):
    filename_without_extension = os.path.splitext(os.path.basename(file_path))[0]
    segment_output_dir = os.path.join(output_dir, filename_without_extension)
    if os.path.exists(segment_output_dir) and len(os.listdir(segment_output_dir)) > 0:
        print(f"Output folder {segment_output_dir} already exists and is not empty. Skipping video processing.")
        return segment_output_dir

    if not os.path.exists(segment_output_dir):
        os.makedirs(segment_output_dir)
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {file_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)  or 640)
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
    fps = float(fps)
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize StrongSort
    tracker = StrongSort(
        reid_weights=Path("osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0.0015_coslr_b64_fb10_softmax_labsmth_flip_jitter.pt"),
        device=device,
        half=device != 'cpu',  # Use half precision if device is not CPU
        max_cos_dist=0.7,
        max_iou_dist=0.7,
        max_age=30,
        n_init=3,
        nn_budget=200,
        mc_lambda=0.98,
        ema_alpha=0.9
    )

    transforms = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
    ])
    track_records = {}

    print("Processing video frames...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to PIL and preprocess
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        w, h = frame_pil.size
        orig_size = torch.tensor([[w, h]]).to(device)
        im_data = transforms(frame_pil).unsqueeze(0).to(device)

        # Model inference with mixed precision
        with torch.inference_mode(), amp.autocast("cuda"):
            output = model(im_data, orig_size)
            labels, boxes, scores = output

        # Filter detections by threshold
        valid_indices = scores[0] > threshold
        filtered_boxes = boxes[0][valid_indices].detach().cpu()  # Detach and move to CPU
        filtered_labels = labels[0][valid_indices].detach().cpu()  # Detach and move to CPU
        filtered_scores = scores[0][valid_indices].detach().cpu()  # Detach and move to CPU

        # Prepare detections for StrongSort
        detection_array = np.array([
            [float(x1), float(y1), float(x2), float(y2), float(score), int(label)]
            for box, score, label in zip(filtered_boxes, filtered_scores, filtered_labels)
            for x1, y1, x2, y2 in [box.tolist()]
        ], dtype=float)

        # Update StrongSort tracker
        track_outputs = tracker.update(detection_array, frame)

        # Process confirmed tracks
        for track in track_outputs:
            x1, y1, x2, y2, track_id, conf, cls, det_ind = track
            bbox = [x1, y1, x2, y2]

            if track_id not in track_records:
                track_records[track_id] = {"labels": [], "bboxes": []}

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

            track_records[track_id]["bboxes"].append({"frame": frame_count, "bbox": bbox})

        frame_count += 1
        if frame_count % 10 == 0:
            print(f"Processed {frame_count}/{total_frames} frames...")

    cap.release()

    # Calculate the final label for each track
    for track_id, record in track_records.items():
        # Use majority vote to determine the final label
        if record["labels"]:
            label_counter = Counter(record["labels"])
            most_common_label, _ = label_counter.most_common(1)[0]
            record["final_label"] = most_common_label
            record["final_label_text"] = label_mapping[most_common_label - 1] if label_mapping else str(most_common_label)
        else:
            record["final_label"] = None
            record["final_label_text"] = None

        record["coverage"] = len(record["bboxes"]) / total_frames

    # Filter tracks based on coverage threshold
    coverage_threshold = 0.2
    valid_tracks = {
        track_id: record
        for track_id, record in track_records.items()
        if record["coverage"] >= coverage_threshold and record["final_label"] is not None
    }
    
    if not valid_tracks:
        print("No valid tracks; returning empty folder.")
        return folder_path
    
    valid_labels = [rec.get("final_label_text") for rec in valid_tracks.values() if rec.get("final_label_text")]

    # Compute disallowed (unique) labels
    disallowed_labels = sorted({lab for lab in valid_labels if lab not in ALLOWED_LABELS})
    print(f"Disallowed labels (not saved): {disallowed_labels}")

    # Only keep tracks whose final label is allowed
    allowed_tracks = {
        track_id: record
        for track_id, record in valid_tracks.items()
        if record.get("final_label_text") in ALLOWED_LABELS
    }

    # Generate videos for each allowed track if folder is empty
    if len(os.listdir(segment_output_dir)) == 0:
        print("Generating videos for allowed tracks...")
        for track_id, record in allowed_tracks.items():
            label_name = record["final_label_text"]
            video_path = os.path.join(segment_output_dir, f"{label_name}_{track_id}.mp4")
            generate_video(file_path, record, track_id, label_name, video_path, frame_width, frame_height, fps)


    print(f"Video processing complete. Individual track videos saved to {segment_output_dir}.")
    return segment_output_dir


@app.route('/process_video', methods=['POST'])
def process_video_api():
    data = request.get_json()
    print(data)
    video_path = data.get('video_path')
    if not video_path:
        return jsonify({"error": "Missing 'video_path' in request data."}), 400

    try:
        output_folder = process_video(model, device="cuda:0", file_path=video_path)
        all_objects, all_captions, all_videos = process_folder_for_captions(output_folder)
        response = {
            "message": "Video processing complete.",
            "objects": all_objects,
            "captions": all_captions,
            "videos":all_videos
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def get_qwen_batch_captions(video_path, objects: list):
    url = f'http://localhost:{QWEN_PORT}/qwen'

    prompts = [
        "What is the {object} doing in the video? Please only provide a caption if there is a clear action.",
        "Describe what the {object} is doing in the video. If there is no action, type 'no action'.",
        "What is the {object} doing in the video? If no action is observed, type 'no action'."
    ]

    vlm_captions = []

    # Open once; re-use handle (rewind before each POST)
    with open(video_path, 'rb') as f:
        for obj in objects:
            questions = [p.format(object=obj) for p in prompts]

            # Build a proper Python object (list of conversations), then JSON-encode it
            # Shape here: batch of 3 conversations, each a one-turn list
            messages_obj = [[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": "file://",
                            "max_pixels": 360 * 420,
                            "fps": 1.0,
                        },
                        {"type": "text", "text": q},
                    ],
                }
            ] for q in questions]

            payload = {
                "messages": json.dumps(messages_obj),
                "is_batch": "true"
            }

            f.seek(0)
            files = {
                "video": (os.path.basename(video_path), f, "video/mp4")
            }

            resp = requests.post(url, data=payload, files=files, timeout=120)
            resp.raise_for_status()
            vlm_captions.append(resp.json()["response"])

    return objects, vlm_captions


def process_folder_for_captions(output_dir):
    all_objects = []
    all_captions = []
    all_videos = []
    print(output_dir)

    for video_file in os.listdir(output_dir):
        print(f"video file: {video_file}")
        if video_file.endswith(".mp4"):
            video_path = os.path.join(output_dir, video_file)
            label_name = video_file.split('_')[0]
            
            all_videos.append(video_path)
            objects, captions = get_qwen_batch_captions(video_path, [label_name])
            all_objects.extend(objects)
            all_captions.append(captions[0])
    print("all objects: ")
    print(all_objects)
    return all_objects, all_captions, all_videos

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=17266, debug=True)
