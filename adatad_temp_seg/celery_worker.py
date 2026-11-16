"""Celery worker that segments a video and explains the actions in each segment.

Start it with `python celery_worker.py -d <gpu>`; the models are loaded before tasks are consumed.
"""

import argparse
import itertools
import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

import coloredlogs
import cv2
import decord
import ffmpeg
import numpy as np
import pandas as pd
import requests
import torch
from celery import Celery
from mmengine.config import Config
from opentad.cores import eval_one_epoch_single_video
from opentad.datasets import build_dataloader, build_dataset
from opentad.models import build_detector
from opentad.utils import set_seed
from segment_utils import (
    SegmentationType,
    actionable_predictions,
    adjust_segments_to_scene_changes,
    build_predictions,
    cluster_representatives,
    flatten_captions,
    frame_range,
    generate_intervals,
    normalize_arguments,
    pair_text,
    parse_arguments,
    select_proposals,
    subject_object_pairs,
)
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "e2e_thumos_videomaev2_g_768x2_224_adapter-inference.py"
CHECKPOINT_PATH = BASE_DIR / "best.pth"
CLASS_MAP_PATH = BASE_DIR / "category_idx_51.txt"
ACTION_CATEGORIES_PATH = BASE_DIR / "action_categories.csv"
SHARED_DIR = BASE_DIR / "shared"
THUMBNAIL_DIR = BASE_DIR / "thumbnails"

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:17253/0")
LLAMA_API_URL = os.environ.get("LLAMA_API_URL", "http://localhost:5003")
OBJECT_PROPOSER_URL = os.environ.get("OBJECT_PROPOSER_URL", "http://localhost:17266")
GROUNDING_API_URL = os.environ.get("GROUNDING_API_URL", "http://localhost:12000")

# (connect, read) timeouts in seconds. Tracking and captioning every object of a long segment
# can take most of an hour; grounding only stores the clip and queues jobs.
LLM_TIMEOUT = (10, 180)
OBJECT_PROPOSER_TIMEOUT = (10, 3600)
GROUNDING_TIMEOUT = (10, 300)
LLM_ATTEMPTS = 3
CLIP_CHUNK_FRAMES = 32

ACTION_PROMPT = (
    "You are a strict classifier. Pick exactly one action from this set:\n"
    "[{actions}]\n\n"
    "Rules:\n"
    "• Output ONLY the action label, in lowercase, no quotes, no punctuation, no explanation.\n"
    "• If there is any uncertainty, output: none\n"
    "• Do not invent labels. Choose from the set or 'none' only."
)
ARGUMENT_PROMPT = (
    "You are a strict information extractor.\n"
    "Given a caption and an action, return a SINGLE JSON object with physical entities only.\n"
    "Rules:\n"
    '• Format: {"argument0":"...", "argument1":"...", "argument2":"..."}\n'
    "• argument0 = subject/actor (or 'none' if absent)\n"
    "• argument1 = direct object/recipient (or 'none' if absent)\n"
    "• argument2 = tool/produced/indirect entity (omit if none)\n"
    "• No inference beyond what is explicitly stated; do not invent entities.\n"
    "• Output ONLY the JSON object; no prose, no code fences, no extra text."
)

logger = logging.getLogger(__name__)

celery_app = Celery("tasks", broker=REDIS_URL, backend=REDIS_URL)
celery_app.conf.update(
    # Tasks run for minutes to hours on a single GPU: reserve no extra messages, and keep Redis
    # from redelivering the reserved one while the current task is still running.
    worker_prefetch_multiplier=1,
    broker_transport_options={"visibility_timeout": 6 * 60 * 60},
    task_track_started=True,
)

# Set by initialize_globals() before the worker starts consuming tasks.
cfg = None
model = None
device = None
use_amp = False
sentence_transformer_model = None


def initialize_globals(device_id):
    global cfg, model, device, use_amp, sentence_transformer_model

    cfg = Config.fromfile(CONFIG_PATH)
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    set_seed(42)

    model = build_detector(cfg.model).to(device)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    key = "state_dict_ema" if checkpoint.get("state_dict_ema") else "state_dict"
    model.load_state_dict({k.removeprefix("module."): v for k, v in checkpoint[key].items()})
    # With pretrain=None, BackboneWrapper warns that the backbone is randomly initialized; the
    # strict load above has replaced every weight.
    logger.info("Loaded %s from %s", key, CHECKPOINT_PATH)

    use_amp = getattr(cfg.solver, "amp", False)
    sentence_transformer_model = SentenceTransformer("all-mpnet-base-v2", device=str(device))

    SHARED_DIR.mkdir(exist_ok=True)
    THUMBNAIL_DIR.mkdir(exist_ok=True)


def open_video(video_path):
    """Open a video with decord, rejecting files without usable frame metadata."""
    vr = decord.VideoReader(video_path, ctx=decord.cpu())
    fps = vr.get_avg_fps()
    if len(vr) == 0 or not fps > 0:
        name = os.path.basename(video_path)
        raise ValueError(f"{name}: unusable video metadata ({len(vr)} frames, {fps} fps)")
    return vr, fps


def get_video_duration_decord(video_path):
    vr, fps = open_video(video_path)
    return len(vr) / fps


def get_video_duration_ffmpeg(video_path):
    duration = float(ffmpeg.probe(video_path)["format"].get("duration", 0))
    if not duration > 0:
        raise ValueError(f"{os.path.basename(video_path)}: ffprobe reports no duration")
    return duration


def detect_transitions(video_path, threshold=30):
    """Times (s, rounded to 0.01) of frames whose mean gray-level change exceeds threshold."""
    cap = cv2.VideoCapture(video_path)
    try:
        success, prev_frame = cap.read()
        if not success:
            logger.warning("OpenCV cannot read %s; skipping scene detection", video_path)
            return []
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps > 0:
            raise ValueError(f"{os.path.basename(video_path)}: OpenCV reports no frame rate")
        # Only a progress hint: some containers report no frame count or a bogus one.
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        transitions = []
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        with tqdm(
            total=frame_count - 1 if frame_count > 1 else None,
            desc=f"Processing {os.path.basename(video_path)}",
        ) as progress:
            for i in itertools.count(1):
                success, frame = cap.read()
                if not success:
                    break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if cv2.absdiff(gray, prev_gray).mean() > threshold:
                    transitions.append(round(i / fps, 2))
                prev_gray = gray
                progress.update()
        return transitions
    finally:
        cap.release()


def segment_video(video_path):
    """Segments per video id from AdaTAD proposals, split at scene changes."""
    scene_changes = detect_transitions(video_path)
    logger.info("Scene changes detected: %s", scene_changes)
    video_duration = get_video_duration_decord(video_path)

    cfg.dataset.test = dict(
        type="SingleVideoDataset",
        video_path=video_path,
        class_map=str(CLASS_MAP_PATH),
        pipeline=cfg.dataset.test.pipeline,
        feature_stride=4,
        sample_stride=1,
        window_size=768,
        window_overlap_ratio=0.5,
        test_mode=True,
    )
    test_loader = build_dataloader(
        build_dataset(cfg.dataset.test), rank=0, world_size=1, **cfg.solver.test
    )
    result = eval_one_epoch_single_video(test_loader, model, cfg, use_amp=use_amp, device=device)

    segments = {}
    for video_id, proposals in result["results"].items():
        selected = select_proposals(proposals)
        logger.info("Selected proposals: %s", selected)
        segments[video_id] = adjust_segments_to_scene_changes(
            selected, scene_changes, round(video_duration, 2), min_duration=1
        )
    return segments


def query_llm(messages, constraint_words=None):
    """Send a batch of conversations to the LLaMA service and return one reply per conversation."""
    payload = {"messages": messages, "is_batch": True}
    if constraint_words is not None:
        payload["constraint_words"] = constraint_words
    response = requests.post(f"{LLAMA_API_URL}/llama33", json=payload, timeout=LLM_TIMEOUT)
    response.raise_for_status()
    replies = response.json().get("response")
    if not (
        isinstance(replies, list)
        and len(replies) == len(messages)
        and all(isinstance(reply, str) for reply in replies)
    ):
        raise ValueError(f"LLM service did not return {len(messages)} text replies")
    return replies


def llm_parse_actions(captions, action_list):
    """Classify each caption as one of the vocabulary actions or "none"."""
    choices = ", ".join(
        f"'{action}': {description} (Example: {examples})"
        for action, description, examples in action_list
    )
    system_prompt = ACTION_PROMPT.format(actions=choices)
    messages = [
        [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"Caption:\n{caption}\n\nReturn exactly one label from the set above or 'none'."
                ),
            },
        ]
        for caption in captions
    ]
    return query_llm(messages, constraint_words=[a[0] for a in action_list] + ["none"])


def llm_parse_arguments(actions, captions, action_dict):
    """Extract normalized argument0..argument2 for every caption; all "none" for "none" actions."""
    messages = []
    for action, caption in zip(actions, captions):
        if action == "none":
            continue
        usage, example = action_dict.get(action, ["none", "none"])
        user = (
            f"Action: {action}\n"
            f"Usage: {usage}\n"
            f"Example: {example}\n"
            f"Caption:\n{caption}\n"
            "Return the JSON object now."
        )
        messages.append(
            [{"role": "system", "content": ARGUMENT_PROMPT}, {"role": "user", "content": user}]
        )
    if not messages:
        return [normalize_arguments({}) for _ in actions]

    replies = iter(query_llm(messages))
    return [
        normalize_arguments({}) if action == "none" else parse_arguments(next(replies))
        for action in actions
    ]


def classify_captions(captions, action_list, action_dict):
    """Parse actions and their arguments from the captions, retrying failed LLM requests."""
    for attempt in range(1, LLM_ATTEMPTS + 1):
        try:
            actions = llm_parse_actions(captions, action_list)
            return actions, llm_parse_arguments(actions, captions, action_dict)
        except (requests.RequestException, ValueError) as exc:
            if attempt == LLM_ATTEMPTS:
                raise
            logger.warning("LLM request failed (attempt %d of %d): %s", attempt, LLM_ATTEMPTS, exc)


def group_predictions_with_dbscan(arguments, actions, eps=0.3, min_samples=1):
    """Cluster action/argument predictions by the cosine distance of their sentence embeddings.

    Returns {label: (member indices, representative index)} and the clustered predictions those
    indices refer to.
    """
    predictions = actionable_predictions(actions, arguments)
    if not predictions:
        return {}, []

    texts = [pair_text(p["action"], p["arguments"]) for p in predictions]
    logger.debug("Texts to cluster: %s", texts)
    embeddings = sentence_transformer_model.encode(texts)
    distances = np.clip(1 - cosine_similarity(embeddings), 0, None)
    labels = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed").fit_predict(distances)
    return cluster_representatives(labels, embeddings), predictions


def object_detection(video_path):
    """Objects tracked in a clip and the captions generated for each of them."""
    response = requests.post(
        f"{OBJECT_PROPOSER_URL}/process_video",
        json={"video_path": video_path},
        timeout=OBJECT_PROPOSER_TIMEOUT,
    )
    response.raise_for_status()
    data = response.json()
    return data.get("objects", []), data.get("captions", [])


def proc_one_video(video_path, action_list, action_dict):
    """Clustered action predictions for one clip, keyed by valid_parsed_actions_<n>."""
    objects, captions = object_detection(video_path)
    logger.info("Objects: %s", objects)
    captions = flatten_captions(objects, captions)
    if not captions:
        logger.info("No objects proposed for %s", video_path)
        return {}

    actions, arguments = classify_captions(captions, action_list, action_dict)
    logger.info("Parsed actions: %s", actions)
    logger.info("Parsed arguments: %s", arguments)
    clusters, predictions = group_predictions_with_dbscan(arguments, actions)
    if not predictions:
        logger.info("No valid actions detected in %s", video_path)
        return {}
    return build_predictions(clusters, predictions, captions)


def request_grounding(clip_path, pairs):
    """Queue one grounding job per subject/object pair and return the job ids."""
    with open(clip_path, "rb") as clip:
        response = requests.post(
            f"{GROUNDING_API_URL}/ground",
            data={"inputs": json.dumps({"subject_object_pairs": pairs})},
            files={"video": clip},
            timeout=GROUNDING_TIMEOUT,
        )
    response.raise_for_status()
    return response.json()


def analyze_clip(clip_path, action_list, action_dict):
    """Action predictions and grounding job ids for one clip.

    A failing service call is logged and reported in the "error" field, so one segment cannot
    fail the whole video or inherit another segment's results.
    """
    entry = {"results": {}, "api_response": []}
    try:
        entry["results"] = proc_one_video(str(clip_path), action_list, action_dict)
        pairs = subject_object_pairs(entry["results"])
        if pairs:
            entry["api_response"] = request_grounding(clip_path, pairs)
    except (requests.RequestException, ValueError) as exc:
        logger.exception("Analysis of %s failed", clip_path.name)
        entry["error"] = f"{type(exc).__name__}: {exc}"
    return entry


def write_clip(vr, start_frame, end_frame, fps, clip_path, thumbnail_path):
    """Write frames [start_frame, end_frame) to an mp4 clip and the first one to a thumbnail."""
    first_frame = vr[start_frame].asnumpy()
    cv2.imwrite(str(thumbnail_path), cv2.cvtColor(first_frame, cv2.COLOR_RGB2BGR))

    height, width, _ = first_frame.shape
    out = cv2.VideoWriter(str(clip_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    try:
        # Decoded segments can reach tens of GB, so frames are streamed in small batches.
        for chunk_start in range(start_frame, end_frame, CLIP_CHUNK_FRAMES):
            chunk_end = min(chunk_start + CLIP_CHUNK_FRAMES, end_frame)
            for frame in vr.get_batch(range(chunk_start, chunk_end)).asnumpy():
                out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        out.release()


def load_action_vocabulary():
    """(action, description, example) rows of the action CSV, and the same keyed by action."""
    action_df = pd.read_csv(ACTION_CATEGORIES_PATH, keep_default_na=False)
    action_list = list(
        zip(
            action_df["action_name"],
            action_df['"as in…"'],
            action_df["example sentence from propbank"],
        )
    )
    return action_list, {name: (usage, example) for name, usage, example in action_list}


def get_subj_obj_act_from_segments(segmentation_result, video_file_path):
    """Analyze every segment of the video.

    Returns the analyzed segments, in the shape of segmentation_result, and one result per
    segment in the same order. Segments that span no frame are dropped from both.
    """
    action_list, action_dict = load_action_vocabulary()
    vr, fps = open_video(video_file_path)

    analyzed, all_results = {}, []
    for video_id, segments in segmentation_result.items():
        video_basename = os.path.basename(video_id)
        analyzed[video_id] = []
        for idx, segment in enumerate(segments):
            start_frame, end_frame = frame_range(segment["segment"], fps, len(vr))
            if start_frame >= end_frame:
                logger.warning("Skipping segment %s: it spans no frame", segment["segment"])
                continue

            logger.info("Processing segment %d: %s s", idx, segment["segment"])
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
            thumbnail = f"thumbnail_{video_basename}_{idx}_{timestamp}.jpg"
            # The object proposer opens the clip by path, so it must live on the shared disk.
            clip_path = SHARED_DIR / f"segment_{uuid.uuid4().hex}.mp4"
            try:
                write_clip(vr, start_frame, end_frame, fps, clip_path, THUMBNAIL_DIR / thumbnail)
                entry = analyze_clip(clip_path, action_list, action_dict)
            finally:
                clip_path.unlink(missing_ok=True)
            logger.info("Segment %d results: %s", idx, entry)

            segment["thumbnail"] = thumbnail
            analyzed[video_id].append(segment)
            all_results.append({**entry, "thumbnail": thumbnail, "segment": segment})
    return analyzed, all_results


@celery_app.task(name="tasks.process_video_task")
def process_video_task(file_path, segmentation_type):
    """Segment a video uploaded to shared/, analyze every segment, then delete the upload."""
    # Relative paths are relative to this directory.
    upload = (BASE_DIR / file_path).resolve()
    # The upload is deleted however the task ends, so only files in shared/ are accepted.
    if upload.parent != SHARED_DIR.resolve():
        raise ValueError(f"{file_path!r} is not a file in the shared/ upload directory")
    video_path = str(upload)
    # Segments are keyed by the upload's name, as SingleVideoDataset names type-0 proposals,
    # so results never carry server paths.
    video_id = upload.stem
    try:
        if model is None:
            raise RuntimeError("models are not loaded; run `python celery_worker.py -d <gpu>`")
        mode = SegmentationType(segmentation_type)
        duration = get_video_duration_ffmpeg(video_path)
        logger.info("Video duration: %s seconds", duration)

        if mode is SegmentationType.WHOLE_VIDEO:
            proposals = {video_id: [{"segment": [0, duration], "score": 1}]}
        elif mode is SegmentationType.UNIFORM:
            proposals = {video_id: generate_intervals(duration)}
        else:
            proposals = segment_video(video_path)
        logger.info("Proposals: %s", proposals)

        proposals, results = get_subj_obj_act_from_segments(proposals, video_path)
    finally:
        upload.unlink(missing_ok=True)

    thumbnails = [
        {"thumbnail": r["thumbnail"], "segment": r["segment"]["segment"]} for r in results
    ]
    return {"proposals": proposals, "results": results, "thumbnails": thumbnails}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CReLeRI temporal segmentation worker")
    parser.add_argument("-d", "--device", type=int, default=0, help="GPU device ID to use")
    args = parser.parse_args()

    # Celery adds no handlers of its own once the root logger has one.
    coloredlogs.install(level="INFO", fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    celery_app.conf.update(worker_hijack_root_logger=False, worker_redirect_stdouts=False)

    initialize_globals(args.device)
    celery_app.worker_main(["worker", "--loglevel=info", "--pool=solo"])
