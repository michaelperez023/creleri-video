"""Pure helpers for the segmentation service.

Segment selection, time-to-frame mapping and parsing of LLM replies. They only need the
standard library and numpy, so they can be tested without the GPU stack.
"""

import json
from enum import IntEnum
from itertools import pairwise

import numpy as np

ARGUMENT_KEYS = ("argument0", "argument1", "argument2")
_EMPTY_VALUES = {"", "none", "null", "n/a"}


class SegmentationType(IntEnum):
    """How a video is split before each segment is analyzed."""

    MODEL = 0  # AdaTAD proposals refined by scene cuts
    UNIFORM = 1  # fixed 5-second windows
    WHOLE_VIDEO = 2  # one segment spanning the video


def segments_overlap(a, b):
    return max(a[0], b[0]) < min(a[1], b[1])


def select_non_overlapping_proposals(proposals):
    """Greedily keep the highest-scoring proposals that overlap none of those already kept."""
    selected = []
    for prop in sorted(proposals, key=lambda p: p["score"], reverse=True):
        if not any(segments_overlap(prop["segment"], sel["segment"]) for sel in selected):
            selected.append({"segment": prop["segment"], "score": prop["score"]})
    return selected


def select_proposals(proposals, min_score=0.001, max_candidates=2000, max_segments=20):
    """Pick up to max_segments non-overlapping proposals among the best-scoring candidates."""
    candidates = sorted(
        (p for p in proposals if p["score"] > min_score), key=lambda p: p["score"], reverse=True
    )
    return select_non_overlapping_proposals(candidates[:max_candidates])[:max_segments]


def adjust_segments_to_scene_changes(proposals, scene_changes, video_duration, min_duration=2):
    """Cut the video at every scene change, then split each scene at proposal boundaries.

    A boundary only splits a scene when it leaves at least min_duration seconds on both sides.
    """
    scenes = []
    last_end = 0.0
    for change in sorted(scene_changes):
        scenes.append((last_end, change))
        last_end = change
    if last_end < video_duration:
        scenes.append((last_end, video_duration))

    proposal_times = sorted(t for p in proposals for t in p["segment"])
    segments = []
    for start, end in scenes:
        split_points = [start]
        for t in proposal_times:
            if start < t < end and t - split_points[-1] >= min_duration and end - t >= min_duration:
                split_points.append(t)
        split_points.append(end)
        segments.extend({"segment": [a, b], "score": 0.0} for a, b in pairwise(split_points))
    return segments


def generate_intervals(duration, interval_length=5):
    intervals = []
    start = 0.0
    while start < duration:
        end = min(start + interval_length, duration)
        intervals.append({"segment": [start, end], "score": 1})
        start = end
    return intervals


def frame_range(segment, fps, total_frames):
    """Map a [start, end] span in seconds to a half-open frame range clamped to the video."""
    start, end = segment
    # Scene cuts are stored rounded to 0.01 s, so rounding (not truncating) maps each boundary
    # back to the frame it was detected on.
    return max(0, round(start * fps)), min(total_frames, round(end * fps))


def flatten_captions(objects, captions):
    """One caption string per (object, caption) pair.

    Each object's captions may be a single string or a list, possibly nested one level.
    """
    flat = []
    for _, object_captions in zip(objects, captions):
        if not isinstance(object_captions, list):
            object_captions = [object_captions]
        for caption in object_captions:
            flat.extend(str(c) for c in (caption if isinstance(caption, list) else [caption]))
    return flat


def parse_json_object(text):
    """Return the first JSON object in text, skipping any prose or code fences around it."""
    decoder = json.JSONDecoder()
    start = text.find("{")
    while start != -1:
        try:
            return decoder.raw_decode(text, start)[0]
        except json.JSONDecodeError:
            start = text.find("{", start + 1)
    return None


def _as_text(value):
    if value is None:
        return ""
    if isinstance(value, list):
        return ", ".join(filter(None, map(_as_text, value)))
    return str(value).strip()


def normalize_arguments(obj):
    """argument0..argument2 as strings, with "none" for missing, null or empty values."""
    arguments = {}
    for key in ARGUMENT_KEYS:
        text = _as_text(obj.get(key))
        arguments[key] = "none" if text.lower() in _EMPTY_VALUES else text
    return arguments


def parse_arguments(reply):
    """Normalized arguments from one LLM reply; all "none" when it holds no JSON object."""
    return normalize_arguments(parse_json_object(reply) or {})


def prune_arguments(arguments):
    """Drop "none" arguments, moving argument2 into argument1 when only argument1 is missing."""
    arguments = dict(arguments)
    if arguments.get("argument1") == "none" and arguments.get("argument2", "none") != "none":
        arguments["argument1"] = arguments.pop("argument2")
    return {key: value for key, value in arguments.items() if value != "none"}


def actionable_predictions(actions, arguments):
    """Predictions whose action is not "none", with their input position and pruned arguments."""
    return [
        {"original_idx": idx, "action": action, "arguments": prune_arguments(args)}
        for idx, (action, args) in enumerate(zip(actions, arguments))
        if action.lower() != "none"
    ]


def pair_text(action, arguments):
    """Sentence embedded for clustering: subject, action, then the remaining arguments."""
    subject = arguments.get("argument0")
    others = [value for key, value in arguments.items() if key != "argument0"]
    return " ".join(([subject] if subject else []) + [action] + others)


def cluster_representatives(labels, embeddings):
    """Map each cluster label to (member indices, index of the member closest to the mean)."""
    clusters = {}
    for idx, label in enumerate(labels):
        if label != -1:
            clusters.setdefault(label, []).append(idx)

    representatives = {}
    for label, indices in clusters.items():
        members = embeddings[indices]
        distances = np.linalg.norm(members - members.mean(axis=0), axis=1)
        representatives[label] = (indices, indices[int(np.argmin(distances))])
    return representatives


def build_predictions(clusters, predictions, captions):
    """Group cluster representatives under "valid_parsed_actions_<n>" keys.

    n counts the action and its arguments; probability is the cluster's share of all clustered
    predictions.
    """
    total = sum(len(indices) for indices, _ in clusters.values())
    grouped = {}
    for indices, representative in clusters.values():
        prediction = predictions[representative]
        pattern = {"action": prediction["action"], **prediction["arguments"]}
        grouped.setdefault(f"valid_parsed_actions_{len(pattern)}", []).append(
            {
                "pattern": pattern,
                "probability": len(indices) / total if total > 0 else 0,
                "caption": captions[prediction["original_idx"]],
            }
        )
    return {key: grouped[key] for key in sorted(grouped, key=lambda k: int(k.rsplit("_", 1)[1]))}


def subject_object_pairs(predictions):
    """Grounding requests for every predicted pattern, in the order job ids are matched back."""
    pairs = []
    for items in predictions.values():
        for item in items:
            pattern = item["pattern"]
            pairs.append(
                {
                    "argument_names": [
                        pattern[key] for key in ARGUMENT_KEYS if pattern.get(key, "none") != "none"
                    ],
                    "action": pattern.get("action", "none"),
                    "caption": item["caption"] or "",
                }
            )
    return pairs
