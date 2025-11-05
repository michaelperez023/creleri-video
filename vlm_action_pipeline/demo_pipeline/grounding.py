import copy
import functools
import itertools
import operator
import os
import re
import threading
from collections import defaultdict
from dataclasses import dataclass
from timeit import default_timer
from typing import Any, Iterator
from transformers import GenerationConfig
import xml.etree.ElementTree as ET

import depth_pro
import numpy as np
import torch
from decord import VideoReader
from decord.bridge import _BridgeScope
from numba import njit
from PIL import Image
from scipy.ndimage import zoom
from skimage.morphology import remove_small_holes, remove_small_objects
from transformers import pipeline
import grounding_multiprocessing as grounding_mp
from torch import multiprocessing
import hashlib, pathlib, json, subprocess, shlex
import inspect
import requests

import time, errno

LLAMA_PORT = 5003

os.environ.setdefault("TQDM_DISABLE", "1")

CACHE_ROOT = pathlib.Path("/tmp/sam2_cache")  # node-local
CACHE_ROOT.mkdir(exist_ok=True, parents=True)

def _video_hash(path):
    h = hashlib.sha1()
    # fast but decent: hash container + size + mtime; or hash first/last 1MB
    st = os.stat(path)
    h.update(f"{path}:{st.st_size}:{int(st.st_mtime)}".encode())
    return h.hexdigest()

def ensure_jpeg_dir(mp4_path, long_edge=1280):
    vid_hash = _video_hash(mp4_path)
    out_dir = CACHE_ROOT / vid_hash
    marker = out_dir / "done.json"
    lock   = out_dir / ".lock"

    out_dir.mkdir(parents=True, exist_ok=True)

    # fast path
    if marker.exists():
        return str(out_dir)

    # try to acquire lock (atomic create)
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        # someone else is extracting; wait for marker
        for _ in range(600):  # up to ~60s
            if marker.exists():
                return str(out_dir)
            time.sleep(0.1)
        raise RuntimeError(f"Timeout waiting for frame extraction in {out_dir}")
    else:
        # we hold the lock
        with os.fdopen(fd, "w") as f:
            f.write(str(os.getpid()))

        try:
            tmpl = str(out_dir / "%05d.jpg")
            cmd = (
                f"ffmpeg -hide_banner -loglevel error -nostats -y "
                f"-i {shlex.quote(mp4_path)} -q:v 2 "
                f"-vf \"scale='if(gt(iw,ih),{long_edge},-2)':'if(gt(iw,ih),-2,{long_edge})'\" "
                f"-start_number 0 {shlex.quote(tmpl)}"
            )
            subprocess.run(cmd, shell=True, check=True)

            # basic sanity
            imgs = list(out_dir.glob("*.jpg"))
            if not imgs or any(p.stat().st_size == 0 for p in imgs[:5]):
                # if something went wrong, clean up so next try can regenerate
                for p in out_dir.glob("*.jpg"):
                    p.unlink(missing_ok=True)
                raise RuntimeError("Frame extraction produced no/empty JPEGs")

            marker.write_text(json.dumps({"src": mp4_path, "long_edge": long_edge}))
            return str(out_dir)
        finally:
            try:
                os.unlink(lock)
            except FileNotFoundError:
                pass

# Fix to get GroundingDINO to build correctly
conda_prefix = os.environ.get("CONDA_PREFIX")
if conda_prefix is not None and "CUDA_HOME" not in os.environ:
    os.environ["CUDA_HOME"] = conda_prefix
    os.environ["CUDA_INC_PATH"] = os.path.join(
        conda_prefix, "targets/x86_64-linux/include"
    )

@dataclass
class Result:
    done: threading.Event
    exception: Exception | None = None
    result: Any = None

@njit(
    "(boolean[:, :], boolean[:, :], float32[:, :], boolean[:, :],  int32, float32)",
    nogil=True,
)
def _find_contact_impl(
    subject_mask, object_mask, depth, contact, radius, depth_threshold
):
    for x in range(subject_mask.shape[0]):
        for y in range(subject_mask.shape[1]):
            if not subject_mask[x, y] or contact[x, y]:
                continue
            for i in range(x - radius, x + radius + 1):
                for j in range(y - radius, y + radius + 1):
                    if i < 0 or i >= subject_mask.shape[0]:
                        continue
                    if j < 0 or j >= subject_mask.shape[1]:
                        continue
                    if not object_mask[i, j]:
                        continue
                    if np.sqrt((x - i) ** 2 + (y - j) ** 2) > radius:
                        continue

                    if abs(depth[x, y] - depth[i, j]) <= depth_threshold:
                        contact[x, y] = True
                        contact[i, j] = True


def has_small_holes_and_islands(
    mask, small_hole_area_divider, small_hole_area_threshold, n_frames
):
    """
    Sample `self.small_hole_n_frames` frames of the mask and check if it has
    too many small holes or islands.
    """
    indices = SubjectObjectGrounding.sample_indices(mask.shape[0], n_frames)
    _, h, w = mask.shape
    area = h * w / small_hole_area_divider
    for i in indices:
        frame = mask[i]
        frame_sum = frame.sum()
        if frame_sum == 0.0:
            continue
        filtered = remove_small_holes(remove_small_objects(frame, area), area)
        diff = (filtered != frame).sum() / frame_sum
        if diff > small_hole_area_threshold:
            return True
    return False


def molmo_inference(args):
    image, text = args[:2]
    image = Image.fromarray(image.numpy())

    model = grounding_mp.models["molmo_model"]
    processor = grounding_mp.models["molmo_processor"]

    # process the image and text
    inputs = processor.process(images=[image], text=text)

    # move inputs to the correct device and make a batch of size 1
    inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}

    # generate output; maximum 200 new tokens; stop generation when <|endoftext|> is generated
    with torch.inference_mode(), torch.autocast(
        device_type="cuda", enabled=True, dtype=torch.bfloat16
    ):
        output = model.generate_from_batch(
            inputs,
            GenerationConfig(max_new_tokens=200, stop_strings="<|endoftext|>"),
            tokenizer=processor.tokenizer,
        )

    # only get generated tokens; decode them to text
    generated_tokens = output[0, inputs["input_ids"].size(1) :]
    generated_text = processor.tokenizer.decode(
        generated_tokens, skip_special_tokens=True
    )
    return generated_text

def _quiet_kwargs(f):
    # Add any commonly-seen flags across versions
    sig = inspect.signature(f)
    kw = {}
    for k, v in [
        ("verbose", False),
        ("progress", False),
        ("show_progress", False),
        ("disable_tqdm", True),
    ]:
        if k in sig.parameters:
            kw[k] = v
    return kw

def _init_state_compat(pred, video_dir):
    f = pred.init_state
    return f(video_dir, **_quiet_kwargs(f))

def _propagate_compat(pred, st, *, reverse=False):
    f = pred.propagate_in_video
    kw = _quiet_kwargs(f)
    if "reverse" in inspect.signature(f).parameters:
        kw["reverse"] = reverse
    return f(st, **kw)

def track_items(
    args: tuple[str, list[tuple[int, dict]], int, int, float, int]
):
    (
        video,
        detections,
        tracking_max_edge,
        small_hole_area_divider,
        small_hole_area_threshold,
        n_frames,
    ) = args
    predictor = grounding_mp.models["sam_model"]

    detections_per_frame = defaultdict(list)
    for frame_idx, det in detections:          # det is a dict from GroundingDINO
        detections_per_frame[frame_idx].append(det)

    item_masks = defaultdict(list)
    with (
        torch.inference_mode(),
        torch.autocast("cuda", dtype=torch.bfloat16),
        torch.cuda.device(predictor.device),
    ):
        video_for_sam2 = ensure_jpeg_dir(video, long_edge=tracking_max_edge)
        state = _init_state_compat(predictor, video_for_sam2)

        # Compute scale from original MP4 → SAM2 JPEG pixels
        vr0 = VideoReader(video)
        orig_h, orig_w = vr0[0].shape[:2]
        sam_w = int(state["video_width"])
        sam_h = int(state["video_height"])
        sx = sam_w / float(orig_w)
        sy = sam_h / float(orig_h)

        for frame_idx, dets in detections_per_frame.items():
            predictor.reset_state(state)

            # if something odd ever yields an empty list, just skip
            if not dets:
                continue

            for i, det in enumerate(dets):
                b = det["box"]
                box = np.array([b["xmin"], b["ymin"], b["xmax"], b["ymax"]], dtype=np.float32)

                # rescale to SAM2 JPEG coords
                box[[0, 2]] *= sx
                box[[1, 3]] *= sy

                # skip degenerate boxes (can otherwise result in no effective points)
                if box[2] <= box[0] or box[3] <= box[1]:
                    continue

                box[[0, 2]] = np.clip(box[[0, 2]], 0, sam_w - 1)
                box[[1, 3]] = np.clip(box[[1, 3]], 0, sam_h - 1)

                predictor.add_new_points_or_box(state, frame_idx=frame_idx, obj_id=i, box=box)

            # DO NOT re-init here — just propagate on the current state
            forward  = _propagate_compat(predictor, state, reverse=False)
            backward = _propagate_compat(predictor, state, reverse=True)
            backward = (out for out in backward
                        if isinstance(out, (list, tuple)) and out and out[0] != frame_idx)
            results = itertools.chain(forward, backward)

            segments = [list() for _ in dets]
            for out_frame_idx, out_obj_ids, out_mask_logits in results:
                for j, out_obj_id in enumerate(out_obj_ids):
                    mask = (out_mask_logits[j] > 0.0).cpu()
                    segments[out_obj_id].append((out_frame_idx, mask))

            for det, segment_frames in zip(dets, segments):
                if segment_frames:
                    mask = SubjectObjectGrounding._merge_masks(segment_frames)
                    # filter obviously-bad tracks
                    if has_small_holes_and_islands(
                        mask.numpy(),
                        small_hole_area_divider=small_hole_area_divider,
                        small_hole_area_threshold=small_hole_area_threshold,
                        n_frames=n_frames,
                    ):
                        continue
                    item_masks[det["label"]].append((mask, det["score"]))

    return item_masks


class SubjectObjectGrounding:
    def __init__(
        self,
        devices: list = [f"cuda{i}" for i in range(8)],
        grounding_threshold=0.3,
        grounding_n_frames: int = 5,
        tracking_max_edge: int = 1280,
        deduplicate_max_edge: int = 360,
        deduplicate_iou_threshold: float = 0.9,
        small_hole_area_divider: int = 5000,
        small_hole_n_frames: int = 5,
        small_hole_area_threshold: float = 0.02,
        depth_pro_ckpt="checkpoints/depth_pro.pt",
        contact_detection_n_frames: int = 5,
        contact_detection_radius_divider: int = 72,
        contact_detection_depth_threshold: float = 0.1,
        contact_detection_noise_area_divider: int = 1350,
        contact_detection_max_edge: int = 720,
        mask_video_mask_edge: int = 1280,
        mask_video_target_fps: int = 10,
        mask_video_max_frames: int = 10,
        max_mask_pairs_to_check: int = 10
    ) -> None:
        self.grounding_threshold = grounding_threshold
        self.grounding_n_frames = grounding_n_frames

        self.tracking_max_edge = tracking_max_edge

        self.deduplicate_max_edge = deduplicate_max_edge
        self.deduplicate_iou_threshold = deduplicate_iou_threshold

        self.small_hole_area_divider = small_hole_area_divider
        self.small_hole_n_frames = small_hole_n_frames
        self.small_hole_area_threshold = small_hole_area_threshold

        self.depth_pro_ckpt = depth_pro_ckpt

        self.contact_detection_n_frames = contact_detection_n_frames
        self.contact_detection_radius_divider = contact_detection_radius_divider
        self.contact_detection_depth_threshold = contact_detection_depth_threshold
        self.contact_detection_noise_area_divider = contact_detection_noise_area_divider
        self.contact_detection_max_edge = contact_detection_max_edge

        self.mask_video_mask_edge = mask_video_mask_edge
        self.mask_video_target_fps = mask_video_target_fps
        self.mask_video_max_frames = mask_video_max_frames

        self.max_mask_pairs_to_check = max_mask_pairs_to_check

        assert len(devices) >= 2, "needs at least 2 GPUs"
        default_device, *shared_devices = devices

        # Worker pool (SAM2 & Molmo)
        self.pool_size = len(shared_devices)
        mp = multiprocessing.get_context("spawn")

        queue = mp.Queue()
        for device in shared_devices:
            queue.put(device)

        worker_ready = mp.Barrier(self.pool_size + 1)

        self.pool = mp.get_context("spawn").Pool(
            self.pool_size,
            initializer=grounding_mp.init,
            initargs=(queue, worker_ready),
        )

        # Other models (main process)
        depth_pro_config = copy.copy(depth_pro.depth_pro.DEFAULT_MONODEPTH_CONFIG_DICT)
        depth_pro_config.checkpoint_uri = self.depth_pro_ckpt
        depth_model, depth_transform = depth_pro.create_model_and_transforms(
            config=depth_pro_config, device=default_device
        )
        depth_model.eval()
        self.depth_model = depth_model
        self.depth_transform = depth_transform

        self.object_detector = pipeline(
            model="IDEA-Research/grounding-dino-base",
            task="zero-shot-object-detection",
            device=default_device,
        )

        worker_ready.wait()

    def shutdown(self):
        self.pool.close()
        self.pool.join()

    @staticmethod
    def sample_indices(total, n_samples):
        n_samples = min(total, n_samples)
        return np.linspace(0, total - 1, n_samples, dtype=np.int64)
    
    @staticmethod
    def load_frames(video, n_frames):
        vr = VideoReader(video)
        total = len(vr)
        indices = SubjectObjectGrounding.sample_indices(total, n_frames)
        frames = [Image.fromarray(vr[int(i)].asnumpy()) for i in indices]
        return frames, indices
    
    def _llama_rewrite_point_to_prompts(self, item_and_captions):
        url = f"http://localhost:{LLAMA_PORT}/llama33"
        msgs = []
        for item, caption in item_and_captions:
            msgs.append([
                {"role":"system","content":
                 "Rewrite into one short imperative that starts with 'Point to' and mentions the given item; "
                 "optionally include a few relevant words from the context. Output one line only."},
                {"role":"user","content": f'item="{item}"\ncontext="{caption}"'}
            ])
        payload = {"messages": msgs, "is_batch": True}
        r = requests.post(url, json=payload, timeout=60)
        r.raise_for_status()
        return r.json()["response"]

    @staticmethod
    def _parse_molmo_points(text):
        m = re.search(r"<points?\b[^>]*>.*?</points?>", text)
        if not m:
            return []

        try:
            tag = ET.fromstring(m.group(0))
        except ET.ParseError:
            return []

        points = []
        try:
            if tag.tag == "point":
                points.append((tag.attrib["x"], tag.attrib["y"]))
            elif tag.tag == "points":
                attrs = {k: v for k, v in tag.attrib.items() if k and k[0] in "xy"}
                if len(attrs) % 2 != 0:
                    return []
                for i in range(len(attrs) // 2):
                    points.append((attrs[f"x{i+1}"], attrs[f"y{i+1}"]))
        except KeyError:
            return []

        try:
            points = [(float(x), float(y)) for x, y in points]
        except ValueError:
            return []

        return points

    def ground_items_grounding_dino(self, video: str, items: list[str]):
        frames, indices = self.load_frames(video, self.grounding_n_frames)

        if not items:
            return [], frames, indices

        detections = []
        unique_items = sorted(set(items))

        # GroundingDINO expect a period at the end of the label
        candidates = {}
        for item in unique_items:
            label = item.removesuffix(".") + "."
            candidates[label] = item

        for frame, i in zip(frames, indices):
            for detection in self.object_detector(
                frame,
                candidate_labels=list(candidates.keys()),
                threshold=self.grounding_threshold,
            ):
                detection["label"] = candidates[detection["label"]]
                detections.append((i, detection))

        return detections, frames, indices

    def ground_items_molmo(
        self,
        item_and_captions: list[tuple[str, str]],
        frames,
        frame_indices,
    ):
        if not item_and_captions:
            return []

        prompts = self._llama_rewrite_point_to_prompts(item_and_captions)

        inputs = []
        for frame, frame_index in zip(frames, frame_indices):
            for (name, _), prompt in zip(item_and_captions, prompts):
                inputs.append((frame, prompt, frame_index, name))

        args = [
            (torch.as_tensor(np.array(frame)), prompt, {})
            for frame, prompt, _, _ in inputs
        ]
        results = self.pool.imap(molmo_inference, args)

        detections = []
        for result, grounding_input in zip(results, inputs):
            *_, frame_index, name = grounding_input
            points = self._parse_molmo_points(result)
            if points:
                points = np.array(points) / 100
                detections.append((name, frame_index, points))

        return detections

    def merge_detections(self, dino_detections, molmo_detections, frame_size):
        def match_detection(dino_detection, points):
            box = dino_detection["box"]
            xmin = box["xmin"]
            ymin = box["ymin"]
            xmax = box["xmax"]
            ymax = box["ymax"]
            points = points * frame_size

            hits = sum(int(xmin <= x <= xmax and ymin <= y <= ymax) for x, y in points)
            area = (xmax - xmin) * (ymax - ymin)
            return hits, area

        dino_detection_groups = defaultdict(list)
        for frame_idx, detection in dino_detections:
            dino_detection_groups[(frame_idx, detection["label"])].append(detection)

        results = []
        for name, frame_idx, points in molmo_detections:
            detections = dino_detection_groups.get((frame_idx, name))
            if not detections:
                continue
            candidates = []
            for detection in detections:
                hits, area = match_detection(detection, points)
                if not hits:
                    continue
                candidates.append(((hits, -area), detection))
            if candidates:
                _, best_candidate = max(candidates, key=lambda p: p[0])
                results.append((frame_idx, best_candidate))

        return results

    @staticmethod
    def _merge_masks(masks):
        masks.sort(key=lambda m: m[0])
        indices, masks = zip(*masks)
        assert (
            len(set(indices)) == len(indices)
            and indices[0] == 0
            and indices[-1] + 1 == len(indices)
        )
        return torch.concatenate(masks, dim=0)

    @staticmethod
    def iou(a, b):
        return (a & b).sum() / (a | b).sum()

    @staticmethod
    def resize_video(video, max_edge, order=1):
        _, h, w = video.shape
        long_edge = max(h, w)

        if long_edge <= max_edge:
            return video

        z = max_edge / long_edge
        return zoom(video, zoom=(1.0, z, z), order=order)

    def has_small_holes_and_islands(self, mask):
        """
        Sample `self.small_hole_n_frames` frames of the mask and check if it has
        too many small holes or islands.
        """
        indices = self.sample_indices(mask.shape[0], self.small_hole_n_frames)
        _, h, w = mask.shape
        area = h * w / self.small_hole_area_divider
        for i in indices:
            frame = mask[i]
            frame_sum = frame.sum()
            if frame_sum == 0.0:
                continue
            filtered = remove_small_holes(remove_small_objects(frame, area), area)
            diff = (filtered != frame).sum() / frame_sum
            if diff > self.small_hole_area_threshold:
                return True
        return False

    def get_unique_masks(self, scores_and_masks):
        sorted_masks = sorted(scores_and_masks, key=lambda s: s[1], reverse=True)

        unique_masks = []
        for mask, score in sorted_masks:
            # Down-sample the mask to make iou computation, etc. faster
            resized_mask = self.resize_video(mask, self.deduplicate_max_edge)
            if resized_mask.sum() == 0.0:
                continue
            for existing_mask, _, _ in unique_masks:
                if (
                    self.iou(resized_mask, existing_mask)
                    > self.deduplicate_iou_threshold
                ):
                    break
            else:
                unique_masks.append((resized_mask, mask, score))
        return [(m, s) for _, m, s in unique_masks]

    def get_depth_estimations(self, video):
        frames, indices = self.load_frames(video, self.contact_detection_n_frames)
        depth_estimations = {}
        for i, frame in zip(indices, frames):
            image = self.depth_transform(frame)
            with torch.inference_mode():
                prediction = self.depth_model.infer(image)
            depth = prediction["depth"]
            depth_estimations[i] = depth.cpu()
        return depth_estimations

    def track_items(self, video, detections):
        if not detections:
            return {}

        detections.sort(key=lambda p: p[0])

        world_size = self.pool_size
        job_per_rank = len(detections) // world_size
        remainder = len(detections) % world_size

        inputs = []
        start = 0
        for i in range(world_size):
            job_count = job_per_rank + int(i < remainder)
            if job_count > 0:
                batch = detections[start : start + job_count]
                inputs.append(
                    (
                        video,
                        batch,
                        self.tracking_max_edge,
                        self.small_hole_area_divider,
                        self.small_hole_area_threshold,
                        self.small_hole_n_frames,
                    )
                )
            start += job_count

        assert start == len(detections)

        item_masks = defaultdict(list)
        for partial_results in self.pool.imap(track_items, inputs):
            for key, values in partial_results.items():
                for mask, dino_score in values:
                    item_masks[key].append((mask.numpy(), dino_score))

        unique_masks = {}
        for name, scores_and_masks in item_masks.items():
            masks = self.get_unique_masks(scores_and_masks)
            if not masks:
                continue
            unique_masks[name] = masks
        return unique_masks

    def _find_contact_points(
        self,
        subject_mask,
        object_mask,
        depth_estimations,
    ):
        subject_mask = np.array(subject_mask, dtype=bool, copy=False)
        object_mask = np.array(object_mask, dtype=bool, copy=False)
        subject_mask = self.resize_video(subject_mask, self.contact_detection_max_edge)
        object_mask = self.resize_video(object_mask, self.contact_detection_max_edge)
        _, mask_h, mask_w = subject_mask.shape

        contacts = {}
        for frame_idx, depth in depth_estimations.items():
            depth_h, depth_w = depth.shape
            if depth_h != mask_h or depth_w != mask_w:
                depth = zoom(depth, (mask_h / depth_h, mask_w / depth_w), order=1)
            depth = np.array(depth, dtype=np.float32, copy=False)

            # remove the overlapping parts
            object_mask = object_mask & ~subject_mask
            radius = round(max(mask_h, mask_w) / self.contact_detection_radius_divider)

            contact = np.zeros_like(subject_mask[frame_idx])
            _find_contact_impl(
                subject_mask[frame_idx],
                object_mask[frame_idx],
                depth,
                contact,
                radius,
                self.contact_detection_depth_threshold,
            )

            noise_area = mask_h * mask_w / self.contact_detection_noise_area_divider
            contact = remove_small_objects(contact, min_size=noise_area)
            contact = remove_small_holes(contact, area_threshold=noise_area)
            contacts[frame_idx] = contact

        return contacts

    def _score_masks(self, item_masks, molmo_detections):
        # Score each mask by the % of Molmo points it hits.
        detections_by_name = defaultdict(list)
        for name, frame_idx, points in molmo_detections:
            for point in points:
                detections_by_name[name].append((frame_idx, point))

        results = {}
        for name, masks in item_masks.items():
            scored_masks = []
            for mask, dino_score in masks:
                h, w = mask.shape[1:]
                score = 0
                total_points = len(detections_by_name[name])
                for frame_idx, (x, y) in detections_by_name[name]:
                    x = min(round(x * w), w - 1)
                    y = min(round(y * h), h - 1)
                    score += int(mask[frame_idx, y, x]) / total_points
                # Use the score from GroundingDINO as a tie-breaker
                scored_masks.append((mask, score, dino_score))
            results[name] = scored_masks
        return results

    def find_best_subject_object_pair(
        self,
        video,
        depth_estimations,
        dino_detections,
        dino_frames,
        dino_frame_indices,
        argument_names,
        caption,
    ):
        extra_info = {}

        if not argument_names:
            extra_info["rejection_reason"] = "no valid argument names"
            return None, extra_info

        detected_arguments = set(detection["label"] for _, detection in dino_detections)
        missing = list(set(argument_names) - detected_arguments)
        if missing:
            extra_info["rejection_reason"] = (
                f"{missing!r} in {caption!r} not found in grounding (GroundingDINO)"
            )
            return None, extra_info

        item_and_captions = [(name, caption) for name in argument_names]
        grounding_molmo_time = -default_timer()
        molmo_detections = self.ground_items_molmo(
            item_and_captions, dino_frames, dino_frame_indices
        )
        grounding_molmo_time += default_timer()
        extra_info["grounding_molmo_time"] = grounding_molmo_time

        detections = self.merge_detections(
            dino_detections, molmo_detections, dino_frames[0].size
        )

        detected_arguments = set(detection["label"] for _, detection in detections)
        missing = list(set(argument_names) - detected_arguments)
        if missing:
            extra_info["rejection_reason"] = (
                f"{missing!r} in {caption!r} not found in grounding (Molmo)"
            )
            return None, extra_info

        tracking_time = -default_timer()
        item_masks = self.track_items(video, detections)
        tracking_time += default_timer()
        extra_info["tracking_time"] = tracking_time

        missing = list(set(argument_names) - set(item_masks.keys()))
        if missing:
            extra_info["rejection_reason"] = (
                f"{missing!r} in {caption!r} not found in tracking"
            )
            return None, extra_info

        masks_with_scores = self._score_masks(item_masks, molmo_detections)

        prepare_candidate_time = -default_timer()
        candidates = []
        for combination in itertools.product(
            *(masks_with_scores[name] for name in argument_names)
        ):
            masks, molmo_scores, dino_scores = zip(*combination)
            # If the mask of one argument is almost the same as another
            for i, mask_a in enumerate(masks):
                for mask_b in masks[i + 1 :]:
                    if self.iou(mask_a, mask_b) > self.deduplicate_iou_threshold:
                        continue
            # Again, the GroundingDINO score is the tie-breaker
            candidates.append((masks, np.prod(molmo_scores), np.prod(dino_scores)))
        candidates.sort(key=lambda p: p[1:], reverse=True)
        candidates = candidates[: self.max_mask_pairs_to_check]
        prepare_candidate_time += default_timer()
        extra_info["prepare_candidate_time"] = prepare_candidate_time

        if not candidates:
            extra_info["rejection_reason"] = (
                "can't find valid combination of argument masks"
            )
            return None, extra_info

        contact_time = -default_timer()
        best_mask, _, _ = candidates[0]
        best_contacts = None
        has_contact = False
        for masks, _, _ in candidates:
            has_contact, contacts = self._check_argument_contact(
                depth_estimations, masks
            )
            if has_contact:
                best_mask = masks
                best_contacts = contacts
                break
        contact_time += default_timer()
        extra_info["contact_time"] = contact_time

        if not has_contact:
            extra_info["rejection_reason"] = "arguments are not in contact"

        return (best_mask, has_contact, best_contacts), extra_info

    def _check_argument_contact(self, depth_estimations, masks):
        all_contacts = []

        if len(masks) < 2:
            return True, all_contacts

        mask_union = functools.reduce(operator.or_, masks)
        for mask in masks:
            rest = mask_union & (~mask)
            contacts = self._find_contact_points(mask, rest, depth_estimations)
            has_contact = any(contact.sum() for contact in contacts.values())
            if not has_contact:
                return False, all_contacts
            all_contacts.append(contacts)

        return True, all_contacts

    def __call__(
        self, video, pair_proposals
    ) -> Iterator[tuple[tuple[np.ndarray, np.ndarray, np.ndarray] | None, dict]]:
        extra_info = {}
        depth_time = -default_timer()
        depth_estimations = self.get_depth_estimations(video)
        depth_time += default_timer()
        extra_info["shared_depth_time"] = depth_time

        unique_arguments = set()
        for argument_names, _ in pair_proposals:
            unique_arguments.update(argument_names)
        unique_arguments = sorted(unique_arguments)

        shared_grounding_dino_time = -default_timer()
        dino_detections, dino_frames, dino_frame_indices = (
            self.ground_items_grounding_dino(video, unique_arguments)
        )
        shared_grounding_dino_time += default_timer()
        extra_info["shared_grounding_dino_time"] = shared_grounding_dino_time

        for argument_names, caption in pair_proposals:
            result, filter_info = self.find_best_subject_object_pair(
                video,
                depth_estimations,
                dino_detections,
                dino_frames,
                dino_frame_indices,
                argument_names,
                caption,
            )
            yield result, {**extra_info, **filter_info}
