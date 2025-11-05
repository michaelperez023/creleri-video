import argparse
import asyncio
from dataclasses import dataclass
import tempfile
import uuid

import mediapy as media
import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Response
from grounding import SubjectObjectGrounding
from pydantic import BaseModel, BeforeValidator, AfterValidator, field_serializer
from typing_extensions import Annotated
from enum import Enum
from timeit import default_timer
import cv2
from io import BytesIO
import os, concurrent.futures
import pathlib
import time
import json
import traceback

UPLOAD_ROOT = os.environ.get(
    "GROUNDING_UPLOAD_DIR",
    os.path.join(tempfile.gettempdir(), "grounding-uploads")
)
pathlib.Path(UPLOAD_ROOT).mkdir(parents=True, exist_ok=True)

CONTACT_REQUIRED_ACTIONS: set[str] = set()

def load_contact_required_actions(path: str | None) -> set[str]:
    """
    One action verb per line; case-insensitive; '#' starts a comment.
    Blank lines ignored.
    """
    actions = set()
    if not path:
        return actions
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.split("#", 1)[0].strip().lower()
                if line:
                    actions.add(line)
    except FileNotFoundError:
        print(f"[grounding] contact-required file not found: {path} (continuing with empty set)")
    return actions

def requires_contact(action: str) -> bool:
    return (action or "").strip().lower() in CONTACT_REQUIRED_ACTIONS

def _write_upload_atomic(root_dir: str, suffix: str, data: bytes) -> str:
    file_id = uuid.uuid4().hex
    final_path = os.path.join(root_dir, f"upload_{file_id}{suffix if suffix.startswith('.') else '.'+suffix}")
    tmp_path = final_path + ".part"

    with open(tmp_path, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())

    # atomic rename in same dir
    os.replace(tmp_path, final_path)
    return final_path

def _wait_until_exists(path: str, tries: int = 5, delay: float = 0.05) -> bool:
    for _ in range(tries):
        if os.path.exists(path):
            return True
        time.sleep(delay)
    return False

ground = None
app = FastAPI()

result_counter = 0
result_time_to_live = 100

THREAD_POOL = concurrent.futures.ThreadPoolExecutor(
    max_workers=max(2, min(8, (os.cpu_count() or 4)))  # tune as you like
)

def _build_artifacts(path, argument_masks, argument_names, action, contacts):
    t = -default_timer()

    # 1) draw overlays (OpenCV) — CPU heavy
    args = list(zip(argument_masks, argument_names))
    frames, fps = visualize_subject_object(path, args, action, contacts)

    # 2) encode to mp4 — CPU heavy
    video_bytes = media.compress_video(frames, fps=fps)

    # 3) package masks — CPU heavy if large arrays
    with BytesIO() as buf:
        np.savez_compressed(buf, **{f"arg{i}": m for i, m in enumerate(argument_masks)})
        masks_bytes = buf.getvalue()

    t += default_timer()
    return video_bytes, masks_bytes, t

class SubjectObjectPair(BaseModel):
    argument_names: list[str]
    action: str
    caption: str

class GroundingInputs(BaseModel):
    filename: str = ".mp4"
    subject_object_pairs: list[SubjectObjectPair]

def validate_inputs(string):
    return GroundingInputs.model_validate_json(string)

class JobStatus(Enum):
    PENDING = 0
    SUCCESS_VALID = 1
    SUCCESS_INVALID = 2
    EXCEPTION = 3
    CANCELLED = 4

@dataclass
class Job:
    id: uuid.UUID
    done: asyncio.Event
    status: JobStatus = JobStatus.PENDING
    annotated_video: bytes | None = None
    argument_masks: bytes | None = None
    extra_info: dict | None = None
    result_index: int | None = None

class JobInfo(BaseModel):
    id: uuid.UUID
    status: JobStatus
    has_visualization: bool
    extra_info: dict | None

    @field_serializer("status")
    def serialize_status(self, status: JobStatus, _info):
        return status.name

ongoing_jobs: dict[uuid.UUID, Job] = {}

def validate_job_id(job_id: uuid.UUID) -> uuid.UUID:
    if job_id not in ongoing_jobs:
        raise HTTPException(404, f"Job {job_id} not found")
    return job_id

JobID = Annotated[uuid.UUID, AfterValidator(validate_job_id)]

def visualize_frame(frame, masks, colors, max_edge, text):
    long_edge = max(frame.shape[:2])
    if long_edge > max_edge:
        scale = max_edge / long_edge
        frame = cv2.resize(frame, dsize=None, fx=scale, fy=scale)
    frame_h, frame_w = frame.shape[:2]

    line_width = max(1.5 * max(frame.shape[:2]) / 1280, 2.0)

    for mask, color in zip(masks, colors):
        mask = np.array(mask, dtype=np.uint8, copy=True)
        if mask.shape[:2] != frame.shape[:2]:
            mask = cv2.resize(mask, (frame_w, frame_h))

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(frame, contours, -1, color, round(line_width))

    scale = max(frame.shape[:2]) / 1280
    h_start = round(20 * scale)
    h_pos = h_start
    v_pos = round(80 * scale)
    for word, color in text:
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = max(1, round(scale))
        font_scale = scale * 2
        (text_w, text_h), _ = cv2.getTextSize(word, font, font_scale, thickness * 4)
        if h_pos != h_start and h_pos + text_w > frame_w:
            h_pos = h_start
            v_pos += round(text_h * 1.5)
        pos = (h_pos, v_pos)
        cv2.putText(
            frame,
            word,
            pos,
            font,
            font_scale,
            (255, 255, 255),
            thickness * 4,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            word,
            pos,
            font,
            font_scale,
            color,
            thickness * 2,
            cv2.LINE_AA,
        )
        h_pos += text_w

    return frame


def largest_mask_first(masks, colors):
    # We want to draw the larger masks first to above covering smaller masks
    mask_and_colors = list((m, c) for m, c in zip(masks, colors) if m is not None)
    mask_and_colors.sort(key=lambda p: p[0].sum(), reverse=True)
    masks, colors = zip(*mask_and_colors)
    return masks, colors


def visualize_subject_object(video, objects, action, all_contacts, max_edge=1280):
    frames = media.read_video(video)
    fps = frames.metadata.fps

    colors = [(255, 0, 0), (0, 0, 255), (0, 255, 0), (0, 255, 255), (255, 0, 255)]
    assert len(colors) >= len(objects)
    colors = colors[: len(objects)]

    all_contacts = [] if all_contacts is None else all_contacts

    contact_color = np.array([255, 255, 0], dtype=np.uint8)
    masks, names = zip(*objects)

    text = [(f"{action}: ", (0, 0, 0))]
    for i, (name, color) in enumerate(zip(names, colors)):
        word = name if i + 1 == len(names) else f"{name}, "
        text.append((word, color))

    masks, mask_colors = largest_mask_first(masks, colors)

    output = []
    for i in range(frames.shape[0]):
        frame = visualize_frame(
            frames[i], [m[i] for m in masks], mask_colors, max_edge=max_edge, text=text
        )

        for contacts in all_contacts:
            contact = contacts.get(i)
            if contact is not None:
                contact = np.array(contact, dtype=np.uint8)
                if contact.shape[:2] != frame.shape[:2]:
                    h, w = frame.shape[:2]
                    contact = cv2.resize(contact, (w, h))
                frame[contact > 0] = contact_color

        output.append(frame)
    return output, fps


async def remove_old_results(older_than):
    old_jobs = [
        job
        for job in ongoing_jobs.values()
        if job.result_index is not None and job.result_index < older_than
    ]
    for job in old_jobs:
        ongoing_jobs.pop(job.id)
        job.done.set()


def is_valid_argument_name(name):
    return name.lower() not in ("none", "unspecified", "unknown", "")

async def do_grounding(video, inputs: GroundingInputs, jobs: list[Job]):
    global result_counter
    path = None
    try:
        suffix = inputs.filename if inputs.filename.startswith(".") else f".{inputs.filename}"
        path = _write_upload_atomic(UPLOAD_ROOT, suffix, video)

        if not _wait_until_exists(path):
            print(f"Upload write looks successful but file not found: {path}")
            raise RuntimeError("Upload write failed (path missing)")

        print(f"Grounding on {path} (size={os.path.getsize(path)/1e6} MB)")

        validated_pairs = []
        for p in inputs.subject_object_pairs:
            argument_names = [n for n in p.argument_names if is_valid_argument_name(n)]
            validated_pairs.append((argument_names, p.caption))

        ongoing_jobs.update((job.id, job) for job in jobs)

        # 3) Build sync generator
        results = ground(path, validated_pairs)

        loop = asyncio.get_running_loop()

        for i, _ in enumerate(inputs.subject_object_pairs):
            t0 = default_timer()
            
            # Double-check existence before each heavy step (optional but helpful)
            if not os.path.exists(path):
                print(f"Video file disappeared before processing job {i}: {path}")
                raise RuntimeError("Video file disappeared")

            # 4) Pull next() in a thread (with its own existence check)
            def _next_with_check():
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Video path missing in worker: {path}")
                return next(results)

            try:
                result, extra_info = await loop.run_in_executor(THREAD_POOL, _next_with_check)
            except Exception as e:
                tb = ''.join(traceback.TracebackException.from_exception(e).format())
                failed_ids = []
                for job in jobs[i:]:
                    job.status = JobStatus.EXCEPTION
                    job.extra_info = {"error": repr(e), "traceback": tb}
                    job.done.set()
                    failed_ids.append(str(job.id))
                    print(f"grounding job finished: {job.id} [EXCEPTION]")
                print(f"Jobs {', '.join(failed_ids)} failed because of exception: {e!r}\n{tb}")
                return

            job = jobs[i]
            job.status = JobStatus.SUCCESS_INVALID
            job.result_index = result_counter

            if result is not None:
                argument_masks, has_contact, contacts = result

                # the requested action for this pair
                requested_action = inputs.subject_object_pairs[i].action

                # apply policy: only *require* contact for some actions
                need_contact = requires_contact(requested_action)
                is_valid = bool(has_contact) or (not need_contact)

                # label and job status
                action = requested_action if is_valid else "???"
                job.status = JobStatus.SUCCESS_VALID if is_valid else JobStatus.SUCCESS_INVALID

                # only overlay contact mask when it matters; skip otherwise
                contacts_for_viz = contacts if need_contact else None

                argument_names, _ = validated_pairs[i]

                if is_valid:
                    # build overlays ONLY for valid results
                    video_bytes, masks_bytes, viz_time = await loop.run_in_executor(
                        THREAD_POOL,
                        _build_artifacts,
                        path, argument_masks, argument_names, action,
                        contacts if need_contact else None
                    )
                    job.annotated_video = video_bytes
                    job.argument_masks = masks_bytes
                    extra_info["visualization_time"] = viz_time
                else:
                    # no artifacts for invalid → has_visualization becomes False
                    job.annotated_video = None
                    job.argument_masks = None
                    extra_info["rejection_reason"] = "contact-required action with no contact detected"

                extra_info.setdefault("contact_verification", {}).update({
                    "requires_contact": need_contact,
                    "detected_contact": bool(has_contact),
                    "valid_under_policy": is_valid,
                })

            extra_info["subject_object_pair"] = inputs.subject_object_pairs[i]
            job.extra_info = extra_info
            job.done.set()
            elapsed = default_timer() - t0
            print(f"grounding job finished: {job.id} [{job.status.name}] in {elapsed:.2f}s")

        result_counter += 1
        loop.create_task(remove_old_results(result_counter - result_time_to_live))

    finally:
        if path:
            try:
                os.remove(path)
            except FileNotFoundError:
                pass
            

@app.post("/ground")
async def ground_subject_object(
    video: Annotated[bytes, File()],
    inputs: Annotated[GroundingInputs, BeforeValidator(validate_inputs), Form()],
) -> list[uuid.UUID]:
    print("received grounding job")

    jobs = [Job(id=uuid.uuid4(), done=asyncio.Event()) for _ in inputs.subject_object_pairs]

    jobs_with_inputs = [
        {"job_id": str(job.id), **pair.model_dump()}
        for job, pair in zip(jobs, inputs.subject_object_pairs)
    ]
    print("grounding requests:\n" + json.dumps(jobs_with_inputs, ensure_ascii=False, indent=2))

    loop = asyncio.get_running_loop()
    loop.create_task(do_grounding(video, inputs, jobs))
    return [job.id for job in jobs]


@app.get("/result/{job_id}")
async def get_result(job_id: JobID) -> JobInfo:
    job = ongoing_jobs[job_id]
    # await job.done.wait()
    return JobInfo(
        id=job.id,
        status=job.status,
        has_visualization=job.annotated_video is not None,
        extra_info=job.extra_info,
    )


@app.get(
    "/result/{job_id}/video",
    responses={200: {"content": {"video/mp4": {}}}},
    response_class=Response,
)
async def get_result(job_id: JobID):
    job = ongoing_jobs[job_id]
    await job.done.wait()
    if job.status in (JobStatus.SUCCESS_VALID, JobStatus.SUCCESS_INVALID):
        if job.annotated_video is None:
            return Response(status_code=204)
        else:
            return Response(job.annotated_video, media_type="video/mp4")
    elif job.status == JobStatus.EXCEPTION:
        return Response(status_code=500)
    elif job.status == JobStatus.CANCELLED:
        return Response(status_code=444)
    raise NotImplementedError(job.status)


@app.get(
    "/result/{job_id}/masks",
    responses={200: {"content": {"application/octet-stream": {}}}},
    response_class=Response,
)
async def get_result(job_id: JobID):
    job = ongoing_jobs[job_id]
    await job.done.wait()
    if job.status in (JobStatus.SUCCESS_VALID, JobStatus.SUCCESS_INVALID):
        if job.annotated_video is None:
            return Response(status_code=204)
        else:
            return Response(job.argument_masks, media_type="application/octet-stream")
    elif job.status == JobStatus.EXCEPTION:
        return Response(status_code=500)
    elif job.status == JobStatus.CANCELLED:
        return Response(status_code=444)
    raise NotImplementedError(job.status)


@app.delete("/result")
async def delete_results() -> list[uuid.UUID]:
    ids = []
    for job in ongoing_jobs.values():
        ids.append(job.id)
        job.done.set()
    ongoing_jobs.clear()
    return ids


@app.delete("/result/{job_id}")
async def delete_result(job_id: JobID) -> uuid.UUID:
    job = ongoing_jobs.pop(job_id)
    job.done.set()
    return job.id


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", "-d", nargs="+", default=list(range(4)), type=int)
    parser.add_argument("--port", "-p", type=int, default=12000)
    parser.add_argument(
        "--depth_pro_ckpt",
        type=str,
        default="./depth_pro.pt",
    )
    parser.add_argument(
        "-t",
        "--result-ttl",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--contact-required-file",
        type=str,
        default="contact_required_actions.txt",
        help="Path to a text file listing actions that require contact (one per line)."
    )
    
    args = parser.parse_args()
    result_time_to_live = args.result_ttl

    CONTACT_REQUIRED_ACTIONS = load_contact_required_actions(args.contact_required_file)
    # print(f"[grounding] actions requiring contact: {sorted(CONTACT_REQUIRED_ACTIONS)}")

    devices = [f"cuda:{i}" for i in args.devices]

    ground = SubjectObjectGrounding(
        devices=devices,
        depth_pro_ckpt=args.depth_pro_ckpt,
        mask_video_max_frames=9,
    )

    uvicorn.run(app, host="0.0.0.0", port=args.port)
    ground.shutdown()
