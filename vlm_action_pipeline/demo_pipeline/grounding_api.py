"""Grounding API: locates an action's arguments in a clip and checks that they touch.

POST /ground takes the clip and its (argument_names, action, caption) proposals and returns one
job id per proposal. GET /result/{id} reports the verdict, and GET /result/{id}/video and
GET /result/{id}/masks return the annotated clip and the argument masks.
"""

import argparse
import asyncio
import concurrent.futures
import contextlib
import logging
import os
import pathlib
import shutil
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from io import BytesIO
from timeit import default_timer
from typing import Annotated, BinaryIO

import cv2
import mediapy as media
import numpy as np
import uvicorn
from fastapi import FastAPI, Form, HTTPException, Response, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from grounding import SubjectObjectGrounding, frame_cache_dir, private_dir
from pydantic import AfterValidator, BaseModel, ValidationError, field_serializer

logger = logging.getLogger(__name__)

HERE = pathlib.Path(__file__).resolve().parent

UPLOAD_ROOT = pathlib.Path(
    os.environ.get("GROUNDING_UPLOAD_DIR")
    or private_dir(pathlib.Path(tempfile.gettempdir()) / f"grounding-uploads-{os.getuid()}")
)
UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)

# Container types accepted for the uploaded clip; `inputs.filename` only picks one of these.
VIDEO_SUFFIXES = frozenset({".avi", ".mkv", ".mov", ".mp4", ".webm"})

CONTACT_REQUIRED_ACTIONS: set[str] = set()


def load_contact_required_actions(path: str | None) -> set[str]:
    """
    One action per line, named as in adatad_temp_seg/action_categories.csv;
    case-insensitive; '#' starts a comment. Blank lines ignored.
    An empty path disables the contact requirement.
    """
    actions = set()
    if not path:
        return actions
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.split("#", 1)[0].strip().lower()
            if line:
                actions.add(line)
    return actions


def requires_contact(action: str) -> bool:
    return (action or "").strip().lower() in CONTACT_REQUIRED_ACTIONS


def _video_suffix(filename: str) -> str:
    suffix = "." + filename.rsplit(".", 1)[-1].lower()
    if suffix not in VIDEO_SUFFIXES:
        raise HTTPException(422, f"unsupported video type: {filename!r}")
    return suffix


def _save_upload(src: BinaryIO, suffix: str) -> str:
    path = os.path.join(UPLOAD_ROOT, f"upload_{uuid.uuid4().hex}{suffix}")
    try:
        with open(path, "wb") as dst:
            shutil.copyfileobj(src, dst)
    except BaseException:
        pathlib.Path(path).unlink(missing_ok=True)
        raise
    return path


def _discard_upload(path: str) -> None:
    """Delete an upload and the frames extracted from it."""
    with contextlib.suppress(FileNotFoundError):
        # The frame cache is keyed by the upload's path and mtime, so locate it first.
        shutil.rmtree(frame_cache_dir(path), ignore_errors=True)
        os.remove(path)


ground = None
app = FastAPI()

# Seconds a finished job's results stay available (--result-ttl). The web UI reads them only
# once the whole video has been processed, which can take hours.
result_ttl = 24 * 60 * 60

THREAD_POOL = concurrent.futures.ThreadPoolExecutor(
    max_workers=max(2, min(8, (os.cpu_count() or 4)))
)

# The event loop keeps only weak references to tasks.
_background_tasks: set[asyncio.Task] = set()


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


class JobStatus(Enum):
    PENDING = 0
    SUCCESS_VALID = 1
    SUCCESS_INVALID = 2
    EXCEPTION = 3
    CANCELLED = 4


@dataclass
class Job:
    id: uuid.UUID
    done: asyncio.Event = field(default_factory=asyncio.Event)
    status: JobStatus = JobStatus.PENDING
    annotated_video: bytes | None = None
    argument_masks: bytes | None = None
    extra_info: dict | None = None
    finished_at: float | None = None

    def finish(
        self,
        status: JobStatus,
        extra_info: dict | None = None,
        annotated_video: bytes | None = None,
        argument_masks: bytes | None = None,
    ) -> None:
        """Publish the final state in one step, so readers never see a partial result.

        Only the first call counts; a cancelled job stays cancelled.
        """
        if self.done.is_set():
            return
        self.annotated_video = annotated_video
        self.argument_masks = argument_masks
        self.extra_info = extra_info
        self.status = status
        self.finished_at = time.monotonic()
        self.done.set()


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


def evict_expired_results():
    cutoff = time.monotonic() - result_ttl
    expired = [
        job_id
        for job_id, job in ongoing_jobs.items()
        if job.finished_at is not None and job.finished_at < cutoff
    ]
    for job_id in expired:
        del ongoing_jobs[job_id]


def is_valid_argument_name(name):
    return name.lower() not in ("none", "unspecified", "unknown", "")


async def do_grounding(path: str, pairs: list[SubjectObjectPair], jobs: list[Job]):
    loop = asyncio.get_running_loop()
    arguments = [[n for n in p.argument_names if is_valid_argument_name(n)] for p in pairs]
    try:
        results = ground(path, [(names, p.caption) for names, p in zip(arguments, pairs)])
        for i, (job, pair, argument_names) in enumerate(zip(jobs, pairs, arguments)):
            if all(j.done.is_set() for j in jobs[i:]):
                break  # every remaining job was cancelled
            start = default_timer()
            result, extra_info = await loop.run_in_executor(THREAD_POOL, next, results)
            if job.done.is_set():
                continue

            status = JobStatus.SUCCESS_INVALID
            video_bytes = masks_bytes = None
            if result is not None:
                argument_masks, has_contact, contacts = result

                # apply policy: only *require* contact for some actions
                need_contact = requires_contact(pair.action)
                is_valid = bool(has_contact) or not need_contact

                if is_valid:
                    # build overlays ONLY for valid results
                    video_bytes, masks_bytes, viz_time = await loop.run_in_executor(
                        THREAD_POOL,
                        _build_artifacts,
                        path,
                        argument_masks,
                        argument_names,
                        pair.action,
                        # only overlay contact points when the action requires them
                        contacts if need_contact else None,
                    )
                    extra_info["visualization_time"] = viz_time
                    status = JobStatus.SUCCESS_VALID
                else:
                    extra_info["rejection_reason"] = (
                        "contact-required action with no contact detected"
                    )

                extra_info.setdefault("contact_verification", {}).update(
                    {
                        "requires_contact": need_contact,
                        "detected_contact": bool(has_contact),
                        "valid_under_policy": is_valid,
                    }
                )

            extra_info["subject_object_pair"] = pair
            job.finish(status, extra_info, video_bytes, masks_bytes)
            elapsed = default_timer() - start
            logger.info("Grounding job %s finished: %s in %.2fs", job.id, job.status.name, elapsed)
    except Exception as e:
        failed = [job for job in jobs if not job.done.is_set()]
        logger.exception("Grounding jobs failed: %s", ", ".join(str(job.id) for job in failed))
        for job in failed:
            job.finish(JobStatus.EXCEPTION, {"error": type(e).__name__})
    finally:
        await asyncio.to_thread(_discard_upload, path)


@app.post("/ground")
async def ground_subject_object(
    video: UploadFile,
    inputs: Annotated[str, Form()],
) -> list[uuid.UUID]:
    # `inputs` is a single JSON-encoded form field. Declared as a model-typed Form(), newer
    # FastAPI versions would instead look for one form field per model field.
    try:
        parsed = GroundingInputs.model_validate_json(inputs)
    except ValidationError as e:
        raise RequestValidationError(e.errors()) from e
    path = await run_in_threadpool(_save_upload, video.file, _video_suffix(parsed.filename))

    evict_expired_results()
    jobs = [Job(id=uuid.uuid4()) for _ in parsed.subject_object_pairs]
    ongoing_jobs.update((job.id, job) for job in jobs)

    task = asyncio.create_task(do_grounding(path, parsed.subject_object_pairs, jobs))
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)

    logger.info("Queued grounding jobs: %s", ", ".join(str(job.id) for job in jobs))
    return [job.id for job in jobs]


@app.get("/result/{job_id}")
async def get_result_info(job_id: JobID) -> JobInfo:
    job = ongoing_jobs[job_id]
    return JobInfo(
        id=job.id,
        status=job.status,
        has_visualization=job.annotated_video is not None,
        extra_info=job.extra_info,
    )


async def _finished_job(job_id: uuid.UUID) -> Job:
    job = ongoing_jobs[job_id]
    await job.done.wait()
    return job


def _artifact_response(job: Job, content: bytes | None, media_type: str) -> Response:
    if job.status is JobStatus.EXCEPTION:
        return Response(status_code=500)
    if job.status is JobStatus.CANCELLED:
        return Response(status_code=444)
    if content is None:
        return Response(status_code=204)
    return Response(content, media_type=media_type)


@app.get(
    "/result/{job_id}/video",
    responses={200: {"content": {"video/mp4": {}}}},
    response_class=Response,
)
async def get_result_video(job_id: JobID) -> Response:
    job = await _finished_job(job_id)
    return _artifact_response(job, job.annotated_video, "video/mp4")


@app.get(
    "/result/{job_id}/masks",
    responses={200: {"content": {"application/octet-stream": {}}}},
    response_class=Response,
)
async def get_result_masks(job_id: JobID) -> Response:
    job = await _finished_job(job_id)
    return _artifact_response(job, job.argument_masks, "application/octet-stream")


@app.delete("/result")
async def delete_results() -> list[uuid.UUID]:
    jobs = list(ongoing_jobs.values())
    ongoing_jobs.clear()
    for job in jobs:
        job.finish(JobStatus.CANCELLED)
    return [job.id for job in jobs]


@app.delete("/result/{job_id}")
async def delete_result(job_id: JobID) -> uuid.UUID:
    job = ongoing_jobs.pop(job_id)
    job.finish(JobStatus.CANCELLED)
    return job.id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--host",
        default=os.environ.get("CRELERI_HOST", "127.0.0.1"),
        help="interface to bind (default: $CRELERI_HOST or 127.0.0.1)",
    )
    parser.add_argument("--port", "-p", type=int, default=12000, help="port (default: %(default)s)")
    parser.add_argument(
        "--devices",
        "-d",
        nargs="+",
        default=list(range(4)),
        type=int,
        help="GPU indices: the first runs DepthPro and GroundingDINO, each other one runs a "
        "SAM 2 + Molmo worker (default: 0 1 2 3)",
    )
    parser.add_argument(
        "--depth_pro_ckpt",
        default=str(HERE / "depth_pro.pt"),
        help="Depth Pro checkpoint (default: depth_pro.pt next to this file)",
    )
    parser.add_argument(
        "-t",
        "--result-ttl",
        type=float,
        default=result_ttl,
        help="seconds to keep a finished job's results. Results, including the annotated video "
        "and masks, stay in memory until the first /ground request after they expire "
        "(default: %(default)s, one day)",
    )
    parser.add_argument(
        "--contact-required-file",
        default=str(HERE / "contact_required_actions.txt"),
        help="actions that require contact, one per line; an empty value turns the check off "
        "(default: contact_required_actions.txt next to this file)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    result_ttl = args.result_ttl
    CONTACT_REQUIRED_ACTIONS = load_contact_required_actions(args.contact_required_file)
    logger.info(
        "Actions requiring contact: %s", ", ".join(sorted(CONTACT_REQUIRED_ACTIONS)) or "none"
    )

    ground = SubjectObjectGrounding(
        devices=[f"cuda:{i}" for i in args.devices],
        depth_pro_ckpt=args.depth_pro_ckpt,
        mask_video_max_frames=9,
    )

    uvicorn.run(app, host=args.host, port=args.port)
    ground.shutdown()
