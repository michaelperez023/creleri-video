"""HTTP front end of the segmentation service: queues uploaded videos for the Celery worker."""

import argparse
import mimetypes
import os
import shutil
import uuid
from pathlib import Path
from typing import Annotated

import uvicorn
from celery import Celery, states
from celery.result import AsyncResult
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.staticfiles import StaticFiles
from segment_utils import SegmentationType

BASE_DIR = Path(__file__).resolve().parent
SHARED_DIR = BASE_DIR / "shared"
THUMBNAIL_DIR = BASE_DIR / "thumbnails"
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:17253/0")
UPLOAD_CHUNK_SIZE = 1024 * 1024

# Celery also reports PENDING for task ids it has never seen.
STATUS_BY_STATE = {
    states.PENDING: "Processing",
    states.RECEIVED: "Processing",
    states.STARTED: "Processing",
    states.RETRY: "Processing",
    states.SUCCESS: "Completed",
    states.FAILURE: "Failed",
    states.REVOKED: "Failed",
}

celery_app = Celery("segmentation_api", broker=REDIS_URL, backend=REDIS_URL)
app = FastAPI()

THUMBNAIL_DIR.mkdir(exist_ok=True)
app.mount("/thumbnails", StaticFiles(directory=THUMBNAIL_DIR), name="thumbnails")


def is_video(upload: UploadFile) -> bool:
    content_type = upload.content_type
    if not content_type or content_type == "application/octet-stream":
        content_type = mimetypes.guess_type(upload.filename or "")[0]
    return (content_type or "").startswith("video/")


@app.get("/task_status/{task_id}")
def get_task_status(task_id: str, request: Request):
    task = AsyncResult(task_id, app=celery_app)
    status = STATUS_BY_STATE.get(task.state, "Failed")
    if status == "Processing":
        return {"status": status}
    if status == "Failed":
        return {"status": status, "error": str(task.info)}

    result = task.result
    thumbnails = [
        {
            "thumbnail_url": str(request.url_for("thumbnails", path=item["thumbnail"])),
            "segment": item["segment"],
        }
        for item in result.get("thumbnails", [])
    ]
    return {
        "status": status,
        "result": {
            "proposals": result.get("proposals"),
            "results": result.get("results"),
            "thumbnails": thumbnails,
        },
    }


@app.post("/analyze_video", status_code=201)
def analyze_video(
    file: Annotated[UploadFile, File()],
    segmentation_type: Annotated[SegmentationType, Query()],
):
    if not is_video(file):
        raise HTTPException(status_code=415, detail="Uploaded file must be a video.")

    # Never reuse the client's filename. The .mp4 suffix is what OpenTAD expects; FFmpeg and
    # decord detect the actual container from the file contents.
    SHARED_DIR.mkdir(exist_ok=True)
    upload_path = SHARED_DIR / f"{uuid.uuid4().hex}.mp4"
    try:
        with upload_path.open("wb") as out:
            shutil.copyfileobj(file.file, out, UPLOAD_CHUNK_SIZE)
        task = celery_app.send_task(
            "tasks.process_video_task", args=[str(upload_path), int(segmentation_type)]
        )
    except Exception:
        upload_path.unlink(missing_ok=True)
        raise
    return {"task_id": task.id}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CReLeRI segmentation API")
    parser.add_argument(
        "--host",
        default=os.environ.get("CRELERI_HOST", "127.0.0.1"),
        help="Interface to bind (default: $CRELERI_HOST or 127.0.0.1)",
    )
    parser.add_argument(
        "-p", "--port", type=int, default=17249, help="Port number to use for the API"
    )
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)
