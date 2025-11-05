from datetime import datetime, timezone
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile, Query, Request
from celery.result import AsyncResult
from celery import Celery
import os
from pathlib import Path
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import argparse

app = FastAPI()
REDIS_PORT = 17253

celery_app = Celery('api-docker', broker=f'redis://localhost:{REDIS_PORT}/0', backend=f'redis://localhost:{REDIS_PORT}/0')

thumbnail_dir = Path("./thumbnails")
thumbnail_dir.mkdir(parents=True, exist_ok=True)

app.mount("/thumbnails", StaticFiles(directory=str(thumbnail_dir)), name="thumbnails")

@app.get("/task_status/{task_id}")
async def get_task_status(task_id: str, request: Request):
    task = AsyncResult(task_id, app=celery_app)
    if task.state == 'PENDING':
        # Task is still running
        response = {'status': 'Processing'}
    elif task.state == 'SUCCESS':
        # Task completed successfully
        result = task.result
        # Build the full URL for each thumbnail
        thumbnails_info = []
        for item in result.get('thumbnails', []):
            thumbnail_filename = item['thumbnail']
            segment_times = item['segment']
            # Use request.url_for to generate the full URL
            thumbnail_url = str(request.url_for('thumbnails', path=thumbnail_filename))
            thumbnails_info.append({
                "thumbnail_url": thumbnail_url,
                "segment": segment_times
            })
        response = {
            'status': 'Completed',
            'result': {
                'proposals': result.get('proposals'),
                'results': result.get('results'),
                'thumbnails': thumbnails_info
            }
        }
        print(f"response: {response}")
    else:
        # Task failed
        response = {'status': 'Failed', 'error': str(task.info)}
    return response

@app.post("/analyze_video", status_code=201)
async def analyze_video(file: UploadFile = File(...), segmentation_type: int = Query(...)):
    if not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be a video.")

    shared_dir = Path("./shared")
    shared_dir.mkdir(parents=True, exist_ok=True)

    # Save the video to the shared directory with an absolute path
    temp_file_path = shared_dir / f"temp_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}_{file.filename}"
    with open(temp_file_path, "wb") as temp_file:
        temp_file.write(await file.read())

    # Enqueue the task
    task = celery_app.send_task('tasks.process_video_task', args=[str(temp_file_path), segmentation_type])

    # Return the task ID to the client
    return {"task_id": task.id}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--port", type=int, default=17249, help="Port number to use for the API")
    args = parser.parse_args()

    uvicorn.run(app, host="0.0.0.0", port=args.port)
