"""CReLeRI web UI: serves the page and proxies the segmentation and grounding APIs."""

import argparse
import asyncio
import logging
import os
import uuid
from urllib.parse import urlparse

import requests
from flask import Flask, Response, jsonify, render_template, request, url_for
from hypercorn.asyncio import serve
from hypercorn.config import Config

logger = logging.getLogger(__name__)

SEGMENTATION_API_URL = os.environ.get("SEGMENTATION_API_URL", "http://localhost:17249").rstrip("/")
GROUNDING_API_URL = os.environ.get("GROUNDING_API_URL", "http://localhost:12000").rstrip("/")

MAX_UPLOAD_BYTES = 512 * 1024 * 1024
SEGMENTATION_TYPES = ("0", "1", "2")  # AdaTAD + scene cuts, uniform 5 s windows, whole video

# requests has already undone any transfer or content encoding, and Werkzeug sets the
# framing headers itself, so only headers that describe the payload are passed on.
FORWARDED_HEADERS = {
    "content-type",
    "content-disposition",
    "cache-control",
    "etag",
    "last-modified",
}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES


def forwarded_headers(r):
    return [(k, v) for k, v in r.headers.items() if k.lower() in FORWARDED_HEADERS]


def upstream_error_detail(r):
    """FastAPI puts its error message under "detail"; fall back to the raw body."""
    try:
        detail = r.json()["detail"]
    except (ValueError, KeyError, TypeError):
        return r.text
    return detail if isinstance(detail, str) else str(detail)


@app.errorhandler(413)
def upload_too_large(_error):
    limit_mb = MAX_UPLOAD_BYTES // (1024 * 1024)
    return jsonify({"error": f"Video is larger than the {limit_mb} MB upload limit"}), 413


@app.route("/proxy/thumbnails/<path:filename>")
def proxy_thumbnails(filename):
    try:
        r = requests.get(f"{SEGMENTATION_API_URL}/thumbnails/{filename}", timeout=15)
    except requests.RequestException as e:
        logger.warning("Thumbnail %s unavailable: %s", filename, e)
        return jsonify({"error": "Segmentation API unreachable"}), 502
    return Response(r.content, r.status_code, forwarded_headers(r))


@app.route("/analyze_video", methods=["POST"])
def analyze_video():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "No selected file"}), 400
    segmentation_type = request.form.get("segmentation_type", "0")
    if segmentation_type not in SEGMENTATION_TYPES:
        return jsonify({"error": "segmentation_type must be 0, 1 or 2"}), 400

    # Send a neutral name rather than the client's; the segmentation API stores the upload
    # under its own name.
    upload_name = f"{uuid.uuid4().hex}.mp4"
    try:
        r = requests.post(
            f"{SEGMENTATION_API_URL}/analyze_video",
            files={"file": (upload_name, file.stream, "video/mp4")},
            params={"segmentation_type": segmentation_type},
            headers={"accept": "application/json"},
            timeout=60,
        )
    except requests.RequestException as e:
        logger.warning("Upload to the segmentation API failed: %s", e)
        return jsonify({"error": "Segmentation API unreachable"}), 502
    if not r.ok:
        detail = upstream_error_detail(r)
        logger.warning("Segmentation API rejected an upload (%s): %s", r.status_code, detail)
        status = r.status_code if r.status_code < 500 else 502
        return jsonify({"error": "Segmentation API rejected the upload", "detail": detail}), status
    return jsonify(r.json())


@app.route("/task_status/<task_id>")
def task_status(task_id):
    try:
        r = requests.get(f"{SEGMENTATION_API_URL}/task_status/{task_id}", timeout=15)
    except requests.RequestException as e:
        # Not final: the task keeps running in Celery, so the page keeps polling.
        logger.warning("Segmentation API unreachable while polling %s: %s", task_id, e)
        return jsonify({"status": "Unavailable", "error": "Segmentation API unreachable"}), 503
    if not r.headers.get("Content-Type", "").startswith("application/json"):
        return Response(r.content, r.status_code, forwarded_headers(r))
    data = r.json()

    if data.get("status") == "Completed" and "result" in data:
        # Point thumbnails at this server's proxy; the segmentation API may not be reachable
        # from the browser.
        rewritten = []
        for t in data["result"].get("thumbnails", []):
            filename = t.get("thumbnail")
            if not filename and t.get("thumbnail_url"):
                filename = os.path.basename(urlparse(t["thumbnail_url"]).path)
            if filename:
                rewritten.append(
                    {
                        "segment": t.get("segment"),
                        "thumbnail_url": url_for("proxy_thumbnails", filename=filename),
                    }
                )
        data["result"]["thumbnails"] = rewritten

    return jsonify(data), r.status_code


@app.route("/video/result/<task_id>")
def video_result(task_id):
    try:
        r = requests.get(f"{GROUNDING_API_URL}/result/{task_id}", timeout=15)
    except requests.RequestException as e:
        logger.warning("Grounding API unreachable for job %s: %s", task_id, e)
        return jsonify({"error": "Grounding API unreachable"}), 502
    return Response(r.content, r.status_code, forwarded_headers(r))


@app.route("/video/get_result/<task_id>")
def get_result(task_id):
    try:
        r = requests.get(f"{GROUNDING_API_URL}/result/{task_id}/video", timeout=60, stream=True)
    except requests.RequestException as e:
        logger.warning("Grounding API unreachable for job %s: %s", task_id, e)
        return jsonify({"error": "Grounding API unreachable"}), 502
    response = Response(r.iter_content(chunk_size=8192), r.status_code, forwarded_headers(r))
    response.call_on_close(r.close)
    return response


@app.route("/")
def index():
    return render_template("sentence.html", max_upload_bytes=MAX_UPLOAD_BYTES)


def main():
    parser = argparse.ArgumentParser(description="Serve the CReLeRI web UI.")
    parser.add_argument("port", nargs="?", type=int, default=5004, help="port (default: 5004)")
    parser.add_argument(
        "--host",
        default=os.environ.get("CRELERI_HOST", "127.0.0.1"),
        help="interface to bind (default: $CRELERI_HOST or 127.0.0.1)",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    config = Config()
    config.bind = [f"{args.host}:{args.port}"]
    # Hand Hypercorn a plain logger so its messages go through the root handler, not a second one.
    config.errorlog = logging.getLogger("hypercorn.error")
    # Hypercorn buffers WSGI request bodies and answers a bare 400 above this size, so give it
    # room for the multipart framing and let Flask reject oversize uploads with a JSON 413.
    config.wsgi_max_body_size = MAX_UPLOAD_BYTES + 1024 * 1024
    asyncio.run(serve(app, config))


if __name__ == "__main__":
    main()
