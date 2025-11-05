import os, sys, asyncio, subprocess, uuid
from io import BytesIO
from flask import Flask, render_template, request, jsonify, redirect, flash, Response, send_file, url_for
from werkzeug.utils import secure_filename
import requests
from hypercorn.config import Config
from hypercorn.asyncio import serve
from urllib.parse import urlparse

app = Flask(__name__, static_url_path="/static")
app.secret_key = 'super secret key'
app.config["SESSION_TYPE"] = "filesystem"
app.config["UPLOAD_FOLDER"] = "./uploads"
app.config["MAX_CONTENT_LENGTH"] = 512 * 1024 * 1024  # 512 MB

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

SEGMENTATION_API_PORT = 17249
GROUNDING_PORT = 12000
SEGMENTATION_TYPE = 0  # default

def _backend_url(port, path):
    return f"http://localhost:{port}{path}"

def compress_and_downscale_video(input_path, output_path, resolution="1280x720"):
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-vf", f"scale={resolution}",
        "-c:v", "libx264", "-preset", "fast", "-crf", "28",
        "-c:a", "aac", "-b:a", "128k",
        output_path
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
@app.route("/set_backend_segmentation_type", methods=["POST"])
def set_backend_segmentation_type():
    global SEGMENTATION_TYPE
    data = request.get_json(silent=True) or {}
    SEGMENTATION_TYPE = int(data.get("segmentationType", 0))
    return jsonify({"status": "success", "type": SEGMENTATION_TYPE})

@app.route("/proxy/thumbnails/<path:filename>")
def proxy_thumbnails(filename):
    url = _backend_url(SEGMENTATION_API_PORT, f"/thumbnails/{filename}")
    try:
        r = requests.get(url, timeout=15)
    except requests.RequestException as e:
        return jsonify({"error": str(e)}), 502
    if r.status_code != 200:
        return Response(r.content, r.status_code)
    # pass through content type if provided
    mimetype = r.headers.get("Content-Type", "image/jpeg")
    return send_file(BytesIO(r.content), mimetype=mimetype)

@app.route("/analyze_video", methods=["POST"])
def analyze_video():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "No selected file"}), 400

    # Save upload with safe, unique filename
    base = secure_filename(file.filename) or "upload.mp4"
    uid = uuid.uuid4().hex
    src_path = os.path.join(app.config["UPLOAD_FOLDER"], f"{uid}_{base}")

    try:
        with open(src_path, "wb") as f:
            for chunk in iter(lambda: file.stream.read(1024 * 1024), b""):
                f.write(chunk)

        size_mb = os.path.getsize(src_path) / (1024 * 1024)
        file_to_send = src_path

        # Compress if > 10MB
        #if size_mb > 10:
        #    dst_path = os.path.join(app.config["UPLOAD_FOLDER"], f"{uid}_compressed.mp4")
        #    compress_and_downscale_video(src_path, dst_path)
        #    file_to_send = dst_path

        api_resp = send_video_to_external_api(file_to_send)
        return jsonify(api_resp)

    except subprocess.CalledProcessError as e:
        return jsonify({"error": "ffmpeg failed", "detail": e.stderr.decode(errors="ignore")}), 500
    except requests.RequestException as e:
        return jsonify({"error": "backend request failed", "detail": str(e)}), 502
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        # cleanup both original and compressed if present
        for p in (src_path, locals().get("file_to_send")):
            try:
                if p and os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass
    
def send_video_to_external_api(video_file_path: str):
    url = _backend_url(SEGMENTATION_API_PORT, "/analyze_video")
    with open(video_file_path, "rb") as fh:
        files = {"file": ("video.mp4", fh, "video/mp4")}
        r = requests.post(
            url,
            files=files,
            params={"segmentation_type": SEGMENTATION_TYPE},
            headers={"accept": "application/json"},
            timeout=60,
        )
    # pass through upstream status + body
    if r.headers.get("Content-Type", "").startswith("application/json"):
        data = r.json()
    else:
        data = {"raw": r.text}
    if r.status_code not in (200, 201, 202):
        raise requests.RequestException(f"{r.status_code}: {data}")
    return data

'''@app.route("/task_status/<task_id>")
def task_status(task_id):
    url = _backend_url(SEGMENTATION_API_PORT, f"/task_status/{task_id}")
    try:
        r = requests.get(url, timeout=15)
        return Response(r.content, r.status_code, r.headers.items())
    except requests.RequestException as e:
        return jsonify({"error": str(e)}), 502'''

@app.route("/task_status/<task_id>")
def task_status(task_id):
    url = _backend_url(SEGMENTATION_API_PORT, f"/task_status/{task_id}")
    try:
        r = requests.get(url, timeout=15)
        # If the upstream isn't JSON (e.g., error HTML), just pass through
        if not r.headers.get("Content-Type", "").startswith("application/json"):
            return Response(r.content, r.status_code, r.headers.items())
        data = r.json()
    except requests.RequestException as e:
        return jsonify({"status": "Failed", "error": str(e)}), 502

    # Only rewrite on success payloads
    if data.get("status") == "Completed" and "result" in data:
        rewritten = []
        for t in data["result"].get("thumbnails", []):
            # backend returns either {"thumbnail_url": ".../thumbnails/foo.jpg"} or {"thumbnail": "foo.jpg"}
            filename = t.get("thumbnail")
            if not filename:
                tu = t.get("thumbnail_url", "")
                # extract "foo.jpg" from ".../thumbnails/foo.jpg"
                filename = os.path.basename(urlparse(tu).path) if tu else None
            if not filename:
                # skip malformed entries rather than returning 'undefined'
                continue

            rewritten.append({
                "segment": t.get("segment"),
                "thumbnail_url": url_for("proxy_thumbnails", filename=filename)  # -> /proxy/thumbnails/foo.jpg
            })

        # keep other fields as-is, but normalize thumbnails
        data["result"]["thumbnails"] = rewritten

    return jsonify(data), r.status_code

@app.route("/video/result/<task_id>")
def video_result(task_id):
    url = _backend_url(GROUNDING_PORT, f"/result/{task_id}")
    try:
        r = requests.get(url, timeout=15)
        return Response(r.content, r.status_code, r.headers.items())
    except requests.RequestException as e:
        return jsonify({"error": str(e)}), 502

@app.route("/video/get_result/<task_id>")
def get_result(task_id):
    url = _backend_url(GROUNDING_PORT, f"/result/{task_id}/video")
    try:
        r = requests.get(url, timeout=60, stream=True)
        return Response(r.iter_content(chunk_size=8192), r.status_code, r.headers.items())
    except requests.RequestException as e:
        return jsonify({"error": str(e)}), 502
    
@app.route("/")
def index():
    return render_template("sentence.html")

if __name__ == "__main__":
    cfg = Config()
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    cfg.bind = [f"0.0.0.0:{port}"]
    cfg.reload = True
    asyncio.run(serve(app, cfg))