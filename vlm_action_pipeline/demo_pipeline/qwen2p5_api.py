"""Qwen2.5-VL captioning service (POST /qwen) for the per-object track clips."""

import argparse
import json
import os
import tempfile
import threading
import traceback

import torch
from flask import Flask, jsonify, request
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

lock = threading.Lock()

MODEL_ID = "Qwen/Qwen2.5-VL-32B-Instruct"
# Sized for 4x B200, keeping GPU 0 nearly free for the other services.
MAX_MEMORY = ["1GiB", "140GiB", "140GiB", "140GiB"]
# Content keys qwen_vl_utils loads media from.
MEDIA_KEYS = ("image", "image_url", "video")

model = processor = None  # set by load_model()


def load_model(model_id: str, max_memory: dict, attn_implementation: str):
    global model, processor
    # We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        attn_implementation=attn_implementation,
        device_map="auto",
        max_memory=max_memory,
    )

    processor = AutoProcessor.from_pretrained(model_id)

    tok = processor.tokenizer
    tok.padding_side = "left"

    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token  # safe default

    # keep model configs in sync (helps generate/beam code paths)
    model.config.pad_token_id = tok.pad_token_id
    model.generation_config.pad_token_id = tok.pad_token_id

    # (optional but often helpful for long prompts)
    tok.truncation_side = "left"


app = Flask(__name__)


def _parse_bool(val):
    if isinstance(val, bool):
        return val
    if val is None:
        return False
    s = str(val).strip().lower()
    return s in {"1", "true", "yes", "y", "on"}


def _is_turn(turn) -> bool:
    if not isinstance(turn, dict):
        return False
    content = turn.get("content")
    return isinstance(content, str) or (
        isinstance(content, list)
        and all(isinstance(part, dict) and "type" in part for part in content)
    )


def _is_conversation(conv) -> bool:
    return isinstance(conv, list) and bool(conv) and all(_is_turn(t) for t in conv)


def _content_parts(convs):
    for conv in convs:
        for turn in conv:
            if isinstance(turn["content"], list):
                yield from turn["content"]


def _inject_image_paths(convs, image_paths):
    """
    Fill 'type':'image' items in order with the given paths. Images beyond the existing
    items are inserted, in order, at the start of the first user turn's content.
    """
    slots = [part for part in _content_parts(convs) if part["type"] == "image"]
    for slot, path in zip(slots, image_paths):
        slot["image"] = path

    extra = image_paths[len(slots) :]
    if extra:
        # prefer the first 'user' turn, else the very first turn
        turn = next((t for conv in convs for t in conv if t.get("role") == "user"), convs[0][0])
        if isinstance(turn["content"], str):
            turn["content"] = [{"type": "text", "text": turn["content"]}]
        turn["content"][:0] = [{"type": "image", "image": p} for p in extra]


def _inject_video_path(convs, path):
    """Point every 'video' item at the uploaded file."""
    for part in _content_parts(convs):
        if part["type"] == "video":
            part["video"] = path


def _only_uploaded_media(convs, paths) -> bool:
    """True if every item qwen_vl_utils would load as an image or video is one of `paths`."""
    for part in _content_parts(convs):
        refs = [part[key] for key in MEDIA_KEYS if key in part]
        if not refs and part["type"] not in MEDIA_KEYS:
            continue
        if not refs or any(ref not in paths for ref in refs):
            return False
    return True


def _is_single_conv(msgs):
    return isinstance(msgs, list) and msgs and isinstance(msgs[0], dict)


def vlm_generate(messages, max_new_tokens=128):
    # Normalize to batch
    batch_msgs = [messages] if _is_single_conv(messages) else messages

    # 1) Build chat texts (no tokenization here)
    texts = [
        processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
        for conv in batch_msgs
    ]

    # 2) Collect per-conversation vision inputs
    images_list, videos_list = [], []
    for conv in batch_msgs:
        imgs, vids = process_vision_info(conv)
        images_list.append(imgs)
        videos_list.append(vids)

    # 3) Normalize: None-per-sample is NOT allowed
    #    - if ALL are None -> pass None
    #    - else replace None with [] per sample
    if all(x is None for x in images_list):
        images_arg = None
    else:
        images_arg = [x if x is not None else [] for x in images_list]

    if all(x is None for x in videos_list):
        videos_arg = None
    else:
        videos_arg = [x if x is not None else [] for x in videos_list]

    # 4) Pack with padding
    inputs = processor(
        text=texts,
        images=images_arg,
        videos=videos_arg,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        gen_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    # 5) Trim prompts: rows are left-padded to the same length, so every answer starts there
    prompt_len = inputs.input_ids.shape[1]

    outs = []
    for out_ids in gen_ids:
        trimmed = out_ids[prompt_len:]
        outs.append(
            processor.batch_decode(
                [trimmed], skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
        )

    return outs[0] if _is_single_conv(messages) else outs


@app.route("/qwen", methods=["POST"])
def qwen():
    """
    POST form fields:
      - messages: JSON string of either:
         * single conversation: [{"role":"user","content":[...]}, ...]
         * batch: [ [{"role":"user","content":[...]}, ...],
                    [{"role":"user","content":[...]}, ...], ... ]
      - is_batch: optional truthy string ("true", "1", etc.)
      - image: one or more uploaded images (optional)
      - video: exactly one uploaded video (optional)
    Exactly one of 'image' or 'video' must be provided. Every 'video' item is pointed at the
    uploaded video, and 'image' items are filled with the uploaded images (see
    _inject_image_paths). Messages may not reference any other media.
    """
    # Prefer 'video' if provided
    if "video" in request.files:
        media_type, uploads, suffix = "video", [request.files["video"]], ".mp4"
    elif "image" in request.files:
        media_type, uploads, suffix = "image", request.files.getlist("image"), ".png"
    else:
        return jsonify({"error": "no vision file found"}), 400
    if any(f.filename == "" for f in uploads):
        return jsonify({"error": f"no chosen {media_type} file"}), 400

    if "messages" not in request.form:
        return jsonify({"error": "messages field is required"}), 400

    try:
        messages = json.loads(request.form["messages"])
    except json.JSONDecodeError as e:
        return jsonify({"error": f"Invalid JSON in messages: {e}"}), 400

    is_batch = _parse_bool(request.form.get("is_batch", False))

    # Normalize to a list of conversations (list[list[turn]])
    convs = messages if is_batch else [messages]
    if not (isinstance(convs, list) and convs and all(_is_conversation(c) for c in convs)):
        if is_batch:
            error = "Batch mode expects messages to be a list of conversations (list[list[turn]])"
        else:
            error = "Single mode expects messages to be a single conversation (list[turn])"
        return jsonify({"error": error}), 400

    with tempfile.TemporaryDirectory() as tmpdir:
        paths = []
        for i, upload in enumerate(uploads):
            path = os.path.join(tmpdir, f"{media_type}{i}{suffix}")
            upload.save(path)
            paths.append(path)

        if media_type == "video":
            _inject_video_path(convs, paths[0])
        else:
            _inject_image_paths(convs, paths)

        if not _only_uploaded_media(convs, paths):
            return jsonify({"error": "image/video items may only refer to the uploaded files"}), 400

        try:
            with lock:
                outputs = vlm_generate(convs, max_new_tokens=128)
        except Exception as e:
            traceback.print_exc()
            return jsonify({"error": f"inference failed ({type(e).__name__})"}), 500

    return jsonify({"response": outputs if is_batch else outputs[0]})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--host",
        default=os.environ.get("CRELERI_HOST", "127.0.0.1"),
        help="interface to bind (default: $CRELERI_HOST or 127.0.0.1)",
    )
    parser.add_argument("--port", type=int, default=17259, help="port (default: %(default)s)")
    parser.add_argument("--model", default=MODEL_ID, help="Hugging Face model id or local path")
    parser.add_argument(
        "--max-memory",
        nargs="+",
        default=MAX_MEMORY,
        metavar="SIZE",
        help=f"memory cap per GPU, in device order (default: {' '.join(MAX_MEMORY)})",
    )
    parser.add_argument(
        "--attn-implementation",
        default="flash_attention_2",
        choices=["flash_attention_2", "sdpa", "eager"],
        help="use sdpa when flash-attn is not installed (default: flash_attention_2)",
    )
    args = parser.parse_args()

    load_model(args.model, dict(enumerate(args.max_memory)), args.attn_implementation)
    app.run(host=args.host, port=args.port, debug=False)
