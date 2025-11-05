from flask import Flask, request, jsonify
import threading
import traceback
import os
import tempfile
import torch
import json

from transformers import AutoProcessor
from transformers import Qwen2_5_VLForConditionalGeneration

from qwen_vl_utils import process_vision_info


lock = threading.Lock()

MODEL_ID = "Qwen/Qwen2.5-VL-32B-Instruct"
max_mem = {0: "1GiB", 1: "140GiB", 2: "140GiB", 3: "140GiB"}
#max_mem = {0: "1GiB", 1: "210GiB", 2: "210GiB"}
#model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#    "Qwen/Qwen2.5-VL-32B-Instruct", 
#    dtype="auto", 
#    device_map="auto",
#    max_memory=max_mem
#)

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
     "Qwen/Qwen2.5-VL-32B-Instruct",
     dtype=torch.bfloat16,
     attn_implementation="flash_attention_2",
     device_map="auto",
     max_memory=max_mem
)

processor = AutoProcessor.from_pretrained(MODEL_ID)

tok = processor.tokenizer
tok.padding_side = "left"   

if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token       # safe default

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

def _ensure_list(x):
    return x if isinstance(x, list) else [x]

def _save_uploaded_image_files(image_files):
    """
    Save uploaded image files to temp PNGs and return a list of file:// paths.
    """
    temp_paths = []
    for file_storage in image_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            if isinstance(file_storage, (bytes, bytearray)):
                tmp.write(file_storage)
            else:
                tmp.write(file_storage.read())
            temp_paths.append("file://" + tmp.name)
    return temp_paths

def _inject_image_paths_into_messages(messages, image_paths):
    """
    Fill/append image content items in order with the given file:// paths.
    If there are more images than existing 'type':'image' items, append the extras
    to the first user message's content.
    """
    if not image_paths:
        return messages

    # Flatten references to all 'image' content slots (in order)
    image_slots = []
    for conv in _ensure_list(messages):
        for turn in conv:
            content = turn.get("content", [])
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image":
                    image_slots.append(item)

    # Fill existing slots first
    i = 0
    for slot in image_slots:
        if i >= len(image_paths):
            break
        slot["image"] = image_paths[i]
        i += 1

    # If leftover images, append them to the first user turn’s content
    if i < len(image_paths):
        # Find first turn to append to (prefer a 'user' role)
        first_turn = None
        for conv in _ensure_list(messages):
            for turn in conv:
                if turn.get("role") == "user":
                    first_turn = turn
                    break
            if first_turn:
                break
        if first_turn is None:
            # fallback: just use the very first turn
            first_turn = messages[0][0] if isinstance(messages[0], list) else messages[0]

        first_turn.setdefault("content", [])
        for p in image_paths[i:]:
            first_turn["content"].insert(0, {"type": "image", "image": p})

    return messages

def _inject_video_path_into_messages(messages, path):
    """Replace any 'video' entries with a *plain* local path."""
    def patch_conv(conv):
        for turn in conv:
            for part in turn.get("content", []):
                if part.get("type") == "video":
                    part["video"] = path  # <- no 'file://'
        return conv

    if isinstance(messages, list) and messages and isinstance(messages[0], list):
        return [patch_conv(conv) for conv in messages]  # batch
    return patch_conv(messages)  # single

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

    # 5) Trim prompts using non-pad lengths
    pad_id = getattr(processor.tokenizer, "pad_token_id", None)
    prompt_lens = (inputs.input_ids != pad_id).sum(dim=1).tolist() if pad_id is not None \
                  else [len(ids) for ids in inputs.input_ids]

    outs = []
    for i, out_ids in enumerate(gen_ids):
        trimmed = out_ids[prompt_lens[i]:]
        outs.append(
            processor.batch_decode([trimmed], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        )

    return outs[0] if _is_single_conv(messages) else outs


@app.route('/qwen', methods=['POST'])
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
    Exactly one of 'image' or 'video' must be provided.
    """
    try:
        media_type = None

        # Prefer 'video' if provided
        if 'video' in request.files:
            video_file = request.files['video']
            if video_file.filename == '':
                return jsonify({'error': 'no chosen video file'}), 400
            media_type = 'video'
        elif 'image' in request.files:
            # Could be multiple images
            image_files = request.files.getlist('image')
            if not image_files:
                return jsonify({'error': 'no chosen image file'}), 400
            media_type = 'image'
        else:
            return jsonify({'error': 'no vision file found'}), 400

        if 'messages' not in request.form:
            return jsonify({'error': 'messages field is required'}), 400

        # Safer than eval
        try:
            messages = json.loads(request.form['messages'])
        except json.JSONDecodeError as e:
            return jsonify({'error': f'Invalid JSON in messages: {e}'}), 400

        is_batch = _parse_bool(request.form.get('is_batch', False))

        # Normalize messages shape for processor:
        # - single: list[turn]
        # - batch:  list[list[turn]]
        if is_batch:
            if not (isinstance(messages, list) and messages and isinstance(messages[0], list)):
                return jsonify({'error': 'Batch mode expects messages to be a list of conversations (list[list[turn]])'}), 400
        else:
            if not (isinstance(messages, list) and messages and isinstance(messages[0], dict)):
                return jsonify({'error': 'Single mode expects messages to be a single conversation (list[turn])'}), 400

        with lock:
            temp_paths_to_cleanup = []

            if media_type == 'video':
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                    tmp.write(video_file.read())
                    tmp_path = tmp.name
                    temp_paths_to_cleanup.append(tmp_path)
                messages = _inject_video_path_into_messages(messages, tmp_path)

            elif media_type == 'image':
                image_paths = []
                for f in image_files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                        f.stream.seek(0)
                        tmp.write(f.read())
                        image_paths.append(tmp.name)
                        temp_paths_to_cleanup.append(tmp.name)
                messages = _inject_image_paths_into_messages(messages, image_paths)

            # -------- Inference --------
            vlm_output = vlm_generate(messages=messages, max_new_tokens=128)

        for p in temp_paths_to_cleanup:
            try:
                os.remove(p)
            except Exception:
                pass

        # Ensure JSON-friendly return shape
        if is_batch and isinstance(vlm_output, list):
            return jsonify({"response": vlm_output})
        else:
            return jsonify({"response": vlm_output if isinstance(vlm_output, str) else (vlm_output[0] if vlm_output else "")})

    except Exception:
        traceback.print_exc()
        return jsonify({"error": traceback.format_exc()}), 500

if __name__ == '__main__':
    app.run(debug=False, host="0.0.0.0", port=17259)
