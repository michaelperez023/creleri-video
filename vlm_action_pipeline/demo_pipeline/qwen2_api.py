from flask import Flask, request, jsonify
import threading
import traceback
import os
import tempfile
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

lock = threading.Lock()

# default: Load the model on the available device(s)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-72B-Instruct", torch_dtype="auto", device_map="auto"
)

processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-72B-Instruct")

def vlm_generate(messages, video_data=None, image_data=None, batch_inference=False):
    # Preparation for inference
    if not batch_inference:
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
    else:
        assert len(messages) > 1, "Batch inference requires more than one message."
        texts = [
            processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in messages
        ]
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
    inputs = inputs.to("cuda")

    # Inference
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    print(output_text)
    return output_text

app = Flask(__name__)

@app.route('/qwen', methods=['POST'])
def qwen():
    media_type = None
    if 'video' in request.files:
        video = request.files['video']
        if video.filename == '':
            return jsonify({'error': 'no chosen file'}), 400
        media_type = 'video'
    elif 'image' in request.files:
        image = request.files.getlist('image')
        image = [x.read() for x in image]
        media_type = 'image'
    else:
        return jsonify({'error': 'no vision file found'}), 400
    
    messages = request.form['messages']
    # video_path = request.form['video_path']
    is_batch = request.form.get('is_batch', False)
    
    messages = eval(messages)

    print("Get text_prompt:", messages)

    try:
        with lock:
            if media_type == 'video':
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                    tmp.write(video.read())
                    tmp_path = tmp.name
                    print(f"Saved temp video to: {tmp_path}")
                    # Fix messages to use the actual file path
                    for msg in messages:
                        for m in msg:
                            for item in m["content"]:
                                if item["type"] == "video":
                                    item["video"] = "file://" + tmp_path
                    vlm_output = vlm_generate(messages=messages, batch_inference=is_batch)
            elif media_type == 'image':
                vlm_output = vlm_generate(messages=messages, image_data=image, batch_inference=is_batch)
        if isinstance(vlm_output, list):
            if not is_batch:
                vlm_output = vlm_output[0]
            else:
                ## output will be a list
                ...
        return jsonify(
            {"response": vlm_output})
    except:
        traceback.print_exc()
        return jsonify({"error": traceback.format_exc()}), 500

if __name__ == '__main__':
    app.run(debug=False, host="0.0.0.0", port=17259)
