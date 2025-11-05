from flask import Flask, request, jsonify
import traceback

from vlm_utils import (clean_object_name, get_first_mid_frame, query_vlm,
                   ObjectProposer, VLMParser)

object_proposer = ObjectProposer()

app = Flask(__name__)


@app.route('/llama32', methods=['POST'])
def gsam():
    if 'video' not in request.files:
        return jsonify({'error': 'no video file found'}), 400

    video = request.files['video']
    if video.filename == '':
        return jsonify({'error': 'no chosen file'}), 400
    
    messages = request.form['messages']
    # video_path = request.form['video_path']
    
    messages = eval(messages)
    

    print("Get text_prompt:", messages)

    try:
        output_object = object_proposer.get_objects_in_video(video.read(), messages=messages)
        return jsonify(
            {"response": output_object})
    except:
        traceback.print_exc()
        return jsonify({"error": traceback.format_exc()}), 500


if __name__ == '__main__':
    app.run(debug=False, host="0.0.0.0", port=5002)
