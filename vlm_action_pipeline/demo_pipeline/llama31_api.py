from flask import Flask, request, jsonify
import traceback
import threading

lock = threading.Lock()

from vlm_utils import (clean_object_name, get_first_mid_frame, query_vlm,
                   ObjectProposer, VLMParser)

vlm_parser = VLMParser(model_name="meta-llama/Llama-3.1-70B-Instruct")

app = Flask(__name__)


@app.route('/llama31', methods=['POST'])
def gsam():
    messages = request.form['messages']
    constraint_words = request.form.get('constraint_words', None)
    is_batch = request.form.get('is_batch', False)

    messages = eval(messages)
    if constraint_words:
        constraint_words = eval(constraint_words)    

    print("Get text_prompt:", messages)

    try:
        with lock:
            if not is_batch:
                llm_response = vlm_parser.query_llm(messages=messages, constraint_words=constraint_words)
            else:
                llm_response = vlm_parser.query_llm_batch(messages=messages, constraint_words=constraint_words)
        return jsonify(
            {"response": llm_response})
    except:
        traceback.print_exc()
        return jsonify({"error": traceback.format_exc()}), 500


if __name__ == '__main__':
    app.run(debug=False, host="0.0.0.0", port=5003)
