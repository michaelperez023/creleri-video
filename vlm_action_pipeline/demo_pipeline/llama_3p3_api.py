from flask import Flask, request, jsonify
import traceback, json, threading
from uf_vlm_parser import UFVLMParser

lock = threading.Lock()
vlm_parser = UFVLMParser(model_name=None)

app = Flask(__name__)

@app.route('/llama33', methods=['POST'])
def llama33():
    try:
        data = request.get_json(silent=True)
        if data is None:
            return jsonify({"error": "Request must be JSON"}), 415

        raw_messages = data.get("messages")
        if raw_messages is None:
            return jsonify({"error": "missing 'messages'"}), 400

        raw_constraints = data.get("constraint_words")
        constraint_words = None
        if raw_constraints is not None:
            constraint_words = raw_constraints if isinstance(raw_constraints, list) else json.loads(raw_constraints)

        is_batch = str(data.get("is_batch", False)).strip().lower() in {"1", "true", "yes", "y", "on"}

        # Ensure messages stays as JSON string for the parser (it handles both)
        if not isinstance(raw_messages, str):
            raw_messages = json.dumps(raw_messages)

        with lock:
            if is_batch:
                convs = json.loads(raw_messages) if isinstance(raw_messages, str) else raw_messages
                if constraint_words:
                    out = [vlm_parser.choose_from_enum(conv, constraint_words) for conv in convs]
                else:
                    out = vlm_parser.query_llm_batch(convs, None)
                llm_response = out
            else:
                conv = json.loads(raw_messages) if isinstance(raw_messages, str) else raw_messages
                if constraint_words:
                    llm_response = vlm_parser.choose_from_enum(conv, constraint_words)
                else:
                    llm_response = vlm_parser.query_llm(conv, None)

        return jsonify({"response": llm_response})

    except Exception:
        traceback.print_exc()
        return jsonify({"error": traceback.format_exc()}), 500

if __name__ == '__main__':
    app.run(debug=False, host="0.0.0.0", port=5003)
