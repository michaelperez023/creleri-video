"""LLaMA-3.3 text service (POST /llama33), backed by the endpoint configured in uf_vlm_parser."""

import argparse
import json
import logging
import os
import threading

from flask import Flask, jsonify, request
from uf_vlm_parser import UFVLMParser

logger = logging.getLogger(__name__)

app = Flask(__name__)
lock = threading.Lock()
vlm_parser = None  # created at startup


def _parse_bool(value) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _from_json(value, field: str):
    """Fields may arrive as JSON or as a JSON-encoded string."""
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError as e:
        raise ValueError(f"'{field}' is not valid JSON: {e}") from e


def _is_conversation(value) -> bool:
    return isinstance(value, list) and bool(value) and all(isinstance(t, dict) for t in value)


@app.route("/llama33", methods=["POST"])
def llama33():
    """
    JSON body (messages and constraint_words may also be JSON-encoded strings):
      - messages: one conversation [{"role": ..., "content": ...}, ...], or a list of
        conversations when is_batch is set
      - constraint_words: optional list of labels; each conversation is then answered with
        one of them (or "none")
      - is_batch: optional bool or truthy string
    Returns {"response": str}, or {"response": [str, ...]} in batch mode.
    """
    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"error": "Request must be JSON"}), 415
    if not isinstance(data, dict):
        return jsonify({"error": "request body must be a JSON object"}), 400
    if data.get("messages") is None:
        return jsonify({"error": "missing 'messages'"}), 400

    is_batch = _parse_bool(data.get("is_batch", False))
    try:
        messages = _from_json(data["messages"], "messages")
        convs = messages if is_batch else [messages]
        if not (isinstance(convs, list) and all(_is_conversation(c) for c in convs)):
            expected = "a list of conversations" if is_batch else "one conversation"
            raise ValueError(
                f"'messages' must be {expected} (a conversation is a non-empty list of turn objects)"
            )

        constraint_words = _from_json(data.get("constraint_words"), "constraint_words")
        if constraint_words is not None and not (
            isinstance(constraint_words, list) and all(isinstance(w, str) for w in constraint_words)
        ):
            raise ValueError("'constraint_words' must be a list of strings")
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    try:
        with lock:
            if constraint_words:
                out = [vlm_parser.choose_from_enum(conv, constraint_words) for conv in convs]
            else:
                out = [vlm_parser.query_llm(conv) for conv in convs]
    except Exception as e:
        logger.exception("LLM request failed")
        return jsonify({"error": f"LLM request failed ({type(e).__name__})"}), 500

    return jsonify({"response": out if is_batch else out[0]})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--host",
        default=os.environ.get("CRELERI_HOST", "127.0.0.1"),
        help="interface to bind (default: $CRELERI_HOST or 127.0.0.1)",
    )
    parser.add_argument("--port", type=int, default=5003, help="port (default: %(default)s)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    vlm_parser = UFVLMParser()
    app.run(host=args.host, port=args.port, debug=False)
