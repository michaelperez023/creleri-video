import importlib
import io
import json
import os
import runpy
import sys
from pathlib import Path
from unittest import mock

import pytest


@pytest.fixture(scope="module")
def qwen_api():
    # Request handling does not touch the model stack, so import the service without it.
    with pytest.MonkeyPatch.context() as mp:
        for name in ("torch", "transformers", "qwen_vl_utils"):
            mp.setitem(sys.modules, name, mock.MagicMock())
        return importlib.import_module("qwen2p5_api")


def media_paths(convs):
    paths = []
    for conv in convs:
        for turn in conv:
            if isinstance(turn["content"], list):
                paths += [p[p["type"]] for p in turn["content"] if p["type"] in ("image", "video")]
    return paths


class FakeGenerate:
    """Stands in for vlm_generate and records the media the model would have loaded."""

    def __init__(self):
        self.error = None
        self.convs = None
        self.paths = []
        self.contents = []

    def __call__(self, convs, max_new_tokens=128):
        self.convs = convs
        self.paths = media_paths(convs)
        self.contents = [Path(p).read_bytes() for p in self.paths]
        if self.error:
            raise self.error
        return [f"caption {i}" for i in range(len(convs))]


@pytest.fixture
def generate(qwen_api, monkeypatch):
    fake = FakeGenerate()
    monkeypatch.setattr(qwen_api, "vlm_generate", fake)
    return fake


@pytest.fixture
def client(qwen_api):
    return qwen_api.app.test_client()


def video_conv(question="What is the person doing in the video?"):
    video = {"type": "video", "video": "file://", "max_pixels": 360 * 420, "fps": 1.0}
    return [{"role": "user", "content": [video, {"type": "text", "text": question}]}]


def post(client, messages, is_batch=False, video=True, images=0):
    data = {"messages": messages if isinstance(messages, str) else json.dumps(messages)}
    if is_batch:
        data["is_batch"] = "true"
    if video:
        data["video"] = (io.BytesIO(b"mp4 bytes"), "clip.mp4")
    if images:
        data["image"] = [(io.BytesIO(f"image {i}".encode()), f"{i}.png") for i in range(images)]
    return client.post("/qwen", data=data, content_type="multipart/form-data")


def test_batch_video_request(client, generate):
    resp = post(client, [video_conv(), video_conv()], is_batch=True)
    assert resp.status_code == 200
    assert resp.get_json() == {"response": ["caption 0", "caption 1"]}
    assert generate.contents == [b"mp4 bytes", b"mp4 bytes"]
    assert not any(os.path.exists(p) for p in generate.paths)


def test_uploads_are_removed_when_inference_fails(client, generate):
    generate.error = RuntimeError("CUDA error in /home/someone/qwen2p5_api.py")
    resp = post(client, video_conv())
    assert resp.status_code == 500
    assert resp.get_json() == {"error": "inference failed (RuntimeError)"}
    assert generate.paths
    assert not any(os.path.exists(p) for p in generate.paths)


@pytest.mark.parametrize(
    "item",
    [
        {"type": "image", "image": "/etc/hosts"},
        {"type": "image_url", "image_url": "http://169.254.169.254/latest/meta-data/"},
        {"type": "text", "text": "and this", "video": "/data/other_user/clip.mp4"},
        {"type": "image"},
    ],
)
def test_rejects_media_other_than_this_upload(client, generate, item):
    conv = video_conv()
    conv[0]["content"].append(item)
    assert post(client, conv).status_code == 400
    assert generate.convs is None


def test_single_conversation_images_keep_upload_order(client, generate):
    conv = [{"role": "user", "content": [{"type": "text", "text": "Compare the images."}]}]
    resp = post(client, conv, video=False, images=3)
    assert resp.status_code == 200
    assert resp.get_json() == {"response": "caption 0"}
    assert generate.contents == [b"image 0", b"image 1", b"image 2"]
    assert generate.convs[0][0]["content"][-1] == {"type": "text", "text": "Compare the images."}


def test_string_content_turns_are_accepted(client, generate):
    conv = [{"role": "system", "content": "Be concise."}, *video_conv()]
    resp = post(client, [conv], is_batch=True)
    assert resp.status_code == 200
    assert generate.contents == [b"mp4 bytes"]


@pytest.mark.parametrize(
    ("messages", "is_batch"),
    [
        ("not json", False),
        ([video_conv()], False),
        (video_conv(), True),
        ([], False),
        ([{"role": "user"}], False),
        ([{"role": "user", "content": [{"text": "no type"}]}], False),
    ],
)
def test_malformed_messages_get_400(client, generate, messages, is_batch):
    assert post(client, messages, is_batch=is_batch).status_code == 400
    assert generate.convs is None


def test_requires_an_upload(client, generate):
    assert post(client, video_conv(), video=False).status_code == 400


def test_generate_trims_left_padded_prompts(qwen_api, monkeypatch):
    np = pytest.importorskip("numpy")
    pad = 0
    prompts = np.array([[11, 12, 13, 14, 15], [pad, pad, 21, 22, 23]])
    answers = np.array([[91, 92], [81, 82]])

    class Inputs(dict):
        @property
        def input_ids(self):
            return self["input_ids"]

        def to(self, device):
            return self

    class Processor:
        def apply_chat_template(self, conv, **kwargs):
            return "prompt"

        def __call__(self, text, **kwargs):
            return Inputs(input_ids=prompts[: len(text)])

        def batch_decode(self, sequences, **kwargs):
            return [row.tolist() for row in sequences]

    class Model:
        device = "cpu"

        def generate(self, input_ids, max_new_tokens):
            return np.concatenate([input_ids, answers[: len(input_ids)]], axis=1)

    monkeypatch.setattr(qwen_api, "processor", Processor())
    monkeypatch.setattr(qwen_api, "model", Model())
    monkeypatch.setattr(qwen_api, "process_vision_info", lambda conv: (None, None))
    assert qwen_api.vlm_generate([video_conv(), video_conv()]) == [[91, 92], [81, 82]]
    # a single conversation still gets a single answer
    assert qwen_api.vlm_generate(video_conv()) == [91, 92]


@pytest.mark.parametrize(
    ("argv", "env_host", "max_memory", "attn", "host", "port"),
    [
        (
            [],
            None,
            {0: "1GiB", 1: "140GiB", 2: "140GiB", 3: "140GiB"},
            "flash_attention_2",
            "127.0.0.1",
            17259,
        ),
        (
            ["--max-memory", "70GiB", "--attn-implementation", "sdpa", "--port", "18000"],
            "0.0.0.0",
            {0: "70GiB"},
            "sdpa",
            "0.0.0.0",
            18000,
        ),
    ],
)
def test_startup_loads_the_model(monkeypatch, argv, env_host, max_memory, attn, host, port):
    import flask

    stack = {name: mock.MagicMock() for name in ("torch", "transformers", "qwen_vl_utils")}
    for name, module in stack.items():
        monkeypatch.setitem(sys.modules, name, module)
    runs = []
    monkeypatch.setattr(flask.Flask, "run", lambda self, **kwargs: runs.append(kwargs))
    if env_host is None:
        monkeypatch.delenv("CRELERI_HOST", raising=False)
    else:
        monkeypatch.setenv("CRELERI_HOST", env_host)
    service = Path(__file__).resolve().parent.parent / "qwen2p5_api.py"
    monkeypatch.setattr(sys, "argv", [str(service), *argv])
    runpy.run_path(str(service), run_name="__main__")

    model_cls = stack["transformers"].Qwen2_5_VLForConditionalGeneration
    model_cls.from_pretrained.assert_called_once_with(
        "Qwen/Qwen2.5-VL-32B-Instruct",
        dtype=stack["torch"].bfloat16,
        attn_implementation=attn,
        device_map="auto",
        max_memory=max_memory,
    )
    processor_cls = stack["transformers"].AutoProcessor
    processor_cls.from_pretrained.assert_called_once_with("Qwen/Qwen2.5-VL-32B-Instruct")
    tok = processor_cls.from_pretrained.return_value.tokenizer
    model = model_cls.from_pretrained.return_value
    assert (tok.padding_side, tok.truncation_side) == ("left", "left")
    assert model.config.pad_token_id is tok.pad_token_id
    assert model.generation_config.pad_token_id is tok.pad_token_id
    assert runs == [{"host": host, "port": port, "debug": False}]
