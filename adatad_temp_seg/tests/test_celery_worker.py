"""process_video_task end to end, with the GPU, video and HTTP stack replaced by small fakes."""

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

WORKER_PATH = Path(__file__).resolve().parents[1] / "celery_worker.py"
FPS, FRAMES = 10, 90  # a 9-second video
VOCABULARY = {
    "action_name": ["cut", "walk"],
    '"as in…"': ["slice", ""],
    "example sentence from propbank": ["", ""],
}
ARGUMENTS_REPLY = '```json\n{"argument0": "man", "argument1": "onion", "argument2": null}\n```'


class RequestException(OSError):
    pass


class HTTPError(RequestException):
    pass


class Response:
    def __init__(self, payload, status=200):
        self.payload = payload
        self.status = status

    def json(self):
        return self.payload

    def raise_for_status(self):
        if self.status >= 400:
            raise HTTPError(f"{self.status} Server Error")


def detected(caption):
    return Response({"objects": ["object"], "captions": [[caption]]})


def decoded(*shape):
    array = np.zeros(shape, np.uint8)
    return SimpleNamespace(asnumpy=lambda: array)


class FakeVideo:
    """A decord reader over FRAMES blank frames, and cv2 writers that count what they write."""

    def __init__(self):
        self.batch_sizes = []
        self.clips = {}  # clip path -> frames written

    def __len__(self):
        return FRAMES

    def get_avg_fps(self):
        return FPS

    def __getitem__(self, index):
        return decoded(4, 6, 3)

    def get_batch(self, indices):
        self.batch_sizes.append(len(indices))
        return decoded(len(indices), 4, 6, 3)

    def writer(self, path, fourcc, fps, size):
        Path(path).touch()
        self.clips[path] = 0

        def write(frame):
            self.clips[path] += 1

        return SimpleNamespace(write=write, release=lambda: None)


class FakeServices:
    """The object proposer, LLaMA and grounding services behind requests.post."""

    def __init__(self):
        self.calls = []  # service of every request, in order
        self.detections = []  # object proposer responses for the next clips

    def post(self, url, timeout=None, **request):
        assert timeout is not None, f"no timeout for {url}"
        service = url.rsplit("/", 1)[-1]
        self.calls.append(service)
        if service == "process_video":
            assert Path(request["json"]["video_path"]).is_file()
            return self.detections.pop(0) if self.detections else detected("a cat sleeps")
        if service == "llama33":
            messages = request["json"]["messages"]
            if "constraint_words" not in request["json"]:
                return Response({"response": [ARGUMENTS_REPLY] * len(messages)})
            labels = ["cut" if "onion" in user["content"] else "none" for _, user in messages]
            return Response({"response": labels})
        assert service == "ground", url
        pairs = json.loads(request["data"]["inputs"])["subject_object_pairs"]
        return Response([f"job-{i}" for i in range(len(pairs))])


class FakeCelery:
    def __init__(self, *args, **kwargs):
        self.conf = SimpleNamespace(update=lambda **settings: None)

    def task(self, **options):
        return lambda function: function


class OneClusterEach:
    """DBSCAN stand-in that puts every prediction in a cluster of its own."""

    def __init__(self, **params):
        pass

    def fit_predict(self, distances):
        return np.arange(len(distances))


def stub(name, **attrs):
    module = ModuleType(name)
    module.__dict__.update(attrs)
    return module


def stub_modules(video, services):
    return [
        stub("celery", Celery=FakeCelery),
        stub("coloredlogs"),
        stub(
            "cv2",
            COLOR_RGB2BGR=4,
            cvtColor=lambda image, code: image,
            imwrite=lambda path, image: Path(path).touch(),
            VideoWriter=video.writer,
            VideoWriter_fourcc=lambda *code: 0,
        ),
        stub("decord", VideoReader=lambda path, ctx=None: video, cpu=lambda: None),
        stub("ffmpeg", probe=lambda path: {"format": {"duration": str(FRAMES / FPS)}}),
        stub("pandas", read_csv=lambda path, **options: VOCABULARY),
        stub("requests", RequestException=RequestException, post=services.post),
        stub("sentence_transformers", SentenceTransformer=None),
        stub("sklearn"),
        stub("sklearn.cluster", DBSCAN=OneClusterEach),
        stub("sklearn.metrics"),
        stub("sklearn.metrics.pairwise", cosine_similarity=lambda x: x @ x.T),
        stub("torch"),
        stub("tqdm", tqdm=None),
        stub("mmengine"),
        stub("mmengine.config", Config=None),
        stub("opentad"),
        stub("opentad.cores", eval_one_epoch_single_video=None),
        stub("opentad.datasets", build_dataloader=None, build_dataset=None),
        stub("opentad.models", build_detector=None),
        stub("opentad.utils", set_seed=None),
    ]


@pytest.fixture
def env(monkeypatch, tmp_path):
    video, services = FakeVideo(), FakeServices()
    for module in stub_modules(video, services):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    spec = importlib.util.spec_from_file_location("celery_worker", WORKER_PATH)
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)

    worker.BASE_DIR = tmp_path
    worker.SHARED_DIR = tmp_path / "shared"
    worker.THUMBNAIL_DIR = tmp_path / "thumbnails"
    worker.SHARED_DIR.mkdir()
    worker.THUMBNAIL_DIR.mkdir()
    worker.model = object()
    worker.sentence_transformer_model = SimpleNamespace(encode=lambda texts: np.eye(len(texts)))

    upload = worker.SHARED_DIR / "0a1b2c.mp4"
    upload.write_bytes(b"video")
    return SimpleNamespace(worker=worker, video=video, services=services, upload=upload)


def test_each_segment_is_analyzed_on_its_own(env):
    stem = env.upload.stem
    spans = [[0.0, 4.0], [4.0, 4.01], [4.01, 7.0], [7.0, 9.0]]
    env.worker.segment_video = lambda path: {stem: [{"segment": s, "score": 0.0} for s in spans]}
    env.services.detections = [
        detected("a man cuts an onion"),
        Response({"error": "CUDA out of memory"}, status=500),
        detected("a cat sleeps"),
    ]
    env.worker.CLIP_CHUNK_FRAMES = 16  # shorter than every segment

    result = env.worker.process_video_task(str(env.upload), 0)

    json.dumps(result)  # Celery stores results as JSON
    # [4.0, 4.01] spans no frame at 10 fps, so it is dropped from proposals and results alike.
    proposals = result["proposals"][stem]
    assert [p["segment"] for p in proposals] == [[0.0, 4.0], [4.01, 7.0], [7.0, 9.0]]
    assert [r["segment"] for r in result["results"]] == proposals
    assert result["thumbnails"] == [
        {"thumbnail": p["thumbnail"], "segment": p["segment"]} for p in proposals
    ]

    cut, failed, idle = result["results"]
    assert cut["results"] == {
        "valid_parsed_actions_3": [
            {
                "pattern": {"action": "cut", "argument0": "man", "argument1": "onion"},
                "probability": 1.0,
                "caption": "a man cuts an onion",
            }
        ]
    }
    assert cut["api_response"] == ["job-0"]
    assert (failed["results"], failed["api_response"]) == ({}, [])
    assert failed["error"] == "HTTPError: 500 Server Error"
    assert (idle["results"], idle["api_response"]) == ({}, [])
    assert "error" not in cut and "error" not in idle
    assert env.services.calls.count("ground") == 1

    # Every frame reaches its clip, but no more than one chunk is decoded at a time.
    assert list(env.video.clips.values()) == [40, 30, 20]
    assert max(env.video.batch_sizes) == 16
    assert all((env.worker.THUMBNAIL_DIR / p["thumbnail"]).is_file() for p in proposals)
    # The upload and the clips are removed; thumbnails stay for the UI.
    assert not any(env.worker.SHARED_DIR.iterdir())


@pytest.mark.parametrize(
    ("segmentation_type", "spans"), [(1, [[0.0, 5.0], [5.0, 9.0]]), (2, [[0, 9.0]])]
)
def test_segments_are_keyed_by_the_upload_name(env, segmentation_type, spans):
    result = env.worker.process_video_task(str(env.upload), segmentation_type)

    assert list(result["proposals"]) == [env.upload.stem]
    assert [p["segment"] for p in result["proposals"][env.upload.stem]] == spans
    assert len(result["results"]) == len(spans)


def test_upload_is_deleted_when_the_task_fails(env):
    with pytest.raises(ValueError, match="SegmentationType"):
        env.worker.process_video_task(str(env.upload), 7)
    assert not env.upload.exists()


def test_files_outside_the_upload_directory_are_left_alone(env):
    other = env.worker.BASE_DIR / "notes.mp4"
    other.write_bytes(b"keep")
    for path in [str(other), "shared/../notes.mp4"]:
        with pytest.raises(ValueError, match="shared/ upload directory"):
            env.worker.process_video_task(path, 7)
    assert other.read_bytes() == b"keep"
