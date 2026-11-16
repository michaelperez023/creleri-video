import re
from types import SimpleNamespace

import api
import pytest
from fastapi.testclient import TestClient

BOUNDARY = "creleri-test-boundary"


@pytest.fixture
def shared_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "SHARED_DIR", tmp_path)
    return tmp_path


@pytest.fixture
def queued(monkeypatch):
    calls = []

    def send_task(name, args):
        calls.append((name, args))
        return SimpleNamespace(id="task-1")

    monkeypatch.setattr(api.celery_app, "send_task", send_task)
    return calls


@pytest.fixture
def client():
    return TestClient(api.app)


def upload(client, filename="video.mp4", content=b"frames", content_type="video/mp4", params=None):
    # Built by hand because HTTP clients fill in a missing part Content-Type themselves.
    part_headers = f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
    if content_type is not None:
        part_headers += f"Content-Type: {content_type}\r\n"
    body = f"--{BOUNDARY}\r\n{part_headers}\r\n".encode() + content
    body += f"\r\n--{BOUNDARY}--\r\n".encode()
    return client.post(
        "/analyze_video",
        params={"segmentation_type": 0} if params is None else params,
        content=body,
        headers={"Content-Type": f"multipart/form-data; boundary={BOUNDARY}"},
    )


def test_upload_is_stored_under_a_new_name_and_queued(client, shared_dir, queued):
    content = bytes(range(256)) * 12_000  # several copy chunks
    response = upload(client, filename="../../escape.mov", content=content)

    assert response.status_code == 201
    assert response.json() == {"task_id": "task-1"}
    [(task_name, (path, segmentation_type))] = queued
    assert task_name == "tasks.process_video_task"
    [saved] = shared_dir.iterdir()
    assert str(saved) == path
    assert re.fullmatch(r"[0-9a-f]{32}\.mp4", saved.name)
    assert saved.read_bytes() == content
    assert segmentation_type == 0
    assert type(segmentation_type) is int


@pytest.mark.parametrize("segmentation_type", ["0", "1", "2"])
def test_known_segmentation_types_are_queued(client, shared_dir, queued, segmentation_type):
    response = upload(client, params={"segmentation_type": segmentation_type})
    assert response.status_code == 201
    assert queued[0][1][1] == int(segmentation_type)


@pytest.mark.parametrize("params", [{"segmentation_type": "3"}, {"segmentation_type": "x"}, {}])
def test_invalid_segmentation_type_is_rejected(client, shared_dir, queued, params):
    assert upload(client, params=params).status_code == 422
    assert not queued
    assert not any(shared_dir.iterdir())


@pytest.mark.parametrize(
    ("filename", "content_type", "status"),
    [
        ("clip.mp4", None, 201),
        ("clip.mov", "application/octet-stream", 201),
        ("clip.mp4", "video/webm", 201),
        ("notes.txt", None, 415),
        ("clip", None, 415),
        ("clip.mp4", "text/plain", 415),
    ],
)
def test_content_type_check(client, shared_dir, queued, filename, content_type, status):
    assert upload(client, filename=filename, content_type=content_type).status_code == status
    expected = 1 if status == 201 else 0
    assert len(queued) == expected
    assert len(list(shared_dir.iterdir())) == expected


def test_upload_is_removed_when_queueing_fails(client, shared_dir, monkeypatch):
    def send_task(name, args):
        raise ConnectionError("redis is down")

    monkeypatch.setattr(api.celery_app, "send_task", send_task)
    with pytest.raises(ConnectionError):
        upload(client)
    assert not any(shared_dir.iterdir())


def fake_task(monkeypatch, state, result=None, info=None):
    task = SimpleNamespace(state=state, result=result, info=info)
    monkeypatch.setattr(api, "AsyncResult", lambda task_id, app: task)


@pytest.mark.parametrize("state", ["PENDING", "RECEIVED", "STARTED", "RETRY"])
def test_unfinished_states_are_processing(client, monkeypatch, state):
    fake_task(monkeypatch, state)
    assert client.get("/task_status/t").json() == {"status": "Processing"}


@pytest.mark.parametrize("state", ["FAILURE", "REVOKED", "REJECTED"])
def test_failed_states_report_the_error(client, monkeypatch, state):
    fake_task(monkeypatch, state, info=RuntimeError("boom"))
    assert client.get("/task_status/t").json() == {"status": "Failed", "error": "boom"}


def test_completed_task_links_thumbnails(client, monkeypatch):
    result = {
        "proposals": {"video": [{"segment": [0.0, 5.0], "score": 1, "thumbnail": "t0.jpg"}]},
        "results": [{"results": {}, "api_response": [], "thumbnail": "t0.jpg"}],
        "thumbnails": [{"thumbnail": "t0.jpg", "segment": [0.0, 5.0]}],
    }
    fake_task(monkeypatch, "SUCCESS", result=result)
    assert client.get("/task_status/t").json() == {
        "status": "Completed",
        "result": {
            "proposals": result["proposals"],
            "results": result["results"],
            "thumbnails": [
                {"thumbnail_url": "http://testserver/thumbnails/t0.jpg", "segment": [0.0, 5.0]}
            ],
        },
    }
