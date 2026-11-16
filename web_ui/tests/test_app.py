"""Route tests for the web UI. The segmentation and grounding APIs are replaced by stubs."""

import io
import json
import re
import sys
from types import SimpleNamespace

import app as webui
import pytest
import requests


class FakeResponse(requests.Response):
    def __init__(self, status=200, body=b"", headers=None, json_body=None):
        super().__init__()
        headers = dict(headers or {})
        if json_body is not None:
            body = json.dumps(json_body).encode()
            headers.setdefault("Content-Type", "application/json")
        self.status_code = status
        self._content = body
        self.raw = io.BytesIO(body)
        self.headers.update(headers)
        self.closed = False

    def close(self):
        self.closed = True
        super().close()


@pytest.fixture
def client():
    return webui.app.test_client()


@pytest.fixture
def backend(monkeypatch):
    """Record outgoing requests and answer them from a queue of responses or exceptions."""
    stub = SimpleNamespace(calls=[], responses=[])

    def fake(method):
        def send(url, **kwargs):
            if "files" in kwargs:
                name, stream, mimetype = kwargs["files"]["file"]
                kwargs["files"] = {"file": (name, stream.read(), mimetype)}
            stub.calls.append(SimpleNamespace(method=method, url=url, **kwargs))
            result = stub.responses.pop(0)
            if isinstance(result, Exception):
                raise result
            return result

        return send

    monkeypatch.setattr(webui.requests, "get", fake("GET"))
    monkeypatch.setattr(webui.requests, "post", fake("POST"))
    return stub


def upload(client, content=b"video bytes", filename="my clip.mov", **form):
    data = {"file": (io.BytesIO(content), filename), **form}
    return client.post("/analyze_video", data=data, content_type="multipart/form-data")


# Uploads


def test_upload_is_forwarded_under_a_fresh_mp4_name(client, backend):
    backend.responses += [FakeResponse(201, json_body={"task_id": "t1"})] * 2

    first = upload(client, segmentation_type="1")
    upload(client, segmentation_type="1")

    assert first.status_code == 200
    assert first.get_json() == {"task_id": "t1"}
    call = backend.calls[0]
    assert call.url == f"{webui.SEGMENTATION_API_URL}/analyze_video"
    name, content, mimetype = call.files["file"]
    assert re.fullmatch(r"[0-9a-f]{32}\.mp4", name)
    assert (content, mimetype) == (b"video bytes", "video/mp4")
    assert backend.calls[1].files["file"][0] != name


@pytest.mark.parametrize(("form", "forwarded"), [({}, "0"), ({"segmentation_type": "2"}, "2")])
def test_segmentation_type_is_forwarded_as_query_parameter(client, backend, form, forwarded):
    backend.responses.append(FakeResponse(201, json_body={"task_id": "t1"}))

    assert upload(client, **form).status_code == 200
    assert backend.calls[0].params == {"segmentation_type": forwarded}


@pytest.mark.parametrize("value", ["3", "-1", "abc", ""])
def test_invalid_segmentation_type_is_rejected(client, backend, value):
    response = upload(client, segmentation_type=value)

    assert response.status_code == 400
    assert "segmentation_type" in response.get_json()["error"]
    assert backend.calls == []


def test_segmentation_mode_is_no_longer_server_state(client):
    response = client.post("/set_backend_segmentation_type", json={"segmentationType": 1})
    assert response.status_code == 404


def test_upload_without_file_is_rejected(client, backend):
    response = client.post("/analyze_video", data={}, content_type="multipart/form-data")

    assert response.status_code == 400
    assert backend.calls == []


def test_backend_rejection_keeps_the_client_error_and_message(client, backend):
    backend.responses.append(
        FakeResponse(400, json_body={"detail": "Uploaded file must be a video."})
    )

    response = upload(client)

    assert response.status_code == 400
    assert response.get_json()["detail"] == "Uploaded file must be a video."


def test_backend_server_error_is_a_bad_gateway(client, backend):
    backend.responses.append(
        FakeResponse(500, b"Internal Server Error", {"Content-Type": "text/plain"})
    )

    response = upload(client)

    assert response.status_code == 502
    assert response.get_json()["detail"] == "Internal Server Error"


def test_unreachable_backend_fails_the_upload_without_exception_details(client, backend):
    backend.responses.append(requests.ConnectionError("HTTPConnectionPool(host='localhost')"))

    response = upload(client)

    assert response.status_code == 502
    assert response.get_json() == {"error": "Segmentation API unreachable"}


# Upload limit


def test_oversize_upload_gets_a_json_413(client, backend, monkeypatch):
    monkeypatch.setitem(webui.app.config, "MAX_CONTENT_LENGTH", 1024)

    response = upload(client, content=b"x" * 4096)

    assert response.status_code == 413
    assert "upload limit" in response.get_json()["error"]
    assert backend.calls == []


def run_main(monkeypatch, *argv, host_env=None):
    seen = {}

    async def fake_serve(app, config):
        seen["config"] = config

    monkeypatch.setattr(webui, "serve", fake_serve)
    monkeypatch.setattr(sys, "argv", ["app.py", *argv])
    if host_env is None:
        monkeypatch.delenv("CRELERI_HOST", raising=False)
    else:
        monkeypatch.setenv("CRELERI_HOST", host_env)
    webui.main()
    return seen["config"]


def test_hypercorn_accepts_every_body_flask_accepts(monkeypatch):
    config = run_main(monkeypatch)

    assert webui.app.config["MAX_CONTENT_LENGTH"] == webui.MAX_UPLOAD_BYTES
    # Below Flask's limit Hypercorn would answer an empty 400 before the 413 handler can run.
    assert config.wsgi_max_body_size > webui.app.config["MAX_CONTENT_LENGTH"]


def test_page_tells_the_browser_the_upload_limit(client):
    page = client.get("/").get_data(as_text=True)

    assert f'data-max-bytes="{webui.MAX_UPLOAD_BYTES}"' in page


@pytest.mark.parametrize(
    ("argv", "host_env", "bind"),
    [
        ((), None, "127.0.0.1:5004"),
        (("5123",), None, "127.0.0.1:5123"),
        (("5123",), "0.0.0.0", "0.0.0.0:5123"),
        (("5123", "--host", "10.0.0.5"), "0.0.0.0", "10.0.0.5:5123"),
    ],
)
def test_bind_address(monkeypatch, argv, host_env, bind):
    assert run_main(monkeypatch, *argv, host_env=host_env).bind == [bind]


# Task status


def test_completed_status_points_thumbnails_at_the_proxy(client, backend):
    backend.responses.append(
        FakeResponse(
            json_body={
                "status": "Completed",
                "result": {
                    "proposals": {"shared/v.mp4": [{"segment": [0, 1], "score": 1}]},
                    "results": [],
                    "thumbnails": [
                        {
                            "thumbnail_url": "http://localhost:17249/thumbnails/a.jpg",
                            "segment": [0, 1],
                        },
                        {"thumbnail": "b.jpg", "segment": [1, 2]},
                        {"segment": [2, 3]},
                    ],
                },
            }
        )
    )

    response = client.get("/task_status/t1")

    assert response.status_code == 200
    assert backend.calls[0].url == f"{webui.SEGMENTATION_API_URL}/task_status/t1"
    assert response.get_json()["result"]["thumbnails"] == [
        {"segment": [0, 1], "thumbnail_url": "/proxy/thumbnails/a.jpg"},
        {"segment": [1, 2], "thumbnail_url": "/proxy/thumbnails/b.jpg"},
    ]


@pytest.mark.parametrize(
    "status", [{"status": "Processing"}, {"status": "Failed", "error": "boom"}]
)
def test_other_statuses_pass_through(client, backend, status):
    backend.responses.append(FakeResponse(json_body=status))

    response = client.get("/task_status/t1")

    assert (response.status_code, response.get_json()) == (200, status)


@pytest.mark.parametrize("error", [requests.ConnectionError("refused"), requests.Timeout("slow")])
def test_unreachable_segmentation_api_is_reported_as_transient(client, backend, error):
    backend.responses.append(error)

    response = client.get("/task_status/t1")

    assert response.status_code == 503
    assert response.get_json()["status"] == "Unavailable"


# Proxies

UPSTREAM_HEADERS = {
    "Server": "uvicorn",
    "Date": "Mon, 16 Nov 2026 00:00:00 GMT",
    "Content-Encoding": "gzip",
    "Transfer-Encoding": "chunked",
    "Connection": "keep-alive",
    "Content-Length": "999",
    "ETag": '"v1"',
}
DROPPED = ("Server", "Date", "Content-Encoding", "Transfer-Encoding", "Connection")


def test_grounding_result_forwards_only_payload_headers(client, backend):
    body = b'{"status":"PENDING"}'
    backend.responses.append(
        FakeResponse(body=body, headers={**UPSTREAM_HEADERS, "Content-Type": "application/json"})
    )

    response = client.get("/video/result/j1")

    assert response.get_json() == {"status": "PENDING"}
    assert response.headers["Content-Type"] == "application/json"
    assert response.headers["ETag"] == '"v1"'
    assert response.headers["Content-Length"] == str(len(body))
    assert not [name for name in DROPPED if name in response.headers]


def test_thumbnail_proxy_keeps_status_and_type(client, backend):
    backend.responses += [
        FakeResponse(
            body=b"\xff\xd8jpeg", headers={**UPSTREAM_HEADERS, "Content-Type": "image/jpeg"}
        ),
        FakeResponse(404, json_body={"detail": "Not Found"}),
    ]

    found = client.get("/proxy/thumbnails/a.jpg")
    missing = client.get("/proxy/thumbnails/b.jpg")

    assert (found.status_code, found.mimetype, found.data) == (200, "image/jpeg", b"\xff\xd8jpeg")
    assert not [name for name in DROPPED if name in found.headers]
    assert (missing.status_code, missing.mimetype) == (404, "application/json")


def test_result_video_is_streamed_and_upstream_closed(client, backend):
    upstream = FakeResponse(
        body=b"\x00\x00mp4", headers={**UPSTREAM_HEADERS, "Content-Type": "video/mp4"}
    )
    backend.responses.append(upstream)

    response = client.get("/video/get_result/j1")

    assert backend.calls[0].stream is True
    assert response.data == b"\x00\x00mp4"
    assert response.mimetype == "video/mp4"
    assert not [name for name in DROPPED if name in response.headers]
    response.close()
    assert upstream.closed


@pytest.mark.parametrize(
    ("path", "status", "body"),
    [
        ("/proxy/thumbnails/a.jpg", 502, {"error": "Segmentation API unreachable"}),
        (
            "/task_status/t1",
            503,
            {"status": "Unavailable", "error": "Segmentation API unreachable"},
        ),
        ("/video/result/j1", 502, {"error": "Grounding API unreachable"}),
        ("/video/get_result/j1", 502, {"error": "Grounding API unreachable"}),
    ],
)
def test_unreachable_backend_is_reported_without_exception_details(
    client, backend, path, status, body
):
    backend.responses.append(requests.ConnectionError("HTTPConnectionPool(host='localhost')"))

    response = client.get(path)

    assert (response.status_code, response.get_json()) == (status, body)
