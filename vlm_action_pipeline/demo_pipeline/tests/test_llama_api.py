import json
from types import SimpleNamespace

import llama_3p3_api
import pytest
from uf_vlm_parser import DEFAULT_MODEL, UFVLMParser

CONV = [
    {"role": "system", "content": "Pick exactly one action."},
    {"role": "user", "content": "Caption: a man throws a ball"},
]
LABELS = ["aim", "carry"]
SETTINGS = [
    "LLM_BASE_URL",
    "LLM_API_KEY",
    "LLM_MODEL",
    "UF_BASE_URL",
    "UF_API_KEY",
    "UF_MODEL",
    "OPENAI_BASE_URL",
    "OPENAI_API_KEY",
]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    # uf_vlm_parser loads any .env when imported; each test sets exactly what it needs.
    for name in SETTINGS:
        monkeypatch.delenv(name, raising=False)


class FakeParser:
    def __init__(self, error=None):
        self.error = error
        self.calls = []

    def query_llm(self, messages):
        if self.error:
            raise self.error
        self.calls.append(("query", messages))
        return f"answer {len(self.calls)}"

    def choose_from_enum(self, messages, choices):
        self.calls.append(("choose", messages, choices))
        return "none"


@pytest.fixture
def fake_parser(monkeypatch):
    parser = FakeParser()
    monkeypatch.setattr(llama_3p3_api, "vlm_parser", parser)
    return parser


@pytest.fixture
def client():
    return llama_3p3_api.app.test_client()


def test_rejects_non_json_body(client, fake_parser):
    assert client.post("/llama33", data="messages=hi").status_code == 415


@pytest.mark.parametrize(
    "body",
    [
        [CONV],
        {},
        {"messages": "[{"},
        {"messages": []},
        {"messages": [CONV]},
        {"messages": CONV, "is_batch": True},
        {"messages": [CONV, "not a conversation"], "is_batch": True},
        {"messages": CONV, "constraint_words": "[aim"},
        {"messages": CONV, "constraint_words": "aim"},
        {"messages": CONV, "constraint_words": ["aim", 3]},
    ],
)
def test_malformed_requests_get_400(client, fake_parser, body):
    resp = client.post("/llama33", json=body)
    assert resp.status_code == 400
    assert resp.get_json()["error"]
    assert fake_parser.calls == []


def test_single_conversation(client, fake_parser):
    resp = client.post("/llama33", json={"messages": CONV})
    assert resp.status_code == 200
    assert resp.get_json() == {"response": "answer 1"}
    assert fake_parser.calls == [("query", CONV)]


def test_batch_accepts_json_encoded_fields(client, fake_parser):
    body = {"messages": json.dumps([CONV, CONV]), "is_batch": "true"}
    resp = client.post("/llama33", json=body)
    assert resp.get_json() == {"response": ["answer 1", "answer 2"]}


def test_batch_with_constraint_words(client, fake_parser):
    body = {"messages": [CONV, CONV], "constraint_words": json.dumps(LABELS), "is_batch": True}
    resp = client.post("/llama33", json=body)
    assert resp.get_json() == {"response": ["none", "none"]}
    assert fake_parser.calls == [("choose", CONV, LABELS)] * 2


def test_empty_batch(client, fake_parser):
    resp = client.post("/llama33", json={"messages": [], "is_batch": True})
    assert resp.get_json() == {"response": []}


def test_llm_errors_are_logged_not_returned(client, monkeypatch, caplog):
    error = RuntimeError("upstream said: invalid key sk-123")
    monkeypatch.setattr(llama_3p3_api, "vlm_parser", FakeParser(error=error))
    resp = client.post("/llama33", json={"messages": CONV})
    assert resp.status_code == 500
    assert resp.get_json() == {"error": "LLM request failed (RuntimeError)"}
    assert "invalid key sk-123" in caplog.text


class FakeCompletions:
    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []

    def create(self, **request):
        self.requests.append(request)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=reply))])


def parser_with_replies(monkeypatch, *replies):
    monkeypatch.setenv("LLM_BASE_URL", "http://llm.test/v1")
    parser = UFVLMParser()
    completions = FakeCompletions(replies)
    parser.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return parser, completions


def test_choose_from_enum_uses_structured_label(monkeypatch):
    parser, _ = parser_with_replies(monkeypatch, '{"label": "carry"}')
    assert parser.choose_from_enum(CONV, LABELS) == "carry"


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("0", "aim"),
        ("1", "carry"),
        ("Carry", "carry"),
        ("none", "none"),
        ("I am not sure", "none"),
        ("7", "none"),
        (None, "none"),
    ],
)
def test_choose_from_enum_fallback_defaults_to_none(monkeypatch, answer, expected):
    parser, completions = parser_with_replies(monkeypatch, RuntimeError("no json_schema"), answer)
    assert parser.choose_from_enum(CONV, LABELS) == expected
    assert "If uncertain choose 2." in completions.requests[1]["messages"][0]["content"]


def test_query_llm_returns_empty_string_without_content(monkeypatch):
    parser, _ = parser_with_replies(monkeypatch, None)
    assert parser.query_llm(CONV) == ""


def test_requires_base_url():
    with pytest.raises(RuntimeError, match="LLM_BASE_URL"):
        UFVLMParser()


LLM_URL = {"LLM_BASE_URL": "http://new.test/v1"}
LLM_REST = {"LLM_API_KEY": "new-key", "LLM_MODEL": "new-model"}
UF_URL = {"UF_BASE_URL": "http://old.test/v1"}
UF_REST = {"UF_API_KEY": "old-key", "UF_MODEL": "old-model"}


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        (LLM_URL | LLM_REST | UF_URL | UF_REST, ("new.test", "new-key", "new-model")),
        (UF_URL | UF_REST, ("old.test", "old-key", "old-model")),
        # The base URL picks the set, so a key never goes to the other set's server.
        (LLM_URL | UF_REST, ("new.test", "", DEFAULT_MODEL)),
        (UF_URL | LLM_REST, ("old.test", "", DEFAULT_MODEL)),
    ],
    ids=["llm-over-uf", "uf-fallback", "llm-url-only", "uf-url-only"],
)
def test_settings_are_read_as_one_set(monkeypatch, env, expected):
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    parser = UFVLMParser()
    assert (parser.client.base_url.host, parser.client.api_key, parser.model) == expected


def test_defaults_never_borrow_the_openai_key(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "http://llm.test/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-meant-for-api.openai.com")
    parser = UFVLMParser()
    assert parser.model == "llama-3.3-70b-instruct"
    assert parser.client.api_key == ""
