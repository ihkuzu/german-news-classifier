import json

import httpx
import pytest

from newsclf.data import LABELS
from newsclf.llm import LLMError, OllamaClassifier, parse_label, system_prompt


def make(handler, **kwargs):
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return OllamaClassifier(host="http://ollama", client=client, **kwargs)


def reply(content: str) -> httpx.Response:
    return httpx.Response(200, json={"message": {"content": content}})


def test_prompt_lists_every_section():
    prompt = system_prompt()
    assert all(f"- {label}:" in prompt for label in LABELS)


@pytest.mark.parametrize(
    "text, expected",
    [
        ('{"label": "Sport"}', "Sport"),
        ('{"label": " wirtschaft "}', "Wirtschaft"),
        ('{"label": "Wetter"}', None),
        ('{"label": 3}', None),
        ('{"section": "Sport"}', None),
        ("Sport", None),
        ('["Sport"]', None),
    ],
)
def test_parse_label(text, expected):
    assert parse_label(text) == expected


def test_classify_sends_truncated_article():
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return reply('{"label": "Web"}')

    classifier = make(handler, model="m", max_chars=10)
    assert classifier.classify("0123456789ABCDEF") == "Web"
    assert seen["model"] == "m"
    assert seen["format"] == "json"
    assert seen["messages"][1]["content"] == "0123456789"
    assert seen["options"]["temperature"] == 0


def test_classify_returns_none_for_an_invalid_label():
    assert make(lambda request: reply('{"label": "Wetter"}')).classify("x") is None


def test_classify_retries_after_a_server_error():
    answers = [httpx.Response(500, text="busy"), reply('{"label": "Sport"}')]
    classifier = make(lambda request: answers.pop(0))
    assert classifier.classify("x") == "Sport"
    assert answers == []


def test_classify_gives_up_after_the_retries():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(500, text="busy")

    with pytest.raises(LLMError, match="HTTP 500"):
        make(handler, retries=1).classify("x")
    assert len(calls) == 2


def test_classify_reports_a_connection_error():
    def handler(request):
        raise httpx.ConnectError("refused")

    with pytest.raises(LLMError, match="refused"):
        make(handler, retries=0).classify("x")
