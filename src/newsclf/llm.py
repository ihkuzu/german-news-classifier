from __future__ import annotations

import json
import os

import httpx

from .data import LABELS

DESCRIPTIONS = {
    "Etat": "media industry, journalism, television and press",
    "Inland": "Austrian domestic politics",
    "International": "foreign politics and world news",
    "Kultur": "arts, film, music, literature, theatre",
    "Panorama": "society, crime, accidents, weather, everyday life",
    "Sport": "sports",
    "Web": "internet, technology, gadgets, games",
    "Wirtschaft": "economy, business, finance",
    "Wissenschaft": "science and research",
}

PROMPT = """You classify German news articles from an Austrian newspaper.
Choose exactly one of these sections:
{sections}

Reply with a JSON object like {{"label": "Sport"}} and nothing else."""


class LLMError(Exception):
    pass


def system_prompt() -> str:
    sections = "\n".join(f"- {label}: {DESCRIPTIONS[label]}" for label in LABELS)
    return PROMPT.format(sections=sections)


def parse_label(reply: str) -> str | None:
    try:
        value = json.loads(reply).get("label")
    except (json.JSONDecodeError, AttributeError):
        return None
    if not isinstance(value, str):
        return None
    for label in LABELS:
        if label.lower() == value.strip().lower():
            return label
    return None


class OllamaClassifier:
    def __init__(
        self,
        model: str = "llama3.2:3b",
        host: str | None = None,
        max_chars: int = 1500,
        retries: int = 2,
        timeout: float = 300.0,
        client: httpx.Client | None = None,
    ):
        self.model = model
        self.host = (host or os.getenv("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")
        self.max_chars = max_chars
        self.retries = retries
        self.client = client or httpx.Client(timeout=timeout)

    def classify(self, text: str) -> str | None:
        payload = {
            "model": self.model,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0, "num_predict": 32},
            "messages": [
                {"role": "system", "content": system_prompt()},
                # the start of an article is enough to tell the section
                {"role": "user", "content": text[: self.max_chars]},
            ],
        }
        error = "no attempt made"
        for _ in range(self.retries + 1):
            try:
                response = self.client.post(f"{self.host}/api/chat", json=payload)
            except httpx.HTTPError as exc:
                error = str(exc)
                continue
            if response.status_code != 200:
                error = f"HTTP {response.status_code}: {response.text[:200]}"
                continue
            return parse_label(response.json().get("message", {}).get("content", ""))
        raise LLMError(f"Ollama request failed: {error}")
