"""Real local LLM via a locally-running Ollama server.

The most common way to run an open-weight chat model entirely for free
on your own machine. Requires Ollama to be installed and running
(`ollama serve`) and the model pulled (`ollama pull llama3.2`) — this
class just talks to its HTTP API, it doesn't manage the server or the
model download itself.
"""

from __future__ import annotations

import requests

from ragengine.llm.base import LLMProvider


class OllamaLLM(LLMProvider):
    name = "ollama"

    def __init__(self, model_name: str = "llama3.2", base_url: str = "http://localhost:11434", timeout: float = 60.0):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def generate(self, prompt: str, system: str | None = None) -> str:
        payload = {"model": self.model_name, "prompt": prompt, "stream": False}
        if system:
            payload["system"] = system
        try:
            resp = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=self.timeout)
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise ConnectionError(
                f"Couldn't reach Ollama at {self.base_url}. Is `ollama serve` running "
                f"and has `ollama pull {self.model_name}` been run? Original error: {exc}"
            ) from exc
        return resp.json()["response"]
