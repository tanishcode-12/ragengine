"""Real local LLM via a HuggingFace `transformers` text-generation pipeline.

The alternative to `OllamaLLM` for running a model with nothing but
Python — no separate server process — at the cost of managing a much
heavier dependency (`transformers` + `torch`) and, typically, a slower
first token. Lazy-imported for the same reason as
`SentenceTransformerEmbedding`: importing this module should never require
`transformers`/`torch` to be installed unless this specific backend is
selected.
"""

from __future__ import annotations

from ragengine.llm.base import LLMProvider

_DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


class HuggingFaceLocalLLM(LLMProvider):
    name = "huggingface"

    def __init__(self, model_name: str = _DEFAULT_MODEL, max_new_tokens: int = 512):
        try:
            from transformers import pipeline
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise ImportError(
                "HuggingFaceLocalLLM requires the 'local-models' extra: "
                "pip install 'ragengine[local-models]'"
            ) from exc

        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self._pipe = pipeline("text-generation", model=model_name)

    def generate(self, prompt: str, system: str | None = None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        output = self._pipe(messages, max_new_tokens=self.max_new_tokens)
        generated = output[0]["generated_text"]
        # depending on transformers version this is either the full
        # conversation (list of message dicts) or a raw string continuation
        if isinstance(generated, list):
            return generated[-1]["content"]
        return generated
