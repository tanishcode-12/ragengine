"""LLM backend registry."""

from __future__ import annotations

from ragengine.llm.base import LLMProvider
from ragengine.llm.huggingface_llm import HuggingFaceLocalLLM
from ragengine.llm.ollama_llm import OllamaLLM
from ragengine.llm.stub_llm import StubLLM

_REGISTRY: dict[str, type[LLMProvider]] = {
    "stub": StubLLM,
    "ollama": OllamaLLM,
    "huggingface": HuggingFaceLocalLLM,
}


def available_llm_backends() -> list[str]:
    return sorted(_REGISTRY.keys())


def get_llm(name: str, **kwargs) -> LLMProvider:
    try:
        cls = _REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"Unknown LLM backend {name!r}. Available: {available_llm_backends()}") from exc
    return cls(**kwargs)
