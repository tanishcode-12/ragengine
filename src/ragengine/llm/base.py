"""The LLMProvider interface every generation backend implements.

`Full_document_retrieve_limitation-v1` calls watsonx-hosted LLMs directly
(`llama-4-maverick`, `mistral-small`) through `WatsonxLLM`/`ModelInference`
to demonstrate the "stuff the whole document in the prompt" failure mode.
This interface is the free/local/pluggable stand-in for that model call —
used for both final-answer generation and the agent's own reasoning steps
(deciding whether/how to retrieve).
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class LLMProvider(ABC):
    name: str = "base"

    @abstractmethod
    def generate(self, prompt: str, system: str | None = None) -> str:
        """Return the model's completion for `prompt`."""
        raise NotImplementedError

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{self.__class__.__name__}()"
