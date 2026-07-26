"""Deterministic stub LLM.

The default backend (`llm_backend="stub"`), so `/query` and `/agent/query`
work immediately with no model, no API key, and no network call — and so
`tests/test_pipeline.py`, `tests/test_agent.py`, and `tests/test_api.py`
can assert on *exact* model output instead of mocking around a real,
nondeterministic one.

Two ways to script it:
- `responses=[...]` — a fixed queue of replies, popped one per call (the
  last one repeats once the queue is exhausted, so a test doesn't have to
  predict exactly how many times the agent loop will call the model).
- nothing configured — falls back to a canned, clearly-labelled reply, so
  the zero-config API path still returns *something* explanatory instead
  of an error.

Every call is recorded in `.calls` (prompt + system) so tests can assert
on exactly what the orchestrator asked the model, not just what it
returned.
"""

from __future__ import annotations

from ragengine.llm.base import LLMProvider


class StubLLM(LLMProvider):
    name = "stub"

    def __init__(self, responses: list[str] | None = None, default_response: str | None = None):
        self._responses = list(responses) if responses else None
        self._default_response = default_response
        self.calls: list[tuple[str, str | None]] = []

    def generate(self, prompt: str, system: str | None = None) -> str:
        self.calls.append((prompt, system))
        if self._responses:
            return self._responses.pop(0) if len(self._responses) > 1 else self._responses[0]
        if self._default_response is not None:
            return self._default_response
        return (
            "[stub-llm] No real LLM is configured (RAG_LLM_BACKEND=stub). "
            f"This deterministic placeholder stands in for a {len(prompt)}-character prompt "
            "so the pipeline can be exercised end-to-end without a model."
        )
