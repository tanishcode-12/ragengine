"""Agentic RAG orchestrator.

This is the "Phase 0 option (c)" piece: instead of a fixed
loader->splitter->embed->store->retrieve pipeline that always retrieves
exactly once, the agent decides — per question — whether it needs to
search at all, what to search for, and (bounded by
`max_iterations`) whether the first search actually returned enough to
answer from or it should reformulate and search again. That reformulation
loop is the same problem `Full_document_retrieve_limitation-v1` raises
(a single fixed retrieval can miss what's needed) approached from the
other direction: instead of avoiding retrieval, let the agent retry it.

The protocol between the orchestrator and the LLM is deliberately simple
— one line, one of two prefixes — so it's parseable without needing a
model that supports native function-calling:

    SEARCH: <query to run>
    ANSWER: <final answer>

This is intentionally text-based rather than JSON so it works with the
weakest local models too (see `llm.huggingface_llm`); `SelfQueryRetriever`
already demonstrates the JSON-structured-output style for a case that
specifically needs it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ragengine.agent.tools import RetrieverTool
from ragengine.llm.base import LLMProvider

_INITIAL_SYSTEM = (
    "You are a research assistant with access to one tool, {tool_name}: {tool_description}\n"
    "Decide whether answering the user's question needs the tool. Reply with EXACTLY one line, "
    "in one of these two forms and nothing else:\n"
    "SEARCH: <query to run>\n"
    "ANSWER: <your answer, only if you're confident you don't need to search>"
)

_FOLLOWUP_SYSTEM = (
    "You previously ran a search and received the context below. Decide whether it's enough to "
    "answer the question, or whether you need to search again with a different query. Reply with "
    "EXACTLY one line, in one of these two forms and nothing else:\n"
    "ANSWER: <answer using the context>\n"
    "SEARCH: <a differently-worded query>"
)

_FORCED_SYSTEM = (
    "Answer the user's question as well as you can using the retrieved context below. If the "
    "context doesn't fully answer it, say what's missing rather than guessing."
)


@dataclass
class AgentResult:
    answer: str
    trace: list[dict[str, Any]] = field(default_factory=list)


class AgentOrchestrator:
    def __init__(self, tool: RetrieverTool, llm: LLMProvider, max_iterations: int = 3):
        if max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")
        self.tool = tool
        self.llm = llm
        self.max_iterations = max_iterations

    def run(self, question: str) -> AgentResult:
        trace: list[dict[str, Any]] = []
        context: str | None = None

        for _ in range(self.max_iterations):
            if context is None:
                system = _INITIAL_SYSTEM.format(
                    tool_name=self.tool.name, tool_description=self.tool.description
                )
                decision = self.llm.generate(f"Question: {question}", system=system)
            else:
                decision = self.llm.generate(
                    f"Question: {question}\n\nRetrieved context:\n{context}",
                    system=_FOLLOWUP_SYSTEM,
                )

            action, _, payload = decision.partition(":")
            action = action.strip().upper()
            payload = payload.strip()

            if action == "SEARCH":
                query = payload or question
                results = self.tool.run(query)
                context = self.tool.format_context(results)
                trace.append({"action": "search", "query": query, "num_results": len(results)})
                continue

            if action == "ANSWER":
                trace.append({"action": "answer", "answer": payload})
                return AgentResult(answer=payload, trace=trace)

            # Model didn't follow the SEARCH:/ANSWER: format - treat the
            # whole reply as a best-effort answer rather than erroring out.
            trace.append({"action": "answer", "answer": decision.strip(), "unparsed": True})
            return AgentResult(answer=decision.strip(), trace=trace)

        # Iteration budget exhausted while still searching - force a final
        # answer from whatever context was gathered instead of looping forever.
        answer = self.llm.generate(
            f"Question: {question}\n\nRetrieved context:\n{context or '(none found)'}",
            system=_FORCED_SYSTEM,
        )
        trace.append({"action": "answer", "answer": answer, "forced": True})
        return AgentResult(answer=answer, trace=trace)
