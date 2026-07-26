"""Wraps any Retriever as a named, described "tool" the agent can call.

This is the seam the project brief calls out specifically: the same
retrieval pipeline built for the direct `/query` endpoint (any of the five
retriever strategies) is reused here unchanged, just handed to the agent
as a callable instead of being queried directly — "agentic RAG" as an
extra layer over the RAG engine, not a separate implementation of it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ragengine.documents import Document
from ragengine.retrievers.base import Retriever


@dataclass
class RetrieverTool:
    retriever: Retriever
    name: str = "search_documents"
    description: str = "Search the indexed document collection for relevant passages."
    k: int = 4
    last_results: list[tuple[Document, float]] = field(default_factory=list, repr=False)

    def run(self, query: str) -> list[tuple[Document, float]]:
        self.last_results = self.retriever.retrieve(query, k=self.k)
        return self.last_results

    @staticmethod
    def format_context(results: list[tuple[Document, float]]) -> str:
        if not results:
            return "(no results found)"
        parts = []
        for i, (doc, score) in enumerate(results, start=1):
            source = doc.metadata.get("source", "unknown")
            parts.append(f"[{i}] (score={score:.2f}, source={source})\n{doc.page_content}")
        return "\n\n".join(parts)
