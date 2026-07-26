"""Multi-query retriever.

`LangChain_retriever-v1`'s `MultiQueryRetriever`: ask an LLM to rephrase
the user's question several different ways, run retrieval for each
rephrasing plus the original, then union and dedupe the results. This
compensates for a single query wording missing relevant chunks that use
different vocabulary for the same idea.
"""

from __future__ import annotations

from ragengine.documents import Document
from ragengine.llm.base import LLMProvider
from ragengine.retrievers.base import Retriever

_DEFAULT_SYSTEM = (
    "You rewrite a single search query into alternative phrasings to improve "
    "document retrieval recall. Output ONLY the rephrasings, one per line, "
    "no numbering, no bullets, no commentary."
)


class MultiQueryRetriever(Retriever):
    name = "multi_query"

    def __init__(self, base_retriever: Retriever, llm: LLMProvider, num_queries: int = 3):
        self.base_retriever = base_retriever
        self.llm = llm
        self.num_queries = num_queries

    def generate_variants(self, query: str) -> list[str]:
        prompt = (
            f"Generate {self.num_queries} alternative phrasings of this search query, "
            f"one per line:\n\n{query}"
        )
        raw = self.llm.generate(prompt, system=_DEFAULT_SYSTEM)
        variants = [line.strip(" \t-*\u2022") for line in raw.splitlines() if line.strip()]
        return variants[: self.num_queries] or [query]

    def retrieve(self, query: str, k: int = 4) -> list[tuple[Document, float]]:
        all_queries = [query] + self.generate_variants(query)

        best_by_key: dict[str, tuple[Document, float]] = {}
        for q in all_queries:
            for doc, score in self.base_retriever.retrieve(q, k=k):
                key = doc.id or doc.page_content
                if key not in best_by_key or score > best_by_key[key][1]:
                    best_by_key[key] = (doc, score)

        ranked = sorted(best_by_key.values(), key=lambda pair: pair[1], reverse=True)
        return ranked[:k]
