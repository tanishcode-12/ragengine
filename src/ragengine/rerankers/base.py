"""The Reranker interface every re-ranking strategy implements.

A reranker sits *after* a first-pass retriever, not instead of one: the
retriever's job is recall (cheaply narrow millions of chunks down to a
candidate pool using vector similarity), a reranker's job is precision
(look at that small pool much more closely and reorder it). Splitting the
work this way is what makes cross-encoder-quality scoring affordable —
running a cross-encoder over an entire corpus per query would be far too
slow, but running it over `fetch_k` candidates is not.

Query and candidates are always handled as full text, never as vectors,
which is exactly what makes rerankers strictly more expensive than
embedding-based retrieval per document, and is also why a reranker only
runs on the candidate pool `RerankingRetriever` fetches, never the whole
corpus.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ragengine.documents import Document


class Reranker(ABC):
    name: str = "base"

    @abstractmethod
    def score(self, query: str, documents: list[Document]) -> list[float]:
        """Return one relevance score per document, same order as input.

        Scores only need to be internally comparable (higher = more
        relevant) — they are not required to be on the same scale as the
        similarity scores the first-pass retriever produced.
        """
        raise NotImplementedError

    def rerank(
        self, query: str, candidates: list[tuple[Document, float]], k: int
    ) -> list[tuple[Document, float]]:
        """Re-score `candidates` and return the top k, highest score first."""
        if not candidates:
            return []
        docs = [doc for doc, _ in candidates]
        scores = self.score(query, docs)
        reranked = sorted(zip(docs, scores), key=lambda pair: pair[1], reverse=True)
        return reranked[:k]

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{self.__class__.__name__}()"
