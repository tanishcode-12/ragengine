"""Reranking retriever: a two-stage retrieve-then-rerank wrapper.

Sits between `VectorStoreRetriever` and the final top-k cut, exactly
where the README's roadmap flagged it: fetch a wider candidate pool with
cheap vector similarity, then re-score that pool with a `Reranker` (see
`rerankers/`) that looks at actual query/document text pairs instead of
precomputed vectors. Wraps a `base_retriever` the same way
`MultiQueryRetriever` does, rather than reimplementing candidate fetching
itself, so any retriever (similarity, MMR, multi_query, ...) can have
reranking layered on top of it.
"""

from __future__ import annotations

from ragengine.documents import Document
from ragengine.rerankers.base import Reranker
from ragengine.retrievers.base import Retriever


class RerankingRetriever(Retriever):
    name = "rerank"

    def __init__(self, base_retriever: Retriever, reranker: Reranker, fetch_k: int = 20):
        self.base_retriever = base_retriever
        self.reranker = reranker
        # How many candidates the base retriever fetches before reranking
        # narrows them down to k. Must be >= k for reranking to have any
        # candidates to choose *among*; larger values trade more base-retriever
        # + reranker work for a better chance the true best-k made the pool.
        self.fetch_k = fetch_k

    def retrieve(self, query: str, k: int = 4) -> list[tuple[Document, float]]:
        candidates = self.base_retriever.retrieve(query, k=max(k, self.fetch_k))
        return self.reranker.rerank(query, candidates, k=k)
