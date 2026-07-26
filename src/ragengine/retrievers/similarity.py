"""Vector-store-backed retriever: similarity / MMR / score-threshold.

Reproduces the three `search_type` options `LangChain_retriever-v1` shows
off `.as_retriever(search_type=...)`:

- "similarity" — plain top-k nearest neighbours.
- "similarity_score_threshold" — top-k, but only scores >= a cutoff.
- "mmr" — Maximal Marginal Relevance: re-rank a larger candidate pool to
  balance relevance against diversity, so top-k isn't five near-duplicate
  chunks.

MMR needs the *vectors* of the candidate documents, not just their text,
to compute the diversity term. Rather than widen the `VectorStore`
interface to return raw vectors (which not every real vector database
makes easy to bulk-fetch), this re-embeds the small candidate pool
(`fetch_k` documents, not the whole corpus) through the same embedding
model — a deliberate, documented trade-off of a few extra embedding calls
for a vector store interface that stays uniform across backends.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from ragengine.documents import Document
from ragengine.embeddings.base import EmbeddingModel
from ragengine.retrievers.base import Retriever
from ragengine.vectorstores.base import VectorStore

SearchType = Literal["similarity", "mmr", "similarity_score_threshold"]


def _cosine_sim_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-10)
    b_norm = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-10)
    return a_norm @ b_norm.T


def _mmr_select(query_vec: np.ndarray, doc_vecs: np.ndarray, k: int, lambda_mult: float) -> list[int]:
    if len(doc_vecs) == 0:
        return []
    query_sim = _cosine_sim_matrix(query_vec[None, :], doc_vecs)[0]
    doc_sim = _cosine_sim_matrix(doc_vecs, doc_vecs)

    selected: list[int] = []
    remaining = list(range(len(doc_vecs)))
    while remaining and len(selected) < k:
        if not selected:
            best = max(remaining, key=lambda i: query_sim[i])
        else:
            def mmr_score(i: int) -> float:
                diversity = max(doc_sim[i][j] for j in selected)
                return lambda_mult * query_sim[i] - (1 - lambda_mult) * diversity

            best = max(remaining, key=mmr_score)
        selected.append(best)
        remaining.remove(best)
    return selected


class VectorStoreRetriever(Retriever):
    def __init__(
        self,
        vector_store: VectorStore,
        embedding_model: EmbeddingModel,
        search_type: SearchType = "similarity",
        score_threshold: float = 0.5,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
    ):
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.search_type = search_type
        self.score_threshold = score_threshold
        self.fetch_k = fetch_k
        self.lambda_mult = lambda_mult
        self.name = search_type

    def retrieve(self, query: str, k: int = 4) -> list[tuple[Document, float]]:
        query_vec = self.embedding_model.embed_query(query)

        if self.search_type == "similarity":
            return self.vector_store.similarity_search_by_vector(query_vec, k=k)

        if self.search_type == "similarity_score_threshold":
            candidates = self.vector_store.similarity_search_by_vector(
                query_vec, k=max(k, self.fetch_k)
            )
            return [pair for pair in candidates if pair[1] >= self.score_threshold][:k]

        if self.search_type == "mmr":
            candidates = self.vector_store.similarity_search_by_vector(query_vec, k=self.fetch_k)
            if not candidates:
                return []
            cand_vecs = np.array(
                self.embedding_model.embed_documents([doc.page_content for doc, _ in candidates])
            )
            order = _mmr_select(np.array(query_vec), cand_vecs, k=k, lambda_mult=self.lambda_mult)
            return [candidates[i] for i in order]

        raise ValueError(f"Unknown search_type: {self.search_type!r}")
