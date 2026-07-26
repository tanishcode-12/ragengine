"""FAISS vector store backend.

`LangChain_vector_store-v1` covers FAISS as the alternative to Chroma.
FAISS itself only ever deals in raw vectors — no document text, no
metadata, no delete-by-id on a plain index — so this class keeps a small
Python-side docstore (`{int_id: Document}`) alongside a
`faiss.IndexIDMap(IndexFlatIP)`, the same pattern LangChain's own FAISS
wrapper uses internally.

All embeddings are L2-normalized before entering the index, so that inner
product (`IndexFlatIP`) is equivalent to cosine similarity — this keeps
FAISS's notion of "similarity score" directly comparable to
`ChromaVectorStore`'s (which is configured with `hnsw:space=cosine`),
regardless of whether the embedding model itself normalizes its output.

FAISS has no metadata filtering, so `similarity_search_by_vector` with a
`filter` over-fetches candidates and post-filters them in Python using
`vectorstores.filters.matches` — the same filter dict shape Chroma's
native `where=` accepts, so callers (e.g. `SelfQueryRetriever`) don't need
to know which backend they're talking to.
"""

from __future__ import annotations

import itertools
import uuid

import numpy as np

from ragengine.documents import Document
from ragengine.vectorstores.base import VectorStore
from ragengine.vectorstores.filters import matches as filter_matches


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


class FaissVectorStore(VectorStore):
    name = "faiss"

    def __init__(self, dimensions: int):
        import faiss

        self._faiss = faiss
        self.dimensions = dimensions
        self._index = faiss.IndexIDMap(faiss.IndexFlatIP(dimensions))
        self._docstore: dict[int, Document] = {}
        self._str_to_int: dict[str, int] = {}
        self._counter = itertools.count(1)

    def add(self, documents: list[Document], embeddings: list[list[float]]) -> list[str]:
        if len(documents) != len(embeddings):
            raise ValueError("documents and embeddings must be the same length")
        str_ids = [doc.id or str(uuid.uuid4()) for doc in documents]
        int_ids = [next(self._counter) for _ in documents]

        vecs = _normalize(np.array(embeddings, dtype="float32"))
        self._index.add_with_ids(vecs, np.array(int_ids, dtype="int64"))

        for doc, str_id, int_id in zip(documents, str_ids, int_ids):
            doc.id = str_id
            self._docstore[int_id] = doc
            self._str_to_int[str_id] = int_id
        return str_ids

    def update(
        self,
        ids: list[str],
        documents: list[Document] | None = None,
        embeddings: list[list[float]] | None = None,
    ) -> None:
        if documents is not None and embeddings is None:
            raise ValueError(
                "update() requires embeddings when documents is provided — "
                "re-embed the new text before calling update() (see VectorStore.update docstring)"
            )
        for i, str_id in enumerate(ids):
            if str_id not in self._str_to_int:
                raise KeyError(f"id not found in FAISS store: {str_id!r}")
            int_id = self._str_to_int[str_id]

            if embeddings is not None:
                # FAISS can't update a vector in place - remove then re-add.
                sel = self._faiss.IDSelectorBatch(np.array([int_id], dtype="int64"))
                self._index.remove_ids(sel)
                vec = _normalize(np.array([embeddings[i]], dtype="float32"))
                self._index.add_with_ids(vec, np.array([int_id], dtype="int64"))

            if documents is not None:
                documents[i].id = str_id
                self._docstore[int_id] = documents[i]

    def delete(self, ids: list[str]) -> None:
        int_ids = [self._str_to_int.pop(i) for i in ids if i in self._str_to_int]
        if not int_ids:
            return
        sel = self._faiss.IDSelectorBatch(np.array(int_ids, dtype="int64"))
        self._index.remove_ids(sel)
        for int_id in int_ids:
            self._docstore.pop(int_id, None)

    def similarity_search_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        filter: dict | None = None,
    ) -> list[tuple[Document, float]]:
        if self.count() == 0:
            return []
        # Over-fetch when a metadata filter is active, since we can only
        # apply it after the ANN search has already picked candidates.
        fetch_k = min(self.count(), k * 20 if filter else k)
        query = _normalize(np.array([embedding], dtype="float32"))
        scores, int_ids = self._index.search(query, fetch_k)

        out: list[tuple[Document, float]] = []
        for score, int_id in zip(scores[0], int_ids[0]):
            if int_id == -1:
                continue
            doc = self._docstore[int_id]
            if filter and not filter_matches(doc.metadata, filter):
                continue
            out.append((doc, max(0.0, float(score))))
            if len(out) >= k:
                break
        return out

    def count(self) -> int:
        return self._index.ntotal

    def get_by_ids(self, ids: list[str]) -> list[Document]:
        out = []
        for str_id in ids:
            int_id = self._str_to_int.get(str_id)
            if int_id is not None and int_id in self._docstore:
                out.append(self._docstore[int_id])
        return out
