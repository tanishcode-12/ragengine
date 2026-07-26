"""Chroma vector store backend.

Wraps `chromadb` directly (rather than LangChain's `Chroma` wrapper used in
`LangChain_vector_store-v1`) so this class can own the "always pass
embeddings explicitly" discipline documented below — the store never
carries an embedding function of its own (see `vectorstores/base.py`).

Two gotchas discovered while building this that are worth knowing if you
touch this file:

1. If you ever call `collection.add/update/upsert` with `documents=` but
   *without* `embeddings=`, Chroma silently falls back to its own default
   ONNX embedding function and tries to download model weights on first
   use. In a network-restricted environment that download fails outright;
   even when it succeeds, it means two different embedding spaces are
   quietly in play. This class always computes and passes `embeddings=`
   itself, so that default path is never triggered.
2. Chroma rejects `None` metadata values and empty metadata dicts outright
   (`Cannot convert Python object to MetadataValue` / "Expected metadata to
   be a non-empty dict"). `_sanitize_metadata` below strips `None`s,
   stringifies anything that isn't a str/int/float/bool, and falls back to
   `{"doc_id": ...}` if nothing is left.
"""

from __future__ import annotations

import uuid

from ragengine.documents import Document
from ragengine.vectorstores.base import VectorStore


def _sanitize_metadata(doc_id: str, metadata: dict) -> dict:
    clean: dict = {}
    for k, v in metadata.items():
        if v is None:
            continue
        clean[k] = v if isinstance(v, (str, int, float, bool)) else str(v)
    if not clean:
        clean["doc_id"] = doc_id
    return clean


class ChromaVectorStore(VectorStore):
    name = "chroma"

    def __init__(self, collection_name: str = "ragengine", persist_dir: str | None = None):
        import chromadb

        self._client = (
            chromadb.PersistentClient(path=persist_dir) if persist_dir else chromadb.Client()
        )
        # hnsw:space=cosine so distances returned by query() are cosine
        # distance (1 - cosine similarity), matching every embedding model
        # used in this project (all of which are compared by cosine sim).
        self._collection = self._client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )

    def add(self, documents: list[Document], embeddings: list[list[float]]) -> list[str]:
        if len(documents) != len(embeddings):
            raise ValueError("documents and embeddings must be the same length")
        ids = [doc.id or str(uuid.uuid4()) for doc in documents]
        metadatas = [_sanitize_metadata(doc_id, doc.metadata) for doc_id, doc in zip(ids, documents)]
        self._collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=[doc.page_content for doc in documents],
            metadatas=metadatas,
        )
        for doc, doc_id in zip(documents, ids):
            doc.id = doc_id
        return ids

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
        kwargs: dict = {"ids": ids}
        if embeddings is not None:
            kwargs["embeddings"] = embeddings
        if documents is not None:
            kwargs["documents"] = [doc.page_content for doc in documents]
            kwargs["metadatas"] = [
                _sanitize_metadata(doc_id, doc.metadata) for doc_id, doc in zip(ids, documents)
            ]
        self._collection.update(**kwargs)

    def delete(self, ids: list[str]) -> None:
        self._collection.delete(ids=ids)

    def similarity_search_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        filter: dict | None = None,
    ) -> list[tuple[Document, float]]:
        n_results = min(k, max(self.count(), 1))
        result = self._collection.query(
            query_embeddings=[embedding],
            n_results=n_results,
            where=filter,
            include=["documents", "metadatas", "distances"],
        )
        out: list[tuple[Document, float]] = []
        ids = result["ids"][0]
        docs = result["documents"][0]
        metas = result["metadatas"][0]
        dists = result["distances"][0]
        for doc_id, content, meta, dist in zip(ids, docs, metas, dists):
            score = max(0.0, 1.0 - dist)  # cosine distance -> similarity
            out.append((Document(page_content=content, metadata=meta, id=doc_id), score))
        return out

    def count(self) -> int:
        return self._collection.count()

    def get_by_ids(self, ids: list[str]) -> list[Document]:
        if not ids:
            return []
        result = self._collection.get(ids=ids, include=["documents", "metadatas"])
        # Chroma's get() does not preserve the requested id order, so we
        # rebuild a lookup and re-emit in the order the caller asked for.
        by_id = {
            doc_id: Document(page_content=content, metadata=meta, id=doc_id)
            for doc_id, content, meta in zip(result["ids"], result["documents"], result["metadatas"])
        }
        return [by_id[i] for i in ids if i in by_id]
