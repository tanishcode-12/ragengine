"""The VectorStore interface every vector database backend implements.

Deliberate architecture note (a difference from the source notebooks):
LangChain's `Chroma`/`FAISS` wrapper classes bundle an embedding function
*inside* the vector store object, so `.add_documents(docs)` and
`.similarity_search("query text")` silently embed things for you. That's
convenient for a notebook, but it welds two independent concerns
together. Here, a VectorStore only ever deals in vectors it's handed —
it does not know an `EmbeddingModel` exists. Something upstream (the
`RagPipeline`, or a retriever) is responsible for calling the embedding
model and passing the resulting vectors in. That's what makes it possible
to unit test CRUD behaviour (this file's job) completely separately from
embedding quality (the embeddings/ package's job) — see
`tests/test_vectorstores.py`, which never loads an embedding model.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ragengine.documents import Document


class VectorStore(ABC):
    name: str = "base"

    @abstractmethod
    def add(self, documents: list[Document], embeddings: list[list[float]]) -> list[str]:
        """Store documents with their precomputed embeddings. Returns the
        ids assigned to each document (existing `document.id` is reused if
        set, otherwise a new id is generated)."""
        raise NotImplementedError

    @abstractmethod
    def update(
        self,
        ids: list[str],
        documents: list[Document] | None = None,
        embeddings: list[list[float]] | None = None,
    ) -> None:
        """Update content/embeddings/metadata for existing ids in place.

        Contract: if `documents` is provided, `embeddings` must be provided
        too. Re-embedding changed text is the caller's responsibility (the
        store has no embedding model of its own to fall back on — see the
        module docstring). This also guards against a real bug found while
        building `ChromaVectorStore`: asking Chroma to update `documents`
        without `embeddings` makes it silently invoke its own default
        embedding function instead of leaving the vector alone.
        """
        raise NotImplementedError

    @abstractmethod
    def delete(self, ids: list[str]) -> None:
        raise NotImplementedError

    @abstractmethod
    def similarity_search_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        filter: dict | None = None,
    ) -> list[tuple[Document, float]]:
        """Return the k nearest Documents to `embedding`, each paired with
        a similarity score in [0, 1] (1 = identical), highest first."""
        raise NotImplementedError

    @abstractmethod
    def count(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_by_ids(self, ids: list[str]) -> list[Document]:
        raise NotImplementedError

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{self.__class__.__name__}(count={self.count()})"
