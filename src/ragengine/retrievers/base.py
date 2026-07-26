"""The Retriever interface every retrieval strategy implements.

`LangChain_retriever-v1` is the richest of the source notebooks: it walks
through similarity/MMR/threshold search, `MultiQueryRetriever`,
`SelfQueryRetriever`, and `ParentDocumentRetriever` as separate techniques.
Here each becomes one implementation of this same interface, so the
pipeline (and the API) can swap retrieval strategy via config/request
parameter instead of rewriting the calling code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ragengine.documents import Document


class Retriever(ABC):
    name: str = "base"

    #: Most retrievers just query a vector store that the pipeline already
    #: populated via the standard loader -> splitter -> embedder -> store
    #: flow. ParentDocumentRetriever is the exception — it needs to index
    #: documents at two granularities (small chunks for search, larger
    #: parents for the actual returned context), so it overrides `ingest`
    #: and sets this flag so `RagPipeline.ingest_file` knows to delegate to
    #: it instead of doing the generic split/embed/add itself.
    supports_custom_ingestion: bool = False

    @abstractmethod
    def retrieve(self, query: str, k: int = 4) -> list[tuple[Document, float]]:
        """Return up to k (Document, similarity_score) pairs for `query`,
        highest score first."""
        raise NotImplementedError

    def ingest(self, documents: list[Document]) -> None:  # pragma: no cover - default no-op
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support custom ingestion "
            "(supports_custom_ingestion is False)"
        )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{self.__class__.__name__}()"
