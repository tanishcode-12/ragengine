"""The EmbeddingModel interface every embedding backend implements.

`Embed_documents_with_watsonx's_embedding-v1` compares a watsonx-hosted
model against a local HuggingFace model, making the same point this
interface encodes: the rest of the system (vector stores, retrievers)
should not care which embedding model produced the vectors, only that it
can turn text into a fixed-size list of floats.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class EmbeddingModel(ABC):
    name: str = "base"
    #: dimensionality of the vectors this model produces
    dimensions: int

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of documents for indexing."""
        raise NotImplementedError

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string for search.

        Kept as a separate method (rather than reusing embed_documents)
        because some real embedding models use different prefixes/
        instructions for queries vs. documents.
        """
        raise NotImplementedError

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{self.__class__.__name__}(dimensions={self.dimensions})"
