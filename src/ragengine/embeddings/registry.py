"""Embedding backend registry."""

from __future__ import annotations

from ragengine.embeddings.base import EmbeddingModel
from ragengine.embeddings.deterministic_fake import DeterministicHashEmbedding
from ragengine.embeddings.sentence_transformer import SentenceTransformerEmbedding

_REGISTRY: dict[str, type[EmbeddingModel]] = {
    "fake": DeterministicHashEmbedding,
    "sentence_transformers": SentenceTransformerEmbedding,
}


def available_embedding_backends() -> list[str]:
    return sorted(_REGISTRY.keys())


def get_embedding_model(name: str, **kwargs) -> EmbeddingModel:
    try:
        cls = _REGISTRY[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown embedding backend {name!r}. Available: {available_embedding_backends()}"
        ) from exc
    return cls(**kwargs)
