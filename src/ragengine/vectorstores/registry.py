"""Vector store backend registry.

Note: `FaissVectorStore` requires a `dimensions` constructor argument
(FAISS must allocate its index for a fixed vector size up front), while
`ChromaVectorStore` does not. `RagPipeline` (see pipeline.py) is the one
place that knows to supply `dimensions=embedding_model.dimensions` when
wiring a FAISS backend — callers of this registry function just pass
whatever kwargs their chosen backend needs.
"""

from __future__ import annotations

from ragengine.vectorstores.base import VectorStore
from ragengine.vectorstores.chroma_store import ChromaVectorStore
from ragengine.vectorstores.faiss_store import FaissVectorStore

_REGISTRY: dict[str, type[VectorStore]] = {
    "chroma": ChromaVectorStore,
    "faiss": FaissVectorStore,
}


def available_vector_stores() -> list[str]:
    return sorted(_REGISTRY.keys())


def get_vector_store(name: str, **kwargs) -> VectorStore:
    try:
        cls = _REGISTRY[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown vector store {name!r}. Available: {available_vector_stores()}"
        ) from exc
    return cls(**kwargs)
