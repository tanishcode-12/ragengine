"""Retriever registry.

Unlike the other registries in this project, this one only maps a name to
a *class*, not a ready-to-call factory function — the five retrievers
genuinely take different constructor arguments (a `VectorStoreRetriever`
needs a vector store + embedding model; a `MultiQueryRetriever` wraps
another retriever and needs an LLM; `ParentDocumentRetriever` needs two
splitters). Pretending otherwise behind one `get_retriever(name, **kwargs)`
signature would hide that difference rather than represent it honestly.
`RagPipeline` (pipeline.py) is where the actual wiring happens.
"""

from __future__ import annotations

from ragengine.retrievers.base import Retriever
from ragengine.retrievers.multi_query import MultiQueryRetriever
from ragengine.retrievers.parent_document import ParentDocumentRetriever
from ragengine.retrievers.self_query import SelfQueryRetriever
from ragengine.retrievers.similarity import VectorStoreRetriever

_REGISTRY: dict[str, type[Retriever]] = {
    "similarity": VectorStoreRetriever,
    "mmr": VectorStoreRetriever,
    "similarity_score_threshold": VectorStoreRetriever,
    "multi_query": MultiQueryRetriever,
    "parent_document": ParentDocumentRetriever,
    "self_query": SelfQueryRetriever,
}


def available_retrievers() -> list[str]:
    return sorted(_REGISTRY.keys())


def get_retriever_class(name: str) -> type[Retriever]:
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"Unknown retriever {name!r}. Available: {available_retrievers()}") from exc
