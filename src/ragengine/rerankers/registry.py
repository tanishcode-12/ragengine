"""Reranker backend registry."""

from __future__ import annotations

from ragengine.rerankers.base import Reranker
from ragengine.rerankers.cross_encoder import CrossEncoderReranker
from ragengine.rerankers.lexical_overlap import LexicalOverlapReranker

_REGISTRY: dict[str, type[Reranker]] = {
    "lexical_overlap": LexicalOverlapReranker,
    "cross_encoder": CrossEncoderReranker,
}


def available_rerankers() -> list[str]:
    return sorted(_REGISTRY.keys())


def get_reranker(name: str, **kwargs) -> Reranker:
    try:
        cls = _REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"Unknown reranker {name!r}. Available: {available_rerankers()}") from exc
    return cls(**kwargs)
