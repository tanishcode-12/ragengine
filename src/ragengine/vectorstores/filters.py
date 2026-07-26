"""A tiny, backend-agnostic metadata filter matcher.

Chroma has its own native `where=` filter engine, so `ChromaVectorStore`
never needs this. FAISS has no metadata filtering at all, so
`FaissVectorStore` uses this to post-filter candidates in Python.

Deliberately supports exactly the same filter *shape* Chroma's `where=`
accepts (`{"field": value}`, `{"field": {"$gt": v}}`, `{"$and": [...]}`,
`{"$or": [...]}`) so that a `Retriever` (e.g. `SelfQueryRetriever`) can
build one filter dict and have it behave identically no matter which
vector store backend is configured underneath it.
"""

from __future__ import annotations

from typing import Any

_OPS = {
    "$eq": lambda a, b: a == b,
    "$ne": lambda a, b: a != b,
    "$gt": lambda a, b: a is not None and a > b,
    "$gte": lambda a, b: a is not None and a >= b,
    "$lt": lambda a, b: a is not None and a < b,
    "$lte": lambda a, b: a is not None and a <= b,
    "$in": lambda a, b: a in b,
    "$nin": lambda a, b: a not in b,
}


def matches(metadata: dict[str, Any], filter: dict[str, Any] | None) -> bool:
    if not filter:
        return True
    for key, condition in filter.items():
        if key == "$and":
            if not all(matches(metadata, sub) for sub in condition):
                return False
        elif key == "$or":
            if not any(matches(metadata, sub) for sub in condition):
                return False
        elif isinstance(condition, dict):
            for op, value in condition.items():
                if op not in _OPS:
                    raise ValueError(f"Unsupported filter operator: {op!r}")
                if not _OPS[op](metadata.get(key), value):
                    return False
        else:
            if metadata.get(key) != condition:
                return False
    return True
