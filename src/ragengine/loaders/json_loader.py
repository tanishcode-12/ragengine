"""JSON loader.

The source notebook uses LangChain's `JSONLoader`, which selects content
with a "jq_schema" string (e.g. `.messages[].content`) via the `jq`
Python binding — a compiled C library dependency that's a common source of
install failures on machines without a C toolchain.

This loader keeps the exact same *interface* — a jq-style path string —
but evaluates it with a small dependency-free interpreter that supports
the subset of jq actually used for this kind of extraction: chained
`.key` access and `[]` to flatten a list. That covers the common RAG
ingestion case (pull one field out of an array of JSON records) without
requiring `libjq` to be installed on the host.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ragengine.documents import Document
from ragengine.loaders.base import Loader


def _apply_jq_like_schema(data: Any, schema: str) -> list[Any]:
    """Evaluate a small subset of jq syntax: '.a.b[].c' style paths.

    Supports plain dotted keys and a trailing '[]' on any segment to
    flatten into a list. `schema="."` returns the whole document.
    """
    if schema in (".", ""):
        return [data]
    if not schema.startswith("."):
        raise ValueError("jq_schema must start with '.' (e.g. '.messages[].content')")

    tokens = schema[1:].split(".")
    values: list[Any] = [data]
    for token in tokens:
        iterate = token.endswith("[]")
        key = token[:-2] if iterate else token
        next_values: list[Any] = []
        for v in values:
            if key:
                v = v.get(key) if isinstance(v, dict) else None
            if iterate:
                if isinstance(v, list):
                    next_values.extend(v)
                elif v is not None:
                    next_values.append(v)
            else:
                next_values.append(v)
        values = next_values
    return values


class JsonLoader(Loader):
    name = "json"

    def __init__(self, jq_schema: str = ".", content_key: str | None = None):
        self.jq_schema = jq_schema
        self.content_key = content_key

    def load(self, source: str) -> list[Document]:
        path = Path(source)
        data = json.loads(path.read_text(encoding="utf-8"))
        items = _apply_jq_like_schema(data, self.jq_schema)

        docs = []
        for i, item in enumerate(items):
            if isinstance(item, str):
                content = item
                metadata: dict[str, Any] = {}
            elif isinstance(item, dict) and self.content_key:
                content = str(item.get(self.content_key, ""))
                metadata = {k: v for k, v in item.items() if k != self.content_key}
            else:
                content = json.dumps(item, ensure_ascii=False)
                metadata = {}
            metadata.update({"source": str(path), "seq_num": i})
            docs.append(Document(page_content=content, metadata=metadata))
        return docs
