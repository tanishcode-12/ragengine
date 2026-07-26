"""Shared data model used by every stage of the pipeline.

Every loader produces ``Document`` objects, every splitter consumes and
produces them, every vector store stores them, and every retriever returns
them. Keeping one small, dependency-free type here (rather than importing
LangChain's ``Document`` everywhere) means the rest of the codebase never
has to care which loader/splitter library implementation is behind an
interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Document:
    """A single piece of content plus metadata about where it came from.

    Mirrors the ``page_content`` / ``metadata`` shape used throughout the
    LangChain ecosystem (``TextLoader``, ``CSVLoader``, ``Chroma``, etc.)
    so the concepts transfer directly, without a hard dependency on
    LangChain's own ``Document`` class.
    """

    page_content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str | None = None

    def with_id(self, doc_id: str) -> "Document":
        self.id = doc_id
        return self
