"""The Splitter interface every text splitter implements.

Each technique from `LangChain_text-splitter-v1` (character, recursive
character, code-aware, markdown-header, html-header) becomes one
implementation of this interface, selectable by name/config instead of
being five separate notebook sections.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ragengine.documents import Document


class Splitter(ABC):
    name: str = "base"

    @abstractmethod
    def split(self, documents: list[Document]) -> list[Document]:
        """Split a list of Documents into smaller chunk Documents.

        Implementations should carry forward the parent's metadata onto
        every chunk (adding to it, never dropping it), so provenance
        (source file, page number, row, ...) survives chunking.
        """
        raise NotImplementedError

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{self.__class__.__name__}()"
