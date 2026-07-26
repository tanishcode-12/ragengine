"""Character-based splitter — splits on a single fixed separator.

Wraps LangChain's `CharacterTextSplitter`, the first, simplest technique
taught in `LangChain_text-splitter-v1`: split on one separator (default a
blank line) and only fall back to a hard cut if a resulting piece is
still bigger than `chunk_size`.
"""

from __future__ import annotations

from langchain_text_splitters import CharacterTextSplitter as _LCCharacterTextSplitter

from ragengine.documents import Document
from ragengine.splitters.base import Splitter


class CharacterSplitter(Splitter):
    name = "character"

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50, separator: str = "\n\n"):
        self._impl = _LCCharacterTextSplitter(
            separator=separator,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def split(self, documents: list[Document]) -> list[Document]:
        out: list[Document] = []
        for doc in documents:
            chunks = self._impl.split_text(doc.page_content)
            for i, chunk in enumerate(chunks):
                out.append(
                    Document(
                        page_content=chunk,
                        metadata={**doc.metadata, "chunk": i, "splitter": self.name},
                    )
                )
        return out
