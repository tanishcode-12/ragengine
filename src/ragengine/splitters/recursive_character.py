"""Recursive character splitter — the general-purpose default.

Wraps LangChain's `RecursiveCharacterTextSplitter`: tries a list of
separators from "most semantic" (`\\n\\n`, paragraph) to "least semantic"
(single character), recursing until every chunk fits `chunk_size`. This is
the splitter `LangChain_text-splitter-v1` recommends as the general
default, and it's this project's `default_splitter` for the same reason.
"""

from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter as _LCRecursive

from ragengine.documents import Document
from ragengine.splitters.base import Splitter


class RecursiveCharacterSplitter(Splitter):
    name = "recursive_character"

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self._impl = _LCRecursive(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

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
