"""Markdown header-aware splitter.

Wraps LangChain's `MarkdownHeaderTextSplitter`: instead of cutting
mid-section, it splits along `#`/`##`/`###` boundaries and attaches each
header's text as chunk metadata (so a chunk "remembers" which section it
came from — useful both for citing sources and for `SelfQueryRetriever`
metadata filtering). Sections that are still longer than `chunk_size`
after the header split are further divided with the recursive splitter,
matching the two-stage flow used in `LangChain_text-splitter-v1`.
"""

from __future__ import annotations

from langchain_text_splitters import MarkdownHeaderTextSplitter as _LCMarkdownHeader
from langchain_text_splitters import RecursiveCharacterTextSplitter as _LCRecursive

from ragengine.documents import Document
from ragengine.splitters.base import Splitter

_DEFAULT_HEADERS = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]


class MarkdownHeaderSplitter(Splitter):
    name = "markdown_header"

    def __init__(
        self,
        headers_to_split_on: list[tuple[str, str]] | None = None,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ):
        self._header_impl = _LCMarkdownHeader(
            headers_to_split_on=headers_to_split_on or _DEFAULT_HEADERS
        )
        self._size_impl = _LCRecursive(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def split(self, documents: list[Document]) -> list[Document]:
        out: list[Document] = []
        for doc in documents:
            header_sections = self._header_impl.split_text(doc.page_content)
            for section in header_sections:
                sub_chunks = self._size_impl.split_text(section.page_content)
                for i, chunk in enumerate(sub_chunks):
                    out.append(
                        Document(
                            page_content=chunk,
                            metadata={
                                **doc.metadata,
                                **section.metadata,  # Header 1 / Header 2 / ...
                                "chunk": i,
                                "splitter": self.name,
                            },
                        )
                    )
        return out
