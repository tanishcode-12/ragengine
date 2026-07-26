"""HTML header-aware splitter.

Wraps LangChain's `HTMLHeaderTextSplitter`, the HTML counterpart to the
markdown header splitter: splits on `<h1>`/`<h2>`/`<h3>` boundaries and
keeps each heading's text as chunk metadata.
"""

from __future__ import annotations

from langchain_text_splitters import HTMLHeaderTextSplitter as _LCHtmlHeader
from langchain_text_splitters import RecursiveCharacterTextSplitter as _LCRecursive

from ragengine.documents import Document
from ragengine.splitters.base import Splitter

_DEFAULT_HEADERS = [("h1", "Header 1"), ("h2", "Header 2"), ("h3", "Header 3")]


class HtmlHeaderSplitter(Splitter):
    name = "html_header"

    def __init__(
        self,
        headers_to_split_on: list[tuple[str, str]] | None = None,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ):
        self._header_impl = _LCHtmlHeader(headers_to_split_on=headers_to_split_on or _DEFAULT_HEADERS)
        self._size_impl = _LCRecursive(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def split(self, documents: list[Document]) -> list[Document]:
        out: list[Document] = []
        for doc in documents:
            header_sections = self._header_impl.split_text(doc.page_content)
            for section in header_sections:
                if not section.page_content.strip():
                    continue
                sub_chunks = self._size_impl.split_text(section.page_content)
                for i, chunk in enumerate(sub_chunks):
                    out.append(
                        Document(
                            page_content=chunk,
                            metadata={
                                **doc.metadata,
                                **section.metadata,
                                "chunk": i,
                                "splitter": self.name,
                            },
                        )
                    )
        return out
