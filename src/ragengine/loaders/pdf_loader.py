"""PDF loader with two selectable engines.

The document-loader notebook specifically contrasts two PDF engines:

- `PyPDFLoader` — one Document per page, minimal metadata.
- `PyMuPDFLoader` — "the fastest of the PDF parsing options... provides
  detailed metadata about the PDF and its pages."

Both are reproduced here as engines of a single `PdfLoader`, selected with
`engine="pypdf"` (default) or `engine="pymupdf"`, so the choice becomes a
config value instead of two separate copy-pasted notebook sections.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from ragengine.documents import Document
from ragengine.loaders.base import Loader

Engine = Literal["pypdf", "pymupdf"]


class PdfLoader(Loader):
    name = "pdf"

    def __init__(self, engine: Engine = "pypdf"):
        if engine not in ("pypdf", "pymupdf"):
            raise ValueError(f"Unknown PDF engine: {engine!r}")
        self.engine = engine

    def load(self, source: str) -> list[Document]:
        if self.engine == "pypdf":
            return self._load_pypdf(source)
        return self._load_pymupdf(source)

    @staticmethod
    def _load_pypdf(source: str) -> list[Document]:
        import pypdf

        reader = pypdf.PdfReader(source)
        docs = []
        for i, page in enumerate(reader.pages):
            docs.append(
                Document(
                    page_content=page.extract_text() or "",
                    metadata={"source": str(source), "page": i},
                )
            )
        return docs

    @staticmethod
    def _load_pymupdf(source: str) -> list[Document]:
        import fitz  # PyMuPDF

        docs = []
        with fitz.open(source) as pdf:
            for i, page in enumerate(pdf):
                docs.append(
                    Document(
                        page_content=page.get_text(),
                        metadata={
                            "source": str(source),
                            "page": i,
                            "total_pages": pdf.page_count,
                            "format": pdf.metadata.get("format"),
                            "title": pdf.metadata.get("title"),
                            "author": pdf.metadata.get("author"),
                        },
                    )
                )
        return docs
