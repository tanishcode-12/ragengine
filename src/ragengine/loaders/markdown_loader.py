"""Markdown loader.

The source notebook uses `UnstructuredMarkdownLoader`, which pulls in the
heavyweight `unstructured` package (plus NLTK data downloads) just to read
a file and strip formatting. This loader keeps the same outcome — one
Document holding the file's content, ready for either plain-text splitting
or the header-aware `MarkdownHeaderSplitter` downstream — without that
dependency weight. The raw markdown (headers included) is preserved,
because `MarkdownHeaderSplitter` needs the `#`/`##` markers to do its job.
"""

from __future__ import annotations

from pathlib import Path

from ragengine.documents import Document
from ragengine.loaders.base import Loader


class MarkdownLoader(Loader):
    name = "markdown"

    def load(self, source: str) -> list[Document]:
        path = Path(source)
        text = path.read_text(encoding="utf-8")
        return [Document(page_content=text, metadata={"source": str(path), "format": "markdown"})]
