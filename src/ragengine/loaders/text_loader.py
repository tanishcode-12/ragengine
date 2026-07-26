"""Plain text loader.

Equivalent to LangChain's `TextLoader`, used across every source notebook
(e.g. `Embed_documents_with_watsonx's_embedding`, `LangChain_vector_store`)
as the simplest possible loader: read a file, wrap it in one Document.
"""

from __future__ import annotations

from pathlib import Path

from ragengine.documents import Document
from ragengine.loaders.base import Loader


class TextLoader(Loader):
    name = "text"

    def __init__(self, encoding: str = "utf-8"):
        self.encoding = encoding

    def load(self, source: str) -> list[Document]:
        path = Path(source)
        text = path.read_text(encoding=self.encoding)
        return [Document(page_content=text, metadata={"source": str(path)})]
