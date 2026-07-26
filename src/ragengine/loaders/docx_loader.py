"""Word document (.docx) loader.

Mirrors LangChain's `Docx2txtLoader`, using the same underlying
`docx2txt` library referenced alongside the other loaders in the
document-loader notebook.
"""

from __future__ import annotations

from pathlib import Path

from ragengine.documents import Document
from ragengine.loaders.base import Loader


class DocxLoader(Loader):
    name = "docx"

    def load(self, source: str) -> list[Document]:
        import docx2txt

        text = docx2txt.process(source)
        return [Document(page_content=text, metadata={"source": str(Path(source))})]
