"""The Loader interface every document loader implements.

This is the "pluggable module" abstraction called for in the project brief:
each format-specific loader in the source notebooks (`TextLoader`,
`PyPDFLoader`, `PyMuPDFLoader`, `UnstructuredMarkdownLoader`, `JSONLoader`,
`CSVLoader`, `WebBaseLoader`, `Docx2txtLoader`) becomes one implementation
of this same interface, instead of six independent, copy-pasted notebook
sections.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ragengine.documents import Document


class Loader(ABC):
    """Loads a source (file path or URL) into a list of Documents."""

    #: short machine-readable name used in the registry / API requests
    name: str = "base"

    @abstractmethod
    def load(self, source: str) -> list[Document]:
        """Read `source` and return one or more Documents."""
        raise NotImplementedError

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{self.__class__.__name__}()"
