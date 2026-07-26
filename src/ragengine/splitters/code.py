"""Language-aware code splitter.

Wraps `RecursiveCharacterTextSplitter.from_language(...)`, the technique
`LangChain_text-splitter-v1` uses to split source code along
language-specific boundaries (e.g. never inside a Python function body if
avoidable) instead of blindly on paragraph breaks.
"""

from __future__ import annotations

from langchain_text_splitters import Language
from langchain_text_splitters import RecursiveCharacterTextSplitter as _LCRecursive

from ragengine.documents import Document
from ragengine.splitters.base import Splitter

#: convenience string aliases -> LangChain's Language enum
_LANGUAGE_ALIASES = {
    "python": Language.PYTHON,
    "py": Language.PYTHON,
    "javascript": Language.JS,
    "js": Language.JS,
    "typescript": Language.TS,
    "ts": Language.TS,
    "java": Language.JAVA,
    "go": Language.GO,
    "rust": Language.RUST,
    "markdown": Language.MARKDOWN,
    "html": Language.HTML,
    "cpp": Language.CPP,
    "csharp": Language.CSHARP,
    "php": Language.PHP,
    "ruby": Language.RUBY,
}


class CodeSplitter(Splitter):
    name = "code"

    def __init__(self, language: str = "python", chunk_size: int = 500, chunk_overlap: int = 0):
        try:
            lang = _LANGUAGE_ALIASES[language.lower()]
        except KeyError as exc:
            raise ValueError(
                f"Unknown language {language!r}. Known: {sorted(_LANGUAGE_ALIASES)}"
            ) from exc
        self.language = language
        self._impl = _LCRecursive.from_language(
            language=lang, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

    def split(self, documents: list[Document]) -> list[Document]:
        out: list[Document] = []
        for doc in documents:
            chunks = self._impl.split_text(doc.page_content)
            for i, chunk in enumerate(chunks):
                out.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            **doc.metadata,
                            "chunk": i,
                            "splitter": self.name,
                            "language": self.language,
                        },
                    )
                )
        return out
