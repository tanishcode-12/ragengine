"""Loader registry.

The one place that knows about every loader implementation. The rest of
the codebase (the pipeline, the API) asks for a loader by name or lets
`for_path` guess from a file extension — nobody else needs an `if/elif`
chain of file extensions or a direct import of a specific loader class.
"""

from __future__ import annotations

from pathlib import Path

from ragengine.loaders.base import Loader
from ragengine.loaders.csv_loader import CsvLoader
from ragengine.loaders.docx_loader import DocxLoader
from ragengine.loaders.json_loader import JsonLoader
from ragengine.loaders.markdown_loader import MarkdownLoader
from ragengine.loaders.pdf_loader import PdfLoader
from ragengine.loaders.text_loader import TextLoader
from ragengine.loaders.web_loader import WebLoader

_REGISTRY: dict[str, type[Loader]] = {
    "text": TextLoader,
    "pdf": PdfLoader,
    "markdown": MarkdownLoader,
    "json": JsonLoader,
    "csv": CsvLoader,
    "web": WebLoader,
    "docx": DocxLoader,
}

_EXTENSION_MAP: dict[str, str] = {
    ".txt": "text",
    ".md": "markdown",
    ".markdown": "markdown",
    ".pdf": "pdf",
    ".json": "json",
    ".csv": "csv",
    ".docx": "docx",
    ".html": "web",
    ".htm": "web",
}


def available_loaders() -> list[str]:
    return sorted(_REGISTRY.keys())


def get_loader(name: str, **kwargs) -> Loader:
    try:
        cls = _REGISTRY[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown loader {name!r}. Available: {available_loaders()}"
        ) from exc
    return cls(**kwargs)


def loader_name_for_path(path: str) -> str:
    ext = Path(path).suffix.lower()
    try:
        return _EXTENSION_MAP[ext]
    except KeyError as exc:
        raise ValueError(
            f"Can't guess a loader for extension {ext!r}. "
            f"Pass loader_name explicitly. Known extensions: {sorted(_EXTENSION_MAP)}"
        ) from exc


def get_loader_for_path(path: str, **kwargs) -> Loader:
    """Guess the right loader from the file extension and instantiate it."""
    return get_loader(loader_name_for_path(path), **kwargs)
