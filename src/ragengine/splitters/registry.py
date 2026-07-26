"""Splitter registry — look up a splitter implementation by name."""

from __future__ import annotations

from ragengine.splitters.base import Splitter
from ragengine.splitters.character import CharacterSplitter
from ragengine.splitters.code import CodeSplitter
from ragengine.splitters.html_header import HtmlHeaderSplitter
from ragengine.splitters.markdown_header import MarkdownHeaderSplitter
from ragengine.splitters.recursive_character import RecursiveCharacterSplitter

_REGISTRY: dict[str, type[Splitter]] = {
    "character": CharacterSplitter,
    "recursive_character": RecursiveCharacterSplitter,
    "code": CodeSplitter,
    "markdown_header": MarkdownHeaderSplitter,
    "html_header": HtmlHeaderSplitter,
}


def available_splitters() -> list[str]:
    return sorted(_REGISTRY.keys())


def get_splitter(name: str, **kwargs) -> Splitter:
    try:
        cls = _REGISTRY[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown splitter {name!r}. Available: {available_splitters()}"
        ) from exc
    return cls(**kwargs)
