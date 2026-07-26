"""Web page loader.

The document-loader notebook makes a specific point of contrasting two
approaches: fetching a page with raw `BeautifulSoup` (which drags in a lot
of `<script>`/`<style>`/nav noise alongside the real content) versus
`WebBaseLoader`, which extracts cleaner visible text. This loader
reproduces the `WebBaseLoader`-style clean extraction as the default
behaviour, and exposes `fetch_raw_html` separately so the "noisy" raw HTML
is still available for anyone who wants to see the difference the
notebook demonstrated.
"""

from __future__ import annotations

import requests
from bs4 import BeautifulSoup

from ragengine.documents import Document
from ragengine.loaders.base import Loader

_DEFAULT_HEADERS = {"User-Agent": "ragengine/0.1 (+https://example.invalid)"}


class WebLoader(Loader):
    name = "web"

    def __init__(self, timeout: float = 10.0):
        self.timeout = timeout

    def load(self, source: str) -> list[Document]:
        html = self._fetch(source)
        text = self.extract_clean_text(html)
        return [Document(page_content=text, metadata={"source": source})]

    def fetch_raw_html(self, source: str) -> str:
        """Return the untouched HTML — useful for showing why raw
        BeautifulSoup text extraction is noisier than the cleaned path."""
        return self._fetch(source)

    def _fetch(self, url: str) -> str:
        resp = requests.get(url, headers=_DEFAULT_HEADERS, timeout=self.timeout)
        resp.raise_for_status()
        return resp.text

    @staticmethod
    def extract_clean_text(html: str) -> str:
        soup = BeautifulSoup(html, "lxml")
        for tag in soup(["script", "style", "nav", "header", "footer", "noscript"]):
            tag.decompose()
        text = soup.get_text(separator="\n")
        lines = [line.strip() for line in text.splitlines()]
        return "\n".join(line for line in lines if line)
