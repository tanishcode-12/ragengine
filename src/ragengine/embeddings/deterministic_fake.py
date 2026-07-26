"""Deterministic, dependency-free embedding.

Not a semantic embedding model — it's a *test double* for one, standing in
for `SentenceTransformerEmbedding` so the rest of the pipeline (vector
stores, retrievers, the API) can be built, run, and unit-tested with zero
downloads and zero network access. Retrieval tests need it to be more than
pure noise, though, so this uses the classic "hashing trick": each text is
tokenized, each token is hashed into one of `dimensions` buckets with a
random +/- sign (this is the same technique behind, e.g., scikit-learn's
`HashingVectorizer`), and the result is L2-normalized. Two texts that
share vocabulary land closer together in cosine distance than two that
don't — enough for retrieval-ranking tests to be meaningful, without it
being an actual semantic model.

Swap in `SentenceTransformerEmbedding` for real semantic search.
"""

from __future__ import annotations

import hashlib
import re

import numpy as np

from ragengine.embeddings.base import EmbeddingModel

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class DeterministicHashEmbedding(EmbeddingModel):
    name = "fake"

    def __init__(self, dimensions: int = 256):
        self.dimensions = dimensions

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        vec = np.zeros(self.dimensions, dtype=np.float64)
        for tok in _tokenize(text):
            digest = hashlib.sha256(tok.encode("utf-8")).digest()
            h = int.from_bytes(digest[:8], "big")
            idx = h % self.dimensions
            sign = 1.0 if (h >> 63) & 1 == 0 else -1.0
            vec[idx] += sign
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()
