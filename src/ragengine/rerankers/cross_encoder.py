"""Real cross-encoder reranker via `sentence-transformers`.

Unlike a bi-encoder (the `SentenceTransformerEmbedding` used for
retrieval), a cross-encoder feeds `(query, document)` as a *single* input
to one transformer, so attention runs across both texts jointly rather
than comparing two independently-computed vectors. That's strictly more
expensive per pair (no precomputed document vectors, no ANN index — every
call is a real forward pass) but meaningfully more accurate, which is
exactly the recall-vs-precision trade this package's docstring describes:
affordable because `RerankingRetriever` only ever calls this on a small
first-pass candidate pool, never the whole corpus.

The import is lazy (inside `__init__`), same as `SentenceTransformerEmbedding`
— importing this module or the package never requires `sentence-transformers`
or `torch`; only instantiating this specific backend does, and only then
does it need network access to huggingface.co to download weights the
first time it runs.
"""

from __future__ import annotations

from ragengine.documents import Document
from ragengine.rerankers.base import Reranker

_DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker(Reranker):
    name = "cross_encoder"

    def __init__(self, model_name: str = _DEFAULT_MODEL):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise ImportError(
                "CrossEncoderReranker requires the 'local-models' extra: "
                "pip install 'ragengine[local-models]'"
            ) from exc

        self.model_name = model_name
        self._model = CrossEncoder(model_name)

    def score(self, query: str, documents: list[Document]) -> list[float]:
        if not documents:
            return []
        pairs = [(query, doc.page_content) for doc in documents]
        scores = self._model.predict(pairs)
        return [float(s) for s in scores]
