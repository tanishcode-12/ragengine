"""Real local embedding model via `sentence-transformers`.

Mirrors the HuggingFace half of `Embed_documents_with_watsonx's_embedding-v1`
(which compared `WatsonxEmbeddings` against
`HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")`).
This is the free/local counterpart of that comparison — same default
model id, no IBM Cloud project/API key required.

The import is lazy (inside `__init__`) so that simply importing this
module — or the package as a whole — never requires `sentence-transformers`
or `torch` to be installed. Only *instantiating* this specific backend
does, and only then does it need network access to huggingface.co to
download the model weights the first time it runs.
"""

from __future__ import annotations

from ragengine.embeddings.base import EmbeddingModel

_DEFAULT_MODEL = "sentence-transformers/all-mpnet-base-v2"


class SentenceTransformerEmbedding(EmbeddingModel):
    name = "sentence_transformers"

    def __init__(self, model_name: str = _DEFAULT_MODEL):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise ImportError(
                "SentenceTransformerEmbedding requires the 'local-models' extra: "
                "pip install 'ragengine[local-models]'"
            ) from exc

        self.model_name = model_name
        self._model = SentenceTransformer(model_name)
        self.dimensions = self._model.get_sentence_embedding_dimension()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        vector = self._model.encode([text], convert_to_numpy=True, show_progress_bar=False)[0]
        return vector.tolist()
