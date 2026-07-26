"""Self-query retriever.

`LangChain_retriever-v1`'s `SelfQueryRetriever`: let the user phrase a
filter in natural language ("policies from after 2022 about leave") and
have an LLM split that into (a) the semantic search text and (b) a
structured metadata filter, instead of making the caller construct the
filter dict by hand.

The LLM is asked to reply with one JSON object only. `filters.matches`'s
operator set (`$eq/$ne/$gt/$gte/$lt/$lte/$in/$nin`, `$and`/`$or`) is
exactly what's described in the prompt, so the same filter works
identically against either vector store backend — see
`vectorstores/filters.py`.
"""

from __future__ import annotations

import json

from ragengine.documents import Document
from ragengine.embeddings.base import EmbeddingModel
from ragengine.llm.base import LLMProvider
from ragengine.retrievers.base import Retriever
from ragengine.vectorstores.base import VectorStore

_SYSTEM = (
    "You output only a single valid JSON object, no markdown fences, no commentary. "
    'The object has exactly two keys: "search_query" (the semantic portion of the '
    'request, as plain text) and "filter" (a metadata filter, or null if none applies). '
    'Filter syntax: {"field": value} for equality, or {"field": {"$gt": v}} using '
    "$eq/$ne/$gt/$gte/$lt/$lte/$in/$nin, optionally combined with top-level "
    '{"$and": [...]}/{"$or": [...]}.'
)


class SelfQueryRetriever(Retriever):
    name = "self_query"

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_model: EmbeddingModel,
        llm: LLMProvider,
        metadata_field_info: str,
    ):
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.llm = llm
        # human-readable description of the filterable fields and their
        # types/meaning, injected into the prompt so the LLM knows what it
        # is and isn't allowed to filter on, e.g.:
        #   "year (int, the policy's effective year), department (string)"
        self.metadata_field_info = metadata_field_info

    def parse_query(self, query: str) -> tuple[str, dict | None]:
        prompt = f"Filterable fields: {self.metadata_field_info}\n\nRequest: {query}"
        raw = self.llm.generate(prompt, system=_SYSTEM)
        return self._parse_response(raw, fallback_query=query)

    @staticmethod
    def _parse_response(raw: str, fallback_query: str) -> tuple[str, dict | None]:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:]
            cleaned = cleaned.strip()
        try:
            parsed = json.loads(cleaned)
        except (json.JSONDecodeError, TypeError):
            return fallback_query, None
        if not isinstance(parsed, dict):
            return fallback_query, None
        search_query = parsed.get("search_query") or fallback_query
        return search_query, parsed.get("filter")

    def retrieve(self, query: str, k: int = 4) -> list[tuple[Document, float]]:
        search_query, filter_ = self.parse_query(query)
        query_vec = self.embedding_model.embed_query(search_query)
        return self.vector_store.similarity_search_by_vector(query_vec, k=k, filter=filter_)
