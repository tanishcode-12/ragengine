"""Parent-document retriever.

`LangChain_retriever-v1`'s `ParentDocumentRetriever`: index small child
chunks (good for precise semantic search) but return their larger parent
chunk (good for giving the LLM enough surrounding context to actually
answer from). This is the one retriever that needs nonstandard ingestion
— see `supports_custom_ingestion` on the base class — because it indexes
two granularities of the same source documents instead of one.
"""

from __future__ import annotations

import uuid

from ragengine.documents import Document
from ragengine.embeddings.base import EmbeddingModel
from ragengine.retrievers.base import Retriever
from ragengine.splitters.base import Splitter
from ragengine.vectorstores.base import VectorStore


class ParentDocumentRetriever(Retriever):
    name = "parent_document"
    supports_custom_ingestion = True

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_model: EmbeddingModel,
        child_splitter: Splitter,
        parent_splitter: Splitter | None = None,
    ):
        # vector_store holds CHILD chunks only, each tagged with a
        # "parent_id" in its metadata pointing back into _parent_store.
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.child_splitter = child_splitter
        # if no parent_splitter is given, the "parent" is the whole
        # original document passed to ingest() (e.g. one whole PDF page)
        self.parent_splitter = parent_splitter
        self._parent_store: dict[str, Document] = {}

    def ingest(self, documents: list[Document]) -> None:
        parents = self.parent_splitter.split(documents) if self.parent_splitter else documents

        for parent in parents:
            parent_id = parent.id or str(uuid.uuid4())
            parent.id = parent_id
            self._parent_store[parent_id] = parent

            children = self.child_splitter.split([parent])
            if not children:
                continue
            for child in children:
                child.metadata["parent_id"] = parent_id
            vectors = self.embedding_model.embed_documents([c.page_content for c in children])
            self.vector_store.add(children, vectors)

    def retrieve(self, query: str, k: int = 4) -> list[tuple[Document, float]]:
        query_vec = self.embedding_model.embed_query(query)
        # over-fetch children since several can share one parent
        candidates = self.vector_store.similarity_search_by_vector(query_vec, k=max(k * 4, 10))

        best_score_by_parent: dict[str, float] = {}
        for child_doc, score in candidates:
            parent_id = child_doc.metadata.get("parent_id")
            if not parent_id or parent_id not in self._parent_store:
                continue
            if parent_id not in best_score_by_parent or score > best_score_by_parent[parent_id]:
                best_score_by_parent[parent_id] = score
            if len(best_score_by_parent) >= k:
                break

        ranked = sorted(best_score_by_parent.items(), key=lambda pair: pair[1], reverse=True)
        return [(self._parent_store[pid], score) for pid, score in ranked[:k]]
