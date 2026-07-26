"""RagPipeline — the single facade wiring every stage together.

This is what replaces "run all cells in order" from the source notebooks:
one object, built from `Settings`, exposing three operations —
`ingest_file`, `query`, `agent_query` — instead of a notebook where
ingestion and querying are whichever cells happen to run next.

Retriever construction is centralized in `get_retriever` because the five
retriever strategies need different dependencies wired in (see
`retrievers/registry.py`); everything else in this class is orchestration,
not technique — the actual techniques live in the `loaders/`, `splitters/`,
`embeddings/`, `vectorstores/`, `retrievers/`, and `agent/` packages.
"""

from __future__ import annotations

from ragengine.agent.orchestrator import AgentOrchestrator, AgentResult
from ragengine.agent.tools import RetrieverTool
from ragengine.config import Settings, settings as default_settings
from ragengine.embeddings import get_embedding_model
from ragengine.embeddings.base import EmbeddingModel
from ragengine.llm import get_llm
from ragengine.llm.base import LLMProvider
from ragengine.loaders import get_loader, get_loader_for_path
from ragengine.retrievers.base import Retriever
from ragengine.retrievers.multi_query import MultiQueryRetriever
from ragengine.retrievers.parent_document import ParentDocumentRetriever
from ragengine.retrievers.self_query import SelfQueryRetriever
from ragengine.retrievers.similarity import VectorStoreRetriever
from ragengine.splitters import get_splitter
from ragengine.vectorstores import get_vector_store
from ragengine.vectorstores.base import VectorStore

_ANSWER_SYSTEM = (
    "You are a helpful assistant that answers strictly from the given context. "
    "If the context doesn't contain the answer, say so plainly rather than guessing."
)

_SIMPLE_RETRIEVER_NAMES = {"similarity", "mmr", "similarity_score_threshold"}


class RagPipeline:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or default_settings

        self.embedding_model: EmbeddingModel = get_embedding_model(
            self.settings.embedding_backend,
            **self._embedding_kwargs(),
        )
        self.llm: LLMProvider = get_llm(self.settings.llm_backend, **self._llm_kwargs())
        self.vector_store: VectorStore = self._make_vector_store(f"{self.settings.collection_name}_main")

        # ParentDocumentRetriever carries its own state (a parent-id ->
        # Document map) across ingest() and retrieve() calls, so it must be
        # a single long-lived instance rather than rebuilt per request.
        self._parent_document_retriever: ParentDocumentRetriever | None = None

    # -- construction helpers -------------------------------------------------

    def _embedding_kwargs(self) -> dict:
        if self.settings.embedding_backend == "fake":
            return {"dimensions": self.settings.embedding_dimensions}
        if self.settings.embedding_backend == "sentence_transformers":
            return {"model_name": self.settings.embedding_model_name}
        return {}

    def _llm_kwargs(self) -> dict:
        if self.settings.llm_backend == "ollama":
            return {"model_name": self.settings.llm_model_name, "base_url": self.settings.ollama_base_url}
        if self.settings.llm_backend == "huggingface":
            return {"model_name": self.settings.llm_model_name}
        return {}

    def _make_vector_store(self, collection_name: str) -> VectorStore:
        backend = self.settings.vector_store_backend
        if backend == "chroma":
            return get_vector_store(
                "chroma", collection_name=collection_name, persist_dir=self.settings.vector_store_persist_dir
            )
        if backend == "faiss":
            return get_vector_store("faiss", dimensions=self.embedding_model.dimensions)
        raise ValueError(f"Unknown vector_store_backend: {backend!r}")

    def _build_parent_document_retriever(
        self,
        child_chunk_size: int = 200,
        child_chunk_overlap: int = 20,
        parent_chunk_size: int | None = None,
        parent_chunk_overlap: int = 0,
    ) -> ParentDocumentRetriever:
        if self._parent_document_retriever is not None:
            return self._parent_document_retriever

        # Deliberately isolated from self.vector_store: child chunks are
        # indexed only so ParentDocumentRetriever can map them back to a
        # parent, and would otherwise leak into plain similarity search
        # results from an unrelated retriever sharing the same collection.
        pd_vector_store = self._make_vector_store(f"{self.settings.collection_name}_parent_child")
        child_splitter = get_splitter(
            "recursive_character", chunk_size=child_chunk_size, chunk_overlap=child_chunk_overlap
        )
        parent_splitter = (
            get_splitter("recursive_character", chunk_size=parent_chunk_size, chunk_overlap=parent_chunk_overlap)
            if parent_chunk_size
            else None
        )
        self._parent_document_retriever = ParentDocumentRetriever(
            vector_store=pd_vector_store,
            embedding_model=self.embedding_model,
            child_splitter=child_splitter,
            parent_splitter=parent_splitter,
        )
        return self._parent_document_retriever

    # -- retriever construction -----------------------------------------------

    def get_retriever(self, name: str | None = None, **kwargs) -> Retriever:
        name = name or self.settings.default_retriever

        if name in _SIMPLE_RETRIEVER_NAMES:
            return VectorStoreRetriever(self.vector_store, self.embedding_model, search_type=name, **kwargs)

        if name == "multi_query":
            base_retriever = kwargs.pop("base_retriever", None) or VectorStoreRetriever(
                self.vector_store, self.embedding_model, search_type="similarity"
            )
            return MultiQueryRetriever(base_retriever=base_retriever, llm=self.llm, **kwargs)

        if name == "self_query":
            metadata_field_info = kwargs.pop(
                "metadata_field_info", "(no metadata field descriptions were provided)"
            )
            return SelfQueryRetriever(
                self.vector_store, self.embedding_model, self.llm, metadata_field_info=metadata_field_info, **kwargs
            )

        if name == "parent_document":
            return self._build_parent_document_retriever(**kwargs)

        raise ValueError(
            f"Unknown retriever {name!r}. Available: "
            f"{sorted(_SIMPLE_RETRIEVER_NAMES | {'multi_query', 'self_query', 'parent_document'})}"
        )

    # -- public operations ------------------------------------------------------

    def ingest_file(
        self,
        path: str,
        loader_name: str | None = None,
        splitter_name: str | None = None,
        retriever_name: str | None = None,
        splitter_kwargs: dict | None = None,
        retriever_kwargs: dict | None = None,
    ) -> dict:
        loader = get_loader(loader_name) if loader_name else get_loader_for_path(path)
        documents = loader.load(path)
        retriever_name = retriever_name or self.settings.default_retriever

        if retriever_name == "parent_document":
            retriever = self._build_parent_document_retriever(**(retriever_kwargs or {}))
            retriever.ingest(documents)
            return {
                "documents_loaded": len(documents),
                "retriever": "parent_document",
                "chunks_indexed": retriever.vector_store.count(),
            }

        splitter_name = splitter_name or self.settings.default_splitter
        splitter = get_splitter(
            splitter_name,
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
            **(splitter_kwargs or {}),
        )
        chunks = splitter.split(documents)
        vectors = self.embedding_model.embed_documents([c.page_content for c in chunks])
        self.vector_store.add(chunks, vectors)
        return {
            "documents_loaded": len(documents),
            "splitter": splitter_name,
            "chunks_indexed": len(chunks),
        }

    def query(
        self,
        question: str,
        retriever_name: str | None = None,
        k: int | None = None,
        retriever_kwargs: dict | None = None,
    ) -> dict:
        k = k or self.settings.top_k
        retriever = self.get_retriever(retriever_name, **(retriever_kwargs or {}))
        results = retriever.retrieve(question, k=k)
        context = RetrieverTool.format_context(results)

        answer = self.llm.generate(
            f"Context:\n{context}\n\nQuestion: {question}",
            system=_ANSWER_SYSTEM,
        )
        return {
            "answer": answer,
            "sources": [
                {"content": doc.page_content, "metadata": doc.metadata, "score": score}
                for doc, score in results
            ],
        }

    def agent_query(
        self,
        question: str,
        retriever_name: str | None = None,
        k: int | None = None,
    ) -> AgentResult:
        retriever = self.get_retriever(retriever_name or "similarity")
        tool = RetrieverTool(retriever=retriever, k=k or self.settings.top_k)
        orchestrator = AgentOrchestrator(tool=tool, llm=self.llm, max_iterations=self.settings.agent_max_iterations)
        return orchestrator.run(question)

    def stats(self) -> dict:
        return {
            "embedding_backend": self.settings.embedding_backend,
            "llm_backend": self.settings.llm_backend,
            "vector_store_backend": self.settings.vector_store_backend,
            "main_index_count": self.vector_store.count(),
            "parent_document_index_count": (
                self._parent_document_retriever.vector_store.count()
                if self._parent_document_retriever
                else 0
            ),
        }
