"""Central configuration.

Every tunable knob in the pipeline (which embedding backend, which vector
store, chunk size, etc.) lives here and is overridable via environment
variables or a ``.env`` file — never hardcoded inside a pipeline stage.
This is what the source notebooks did NOT have: every notebook hardcoded
its model id, chunk size, and project id directly in the cell that used it.

Defaults are chosen so the whole system runs with zero external services
and zero API keys (fake embeddings, in-memory vector store, stub LLM) —
good for tests and a first `docker compose up`. Swap in real local models
by setting EMBEDDING_BACKEND / LLM_BACKEND, see README.md.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="RAG_", extra="ignore")

    # --- Splitting -------------------------------------------------------
    # 500/50 is a reasonable production default for prose; the source
    # notebooks used 100/20 for the sake of a small demo corpus.
    chunk_size: int = 500
    chunk_overlap: int = 50
    default_splitter: str = "recursive_character"

    # --- Embeddings --------------------------------------------------------
    # "fake" = deterministic, dependency-free, hash-based embedding. Good for
    # tests/CI and for running the API with no downloads at all.
    # "sentence_transformers" = real local embedding model (requires the
    # `local-models` extra and network access to huggingface.co the first
    # time it runs, to download weights).
    embedding_backend: str = "fake"
    embedding_model_name: str = "sentence-transformers/all-mpnet-base-v2"
    embedding_dimensions: int = 768  # only used by the fake backend

    # --- Vector store --------------------------------------------------------
    vector_store_backend: str = "chroma"  # "chroma" | "faiss"
    vector_store_persist_dir: str | None = None  # None = in-memory
    # Base name for the collection(s) this pipeline uses. Chroma's in-memory
    # client shares collections by name across every `chromadb.Client()`
    # instance in the same process (confirmed while building the test
    # suite) — two RagPipeline instances that don't override this will
    # silently share one index. Give each independent pipeline (e.g. each
    # test, or each tenant) its own collection_name.
    collection_name: str = "ragengine"

    # --- LLM (generation + agent reasoning) ---------------------------------
    # "stub" = deterministic, no model, no network — used by default and by
    # the test suite. "ollama" and "huggingface" are real local backends.
    llm_backend: str = "stub"
    llm_model_name: str = "llama3.2"
    ollama_base_url: str = "http://localhost:11434"

    # --- Retrieval -----------------------------------------------------------
    # "rerank" added: "similarity" | "mmr" | "multi_query" | "parent_document" | "self_query" | "rerank"
    default_retriever: str = "similarity"
    top_k: int = 4

    # --- Reranking -------------------------------------------------------------
    # "lexical_overlap" = deterministic BM25, dependency-free (default, matches
    # the "fake" embedding / "stub" LLM philosophy of zero-setup-by-default).
    # "cross_encoder" = real local cross-encoder model (requires the
    # `local-models` extra and network access to huggingface.co on first run).
    reranker_backend: str = "lexical_overlap"
    reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    # How many candidates the base retriever fetches before RerankingRetriever
    # narrows them down to top_k.
    rerank_fetch_k: int = 20

    # --- Agent ---------------------------------------------------------------
    agent_max_iterations: int = 3


settings = Settings()