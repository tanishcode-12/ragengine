"""FastAPI dependencies.

`get_pipeline` is cached (one instance per process) rather than
constructed per-request, because the vector store it holds is stateful —
a fresh `RagPipeline()` per request would silently wipe an in-memory
Chroma/FAISS index on every single call. See README.md's "Running with
multiple workers" note: this per-process singleton means in-memory
backends do NOT share state across `uvicorn --workers N > 1`; use
`vector_store_persist_dir` (Chroma) or a real external vector database for
that deployment shape.
"""

from __future__ import annotations

from functools import lru_cache

from ragengine.config import Settings
from ragengine.pipeline import RagPipeline


@lru_cache
def get_pipeline() -> RagPipeline:
    return RagPipeline(Settings())
