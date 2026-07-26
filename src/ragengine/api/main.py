"""FastAPI application entry point.

Run with: `uvicorn ragengine.api.main:app --reload`
Docs at:  http://localhost:8000/docs
"""

from __future__ import annotations

from fastapi import FastAPI

from ragengine import __version__
from ragengine.api.routes import router

app = FastAPI(
    title="ragengine",
    description="A modular, local-first RAG engine with pluggable loaders, splitters, "
    "embeddings, vector stores, retrievers, and an agentic retrieval mode.",
    version=__version__,
)
app.include_router(router)


@app.get("/")
def root() -> dict:
    return {"name": "ragengine", "version": __version__, "docs": "/docs", "health": "/health"}
