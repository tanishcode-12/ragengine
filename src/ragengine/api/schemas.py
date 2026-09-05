"""Pydantic request/response models for the FastAPI layer.

Kept separate from routes.py so the API's public contract is readable in
one place (and reusable for client codegen / OpenAPI docs) without wading
through endpoint logic.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class IngestResponse(BaseModel):
    documents_loaded: int
    chunks_indexed: int
    splitter: str | None = None
    retriever: str | None = None


class IngestTextRequest(BaseModel):
    text: str = Field(..., min_length=1)
    source_name: str | None = Field(None, description="Stored as metadata['source']")
    splitter_name: str | None = Field(None, description="Defaults to the server's configured default splitter")


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    retriever_name: str | None = Field(
        None,
        description="similarity | mmr | similarity_score_threshold | multi_query | parent_document | "
        "self_query | rerank. Defaults to the server's configured default retriever.",
    )
    k: int | None = Field(None, ge=1, le=50, description="Number of chunks to retrieve")


class SourceItem(BaseModel):
    content: str
    metadata: dict[str, Any]
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceItem]


class AgentQueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    retriever_name: str | None = None
    k: int | None = Field(None, ge=1, le=50)


class AgentTraceStep(BaseModel):
    action: str
    query: str | None = None
    num_results: int | None = None
    answer: str | None = None
    forced: bool | None = None
    unparsed: bool | None = None


class AgentQueryResponse(BaseModel):
    answer: str
    trace: list[AgentTraceStep]


class HealthResponse(BaseModel):
    status: str
    embedding_backend: str
    llm_backend: str
    vector_store_backend: str
    main_index_count: int
    parent_document_index_count: int