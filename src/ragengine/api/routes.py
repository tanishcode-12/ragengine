"""API routes.

Four endpoints: ingest a file, ingest raw text, query (fixed one-shot
retrieval), and agent-query (the agentic RAG mode). Each wraps
`RagPipeline` — see pipeline.py for the actual orchestration; this module
is deliberately thin (HTTP concerns only: file handling, status codes,
response shaping).
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from ragengine.api.deps import get_pipeline
from ragengine.api.schemas import (
    AgentQueryRequest,
    AgentQueryResponse,
    HealthResponse,
    IngestResponse,
    IngestTextRequest,
    QueryRequest,
    QueryResponse,
    SourceItem,
)
from ragengine.documents import Document
from ragengine.pipeline import RagPipeline
from ragengine.splitters import get_splitter

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health(pipeline: RagPipeline = Depends(get_pipeline)) -> HealthResponse:
    return HealthResponse(status="ok", **pipeline.stats())


@router.post("/ingest", response_model=IngestResponse)
async def ingest_file(
    file: UploadFile = File(...),
    loader_name: str | None = Form(None),
    splitter_name: str | None = Form(None),
    retriever_name: str | None = Form(None),
    pipeline: RagPipeline = Depends(get_pipeline),
) -> IngestResponse:
    suffix = Path(file.filename or "").suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        result = pipeline.ingest_file(
            tmp_path,
            loader_name=loader_name,
            splitter_name=splitter_name,
            retriever_name=retriever_name,
        )
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return IngestResponse(**result)


@router.post("/ingest/text", response_model=IngestResponse)
def ingest_text(body: IngestTextRequest, pipeline: RagPipeline = Depends(get_pipeline)) -> IngestResponse:
    """Ingest raw text with no file upload — handy for quick testing/demos."""
    doc = Document(page_content=body.text, metadata={"source": body.source_name or "inline-text"})
    splitter_name = body.splitter_name or pipeline.settings.default_splitter

    try:
        splitter = get_splitter(
            splitter_name,
            chunk_size=pipeline.settings.chunk_size,
            chunk_overlap=pipeline.settings.chunk_overlap,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    chunks = splitter.split([doc])
    vectors = pipeline.embedding_model.embed_documents([c.page_content for c in chunks])
    pipeline.vector_store.add(chunks, vectors)
    return IngestResponse(documents_loaded=1, splitter=splitter_name, chunks_indexed=len(chunks))


@router.post("/query", response_model=QueryResponse)
def query(body: QueryRequest, pipeline: RagPipeline = Depends(get_pipeline)) -> QueryResponse:
    try:
        result = pipeline.query(body.question, retriever_name=body.retriever_name, k=body.k)
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return QueryResponse(answer=result["answer"], sources=[SourceItem(**s) for s in result["sources"]])


@router.post("/agent/query", response_model=AgentQueryResponse)
def agent_query(body: AgentQueryRequest, pipeline: RagPipeline = Depends(get_pipeline)) -> AgentQueryResponse:
    try:
        result = pipeline.agent_query(body.question, retriever_name=body.retriever_name, k=body.k)
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return AgentQueryResponse(answer=result.answer, trace=result.trace)
