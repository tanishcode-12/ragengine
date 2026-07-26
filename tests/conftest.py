"""Shared pytest fixtures.

Every fixture here builds fully offline components (fake embeddings, stub
LLM, in-memory Chroma) so the whole suite runs with no network access and
no model downloads. `pipeline`/`pipeline_factory` each get a unique
`collection_name` — see config.py's note on why that matters: Chroma's
in-memory client shares collections by name across every `Client()`
instance in the *same process*, so two tests using the same name would
silently see each other's data.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES_DIR


@pytest.fixture
def fake_embedding():
    from ragengine.embeddings import get_embedding_model

    return get_embedding_model("fake", dimensions=256)


@pytest.fixture
def stub_llm():
    from ragengine.llm import get_llm

    return get_llm("stub")


def _make_settings(**overrides):
    from ragengine.config import Settings

    defaults = dict(
        embedding_backend="fake",
        embedding_dimensions=256,
        vector_store_backend="chroma",
        llm_backend="stub",
        chunk_size=200,
        chunk_overlap=20,
        collection_name=f"test_{uuid.uuid4().hex[:12]}",
    )
    defaults.update(overrides)
    return Settings(**defaults)


@pytest.fixture
def pipeline_factory():
    """Returns a callable that builds a fresh, isolated RagPipeline.

    Use this (instead of the `pipeline` fixture) when a test needs
    non-default settings, e.g. a FAISS backend or a smaller chunk size.
    """
    from ragengine.pipeline import RagPipeline

    def _factory(**settings_overrides):
        return RagPipeline(_make_settings(**settings_overrides))

    return _factory


@pytest.fixture
def pipeline(pipeline_factory):
    """A fresh, fully offline RagPipeline with default test settings."""
    return pipeline_factory()


@pytest.fixture
def api_client(pipeline):
    """A TestClient wired to the `pipeline` fixture via dependency override,
    so API tests exercise the real routes/schemas against a controlled,
    isolated pipeline instead of the process-wide cached singleton."""
    from fastapi.testclient import TestClient

    from ragengine.api.deps import get_pipeline
    from ragengine.api.main import app

    app.dependency_overrides[get_pipeline] = lambda: pipeline
    client = TestClient(app)
    yield client
    app.dependency_overrides.pop(get_pipeline, None)
