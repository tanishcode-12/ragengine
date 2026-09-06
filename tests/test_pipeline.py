import pytest


def test_ingest_text_file_and_query(pipeline, fixtures_dir):
    ingest_result = pipeline.ingest_file(str(fixtures_dir / "sample.txt"))
    assert ingest_result["documents_loaded"] == 1
    assert ingest_result["chunks_indexed"] > 0

    result = pipeline.query("How many vacation days do employees get?", k=2)
    assert len(result["sources"]) == 2
    assert "vacation" in result["sources"][0]["content"].lower()
    assert isinstance(result["answer"], str) and result["answer"]


def test_ingest_markdown_file_preserves_header_metadata_when_using_that_splitter(pipeline, fixtures_dir):
    result = pipeline.ingest_file(str(fixtures_dir / "sample.md"), splitter_name="markdown_header")
    assert result["chunks_indexed"] > 0
    assert result["splitter"] == "markdown_header"


def test_ingest_pdf_file(pipeline, fixtures_dir):
    result = pipeline.ingest_file(str(fixtures_dir / "sample.pdf"))
    assert result["documents_loaded"] == 2  # two pages


def test_ingest_csv_file(pipeline, fixtures_dir):
    result = pipeline.ingest_file(str(fixtures_dir / "sample.csv"))
    assert result["documents_loaded"] == 3  # three data rows


def test_query_with_explicit_retriever_name(pipeline, fixtures_dir):
    pipeline.ingest_file(str(fixtures_dir / "sample.txt"))
    result = pipeline.query("vacation days", retriever_name="mmr", k=2)
    assert len(result["sources"]) <= 2


def test_query_with_rerank_retriever(pipeline, fixtures_dir):
    pipeline.ingest_file(str(fixtures_dir / "sample.txt"))
    result = pipeline.query("vacation days", retriever_name="rerank", k=2)
    assert len(result["sources"]) <= 2
    assert isinstance(result["answer"], str) and result["answer"]


def test_query_with_unknown_retriever_raises(pipeline, fixtures_dir):
    pipeline.ingest_file(str(fixtures_dir / "sample.txt"))
    with pytest.raises(ValueError):
        pipeline.query("vacation days", retriever_name="not-a-real-retriever")


def test_agent_query_end_to_end(pipeline, fixtures_dir):
    pipeline.ingest_file(str(fixtures_dir / "sample.txt"))
    result = pipeline.agent_query("How many vacation days do I get?")
    assert isinstance(result.answer, str) and result.answer
    assert isinstance(result.trace, list) and len(result.trace) >= 1


def test_parent_document_ingestion_and_retrieval_via_pipeline(pipeline_factory, fixtures_dir):
    pipeline = pipeline_factory()
    ingest_result = pipeline.ingest_file(
        str(fixtures_dir / "sample.txt"),
        retriever_name="parent_document",
        retriever_kwargs={"child_chunk_size": 60, "child_chunk_overlap": 10},
    )
    assert ingest_result["retriever"] == "parent_document"
    assert ingest_result["chunks_indexed"] > 0

    retriever = pipeline.get_retriever("parent_document")
    results = retriever.retrieve("vacation leave days", k=1)
    assert len(results) == 1
    # returned content should be a full parent (the whole sample.txt, since
    # no parent_splitter was configured), not a ~60-char child chunk
    assert len(results[0][0].page_content) > 200


def test_parent_document_uses_isolated_collection_from_main_store(pipeline_factory, fixtures_dir):
    pipeline = pipeline_factory()
    pipeline.ingest_file(str(fixtures_dir / "sample.txt"))  # goes to the main similarity index
    pipeline.ingest_file(
        str(fixtures_dir / "sample.md"), retriever_name="parent_document"
    )  # goes to the isolated parent/child index

    stats = pipeline.stats()
    assert stats["main_index_count"] > 0
    assert stats["parent_document_index_count"] > 0
    # a plain similarity search must not surface parent_document's child
    # chunks (they carry different metadata and would be meaningless
    # returned directly instead of resolved to their parent)
    result = pipeline.query("IT support password reset", k=5)
    for source in result["sources"]:
        assert "parent_id" not in source["metadata"]


def test_faiss_backend_end_to_end(pipeline_factory, fixtures_dir):
    pipeline = pipeline_factory(vector_store_backend="faiss")
    pipeline.ingest_file(str(fixtures_dir / "sample.txt"))
    result = pipeline.query("vacation days", k=2)
    assert len(result["sources"]) == 2


def test_stats_reports_backends_and_counts(pipeline, fixtures_dir):
    stats_before = pipeline.stats()
    assert stats_before["main_index_count"] == 0
    pipeline.ingest_file(str(fixtures_dir / "sample.txt"))
    stats_after = pipeline.stats()
    assert stats_after["main_index_count"] > 0
    assert stats_after["embedding_backend"] == "fake"
    assert stats_after["llm_backend"] == "stub"