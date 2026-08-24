import pytest

from ragengine.documents import Document
from ragengine.llm import get_llm
from ragengine.retrievers import (
    MultiQueryRetriever,
    ParentDocumentRetriever,
    SelfQueryRetriever,
    VectorStoreRetriever,
    available_retrievers,
    get_retriever_class,
)
from ragengine.splitters import get_splitter
from ragengine.vectorstores import get_vector_store

LEAVE_DOCS = [
    Document(page_content="The vacation policy allows 20 days of paid leave per year.", metadata={"year": 2022}),
    Document(page_content="To reset your printer, hold the power button for 10 seconds.", metadata={"year": 2019}),
    Document(page_content="Employees get extra paid leave after five years of tenure.", metadata={"year": 2023}),
]


@pytest.fixture
def populated_store(fake_embedding):
    store = get_vector_store("chroma", collection_name="retriever_test_leave")
    vectors = fake_embedding.embed_documents([d.page_content for d in LEAVE_DOCS])
    store.add(list(LEAVE_DOCS), vectors)
    return store


def test_available_retrievers():
    assert available_retrievers() == [
        "mmr",
        "multi_query",
        "parent_document",
        "rerank",
        "self_query",
        "similarity",
        "similarity_score_threshold",
    ]

def test_get_retriever_class_raises_on_unknown_name():
    with pytest.raises(ValueError):
        get_retriever_class("not-a-real-retriever")


def test_similarity_retriever_ranks_relevant_docs_first(populated_store, fake_embedding):
    retriever = VectorStoreRetriever(populated_store, fake_embedding, search_type="similarity")
    results = retriever.retrieve("how much paid leave do I get", k=2)
    assert len(results) == 2
    assert "printer" not in results[0][0].page_content
    assert results[0][1] >= results[1][1]


def test_score_threshold_retriever_filters_out_low_scores(populated_store, fake_embedding):
    retriever = VectorStoreRetriever(
        populated_store, fake_embedding, search_type="similarity_score_threshold", score_threshold=0.9
    )
    # nothing scores that high against the fake embedding at this corpus size
    results = retriever.retrieve("how much paid leave do I get", k=3)
    assert results == []


def test_mmr_retriever_prefers_diversity_over_a_near_duplicate(fake_embedding):
    store = get_vector_store("faiss", dimensions=fake_embedding.dimensions)
    docs = [
        Document(page_content="Quarterly revenue grew due to strong enterprise sales this quarter."),
        Document(page_content="Revenue grew this quarter thanks to strong enterprise sales figures."),
        Document(page_content="The engineering team migrated the database to a new cluster."),
    ]
    store.add(docs, fake_embedding.embed_documents([d.page_content for d in docs]))

    plain = VectorStoreRetriever(store, fake_embedding, search_type="similarity")
    mmr = VectorStoreRetriever(store, fake_embedding, search_type="mmr", fetch_k=3, lambda_mult=0.5)

    plain_top2 = {doc.page_content for doc, _ in plain.retrieve("enterprise sales revenue growth", k=2)}
    mmr_top2 = {doc.page_content for doc, _ in mmr.retrieve("enterprise sales revenue growth", k=2)}

    # plain similarity picks both near-duplicate revenue sentences; MMR
    # should swap one out for the diverse "engineering" doc
    assert docs[0].page_content in plain_top2 and docs[1].page_content in plain_top2
    assert docs[2].page_content in mmr_top2


def test_multi_query_retriever_unions_and_dedupes(populated_store, fake_embedding):
    stub = get_llm("stub", responses=["leave entitlement\npaid time off amount"])
    base = VectorStoreRetriever(populated_store, fake_embedding, search_type="similarity")
    retriever = MultiQueryRetriever(base_retriever=base, llm=stub, num_queries=2)

    results = retriever.retrieve("paid leave amount", k=2)
    assert len(results) == 2  # deduped, not 2 queries x 2 results = 4
    assert stub.calls[0][0].startswith("Generate 2 alternative phrasings")


def test_multi_query_falls_back_to_original_query_if_llm_returns_nothing_usable(populated_store, fake_embedding):
    stub = get_llm("stub", default_response="   ")  # blank after stripping
    base = VectorStoreRetriever(populated_store, fake_embedding, search_type="similarity")
    retriever = MultiQueryRetriever(base_retriever=base, llm=stub, num_queries=2)
    # should not raise, and should still retrieve using the original query
    results = retriever.retrieve("paid leave amount", k=2)
    assert len(results) == 2


def test_parent_document_retriever_returns_full_parent_not_small_chunk(fake_embedding):
    store = get_vector_store("chroma", collection_name="parent_doc_test")
    child_splitter = get_splitter("recursive_character", chunk_size=40, chunk_overlap=0)
    retriever = ParentDocumentRetriever(
        vector_store=store, embedding_model=fake_embedding, child_splitter=child_splitter
    )

    long_text = "Section on leave. " + ("The vacation policy allows 20 days of paid leave per year. " * 3)
    retriever.ingest([Document(page_content=long_text, metadata={"source": "policy.txt"})])

    results = retriever.retrieve("paid leave days", k=1)
    assert len(results) == 1
    returned_doc, _ = results[0]
    assert returned_doc.page_content == long_text  # full parent, not a ~40-char child chunk
    assert len(returned_doc.page_content) > 40


def test_parent_document_retriever_supports_custom_ingestion_flag():
    assert ParentDocumentRetriever.supports_custom_ingestion is True
    assert VectorStoreRetriever.supports_custom_ingestion is False


def test_self_query_retriever_applies_llm_parsed_filter(populated_store, fake_embedding):
    stub = get_llm("stub", responses=['{"search_query": "paid leave", "filter": {"year": {"$gte": 2022}}}'])
    retriever = SelfQueryRetriever(
        vector_store=populated_store,
        embedding_model=fake_embedding,
        llm=stub,
        metadata_field_info="year (int)",
    )
    results = retriever.retrieve("paid leave policies from 2022 onward", k=5)
    years = {doc.metadata["year"] for doc, _ in results}
    assert years == {2022, 2023}
    assert 2019 not in years


def test_self_query_retriever_falls_back_gracefully_on_unparseable_llm_output(populated_store, fake_embedding):
    stub = get_llm("stub", responses=["not valid json at all"])
    retriever = SelfQueryRetriever(
        vector_store=populated_store, embedding_model=fake_embedding, llm=stub, metadata_field_info="year (int)"
    )
    # should not raise - falls back to using the raw query with no filter
    results = retriever.retrieve("paid leave amount", k=2)
    assert len(results) == 2


def test_self_query_retriever_strips_markdown_code_fences():
    from ragengine.retrievers.self_query import SelfQueryRetriever as SQ

    query, filter_ = SQ._parse_response(
        '```json\n{"search_query": "leave", "filter": null}\n```', fallback_query="fallback"
    )
    assert query == "leave"
    assert filter_ is None
