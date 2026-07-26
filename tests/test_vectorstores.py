import uuid

import pytest

from ragengine.documents import Document
from ragengine.vectorstores import available_vector_stores, get_vector_store

DIM = 8


def _store(backend: str):
    if backend == "chroma":
        return get_vector_store("chroma", collection_name=f"test_{uuid.uuid4().hex[:12]}")
    return get_vector_store("faiss", dimensions=DIM)


@pytest.fixture(params=["chroma", "faiss"])
def backend(request):
    return request.param


@pytest.fixture
def store(backend):
    return _store(backend)


def test_available_vector_stores():
    assert available_vector_stores() == ["chroma", "faiss"]


def test_add_and_count(store):
    docs = [Document(page_content="a"), Document(page_content="b")]
    vectors = [[1.0] + [0.0] * (DIM - 1), [0.0, 1.0] + [0.0] * (DIM - 2)]
    ids = store.add(docs, vectors)
    assert len(ids) == 2
    assert store.count() == 2
    # ids are also written back onto the Document objects
    assert docs[0].id == ids[0]


def test_add_rejects_mismatched_lengths(store):
    with pytest.raises(ValueError):
        store.add([Document(page_content="a")], [[1.0] * DIM, [0.0] * DIM])


def test_similarity_search_returns_closest_first(store):
    docs = [
        Document(page_content="north", metadata={"tag": "n"}),
        Document(page_content="south", metadata={"tag": "s"}),
    ]
    vectors = [[1.0] + [0.0] * (DIM - 1), [-1.0] + [0.0] * (DIM - 1)]
    store.add(docs, vectors)

    results = store.similarity_search_by_vector([1.0] + [0.0] * (DIM - 1), k=2)
    assert results[0][0].metadata["tag"] == "n"
    assert results[0][1] > results[1][1]


def test_similarity_search_respects_k(store):
    docs = [Document(page_content=f"doc {i}") for i in range(5)]
    vectors = [[float(i)] + [0.0] * (DIM - 1) for i in range(5)]
    store.add(docs, vectors)
    results = store.similarity_search_by_vector([1.0] + [0.0] * (DIM - 1), k=2)
    assert len(results) == 2


def test_metadata_filter(store):
    docs = [
        Document(page_content="a", metadata={"year": 2020}),
        Document(page_content="b", metadata={"year": 2023}),
        Document(page_content="c", metadata={"year": 2024}),
    ]
    vectors = [[1.0] + [0.0] * (DIM - 1)] * 3
    store.add(docs, vectors)

    results = store.similarity_search_by_vector([1.0] + [0.0] * (DIM - 1), k=10, filter={"year": {"$gte": 2023}})
    years = sorted(doc.metadata["year"] for doc, _ in results)
    assert years == [2023, 2024]


def test_update_requires_embeddings_when_documents_given(store):
    docs = [Document(page_content="a")]
    ids = store.add(docs, [[1.0] * DIM])
    with pytest.raises(ValueError):
        store.update(ids, documents=[Document(page_content="a-updated")])


def test_update_changes_content_and_embedding(store):
    docs = [Document(page_content="a", metadata={"v": 1})]
    ids = store.add(docs, [[1.0] + [0.0] * (DIM - 1)])

    new_doc = Document(page_content="a-updated", metadata={"v": 2})
    store.update(ids, documents=[new_doc], embeddings=[[0.0, 1.0] + [0.0] * (DIM - 2)])

    [fetched] = store.get_by_ids(ids)
    assert fetched.page_content == "a-updated"
    assert fetched.metadata["v"] == 2

    # the vector moved too - searching near the OLD vector should no
    # longer return this doc as the top hit once something else claims it
    results = store.similarity_search_by_vector([0.0, 1.0] + [0.0] * (DIM - 2), k=1)
    assert results[0][0].page_content == "a-updated"


def test_delete_removes_from_search_and_count(store):
    docs = [Document(page_content="a"), Document(page_content="b")]
    ids = store.add(docs, [[1.0] + [0.0] * (DIM - 1), [0.0, 1.0] + [0.0] * (DIM - 2)])
    store.delete([ids[0]])
    assert store.count() == 1
    remaining = store.get_by_ids(ids)
    assert len(remaining) == 1
    assert remaining[0].page_content == "b"


def test_get_by_ids_preserves_requested_order(store):
    docs = [Document(page_content="a"), Document(page_content="b"), Document(page_content="c")]
    ids = store.add(docs, [[float(i)] + [0.0] * (DIM - 1) for i in range(3)])
    fetched = store.get_by_ids([ids[2], ids[0], ids[1]])
    assert [d.page_content for d in fetched] == ["c", "a", "b"]


def test_none_and_empty_metadata_are_handled_safely(store):
    # Regression test for the two Chroma gotchas found while building this
    # (None metadata values and empty metadata dicts both raise inside
    # chromadb) - both backends must accept these without erroring.
    docs = [
        Document(page_content="a", metadata={"title": None, "year": 2020}),
        Document(page_content="b", metadata={}),
    ]
    ids = store.add(docs, [[1.0] * DIM, [0.0, 1.0] + [0.0] * (DIM - 2)])
    fetched = store.get_by_ids(ids)
    assert len(fetched) == 2


def test_similarity_search_on_empty_store_returns_empty(store):
    assert store.similarity_search_by_vector([1.0] * DIM, k=5) == []
