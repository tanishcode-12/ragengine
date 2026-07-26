from ragengine.documents import Document


def test_document_defaults_to_empty_metadata():
    doc = Document(page_content="hello")
    assert doc.page_content == "hello"
    assert doc.metadata == {}
    assert doc.id is None


def test_document_with_id_sets_and_returns_self():
    doc = Document(page_content="hello")
    result = doc.with_id("abc123")
    assert doc.id == "abc123"
    assert result is doc
