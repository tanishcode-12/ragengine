import pytest

from ragengine.documents import Document
from ragengine.splitters import available_splitters, get_splitter


def test_available_splitters_lists_all_five():
    assert available_splitters() == [
        "character",
        "code",
        "html_header",
        "markdown_header",
        "recursive_character",
    ]


def test_character_splitter_splits_on_blank_lines():
    doc = Document(page_content="Paragraph one.\n\nParagraph two.\n\nParagraph three.")
    # chunk_size must be smaller than any two merged paragraphs combined,
    # or CharacterTextSplitter will greedily merge adjacent pieces back
    # together (by design) up to chunk_size before finalizing chunks.
    chunks = get_splitter("character", chunk_size=20, chunk_overlap=0, separator="\n\n").split([doc])
    assert [c.page_content for c in chunks] == ["Paragraph one.", "Paragraph two.", "Paragraph three."]


def test_character_splitter_carries_forward_parent_metadata():
    doc = Document(page_content="a" * 5 + "\n\n" + "b" * 5, metadata={"source": "x.txt"})
    chunks = get_splitter("character", chunk_size=10, chunk_overlap=0).split([doc])
    for chunk in chunks:
        assert chunk.metadata["source"] == "x.txt"
        assert chunk.metadata["splitter"] == "character"


def test_recursive_character_splitter_respects_chunk_size():
    text = "This is a longer piece of text that should get split into multiple recursive chunks nicely."
    chunks = get_splitter("recursive_character", chunk_size=30, chunk_overlap=5).split(
        [Document(page_content=text)]
    )
    assert len(chunks) > 1
    for c in chunks:
        assert len(c.page_content) <= 30
    # overlap means the join isn't lossless, but every chunk should be a
    # real substring of the source
    for c in chunks:
        assert c.page_content in text


def test_code_splitter_splits_along_function_boundaries():
    code = "def foo():\n    return 1\n\ndef bar():\n    return 2\n"
    chunks = get_splitter("code", language="python", chunk_size=40, chunk_overlap=0).split(
        [Document(page_content=code)]
    )
    joined = "\n".join(c.page_content for c in chunks)
    assert "def foo" in joined and "def bar" in joined
    assert all(c.metadata["language"] == "python" for c in chunks)


def test_code_splitter_rejects_unknown_language():
    with pytest.raises(ValueError):
        get_splitter("code", language="cobol")


def test_markdown_header_splitter_attaches_header_metadata():
    md = "# Title\ncontent A here that is reasonably long for the test.\n## Sub\ncontent B also fairly long.\n"
    chunks = get_splitter("markdown_header", chunk_size=200, chunk_overlap=0).split(
        [Document(page_content=md, metadata={"source": "handbook.md"})]
    )
    top_level = [c for c in chunks if "Header 2" not in c.metadata]
    nested = [c for c in chunks if c.metadata.get("Header 2") == "Sub"]
    assert any("content A" in c.page_content for c in top_level)
    assert all(c.metadata["Header 1"] == "Title" for c in chunks)
    assert any("content B" in c.page_content for c in nested)
    # parent metadata (source) must survive the header split
    assert all(c.metadata["source"] == "handbook.md" for c in chunks)


def test_markdown_header_splitter_further_splits_long_sections():
    long_section = "word " * 100
    md = f"# Title\n{long_section}"
    chunks = get_splitter("markdown_header", chunk_size=50, chunk_overlap=0).split(
        [Document(page_content=md)]
    )
    assert len(chunks) > 1
    assert all(c.metadata["Header 1"] == "Title" for c in chunks)


def test_html_header_splitter_attaches_header_metadata():
    html = "<html><body><h1>Title</h1><p>intro text here</p><h2>Sub</h2><p>sub text here</p></body></html>"
    chunks = get_splitter("html_header", chunk_size=200, chunk_overlap=0).split([Document(page_content=html)])
    assert any(c.metadata.get("Header 2") == "Sub" for c in chunks)
    assert any("sub text" in c.page_content for c in chunks)


def test_get_splitter_raises_on_unknown_name():
    with pytest.raises(ValueError):
        get_splitter("not-a-real-splitter")
