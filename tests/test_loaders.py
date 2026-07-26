from pathlib import Path

import pytest

from ragengine.loaders import get_loader, get_loader_for_path, loader_name_for_path
from ragengine.loaders.json_loader import _apply_jq_like_schema


def test_text_loader(fixtures_dir):
    docs = get_loader_for_path(str(fixtures_dir / "sample.txt")).load(str(fixtures_dir / "sample.txt"))
    assert len(docs) == 1
    assert "vacation leave" in docs[0].page_content
    assert docs[0].metadata["source"].endswith("sample.txt")


def test_markdown_loader_preserves_headers_for_downstream_splitting(fixtures_dir):
    path = str(fixtures_dir / "sample.md")
    docs = get_loader_for_path(path).load(path)
    assert len(docs) == 1
    assert docs[0].page_content.startswith("# Employee Handbook")
    assert docs[0].metadata["format"] == "markdown"


def test_csv_loader_one_document_per_row(fixtures_dir):
    path = str(fixtures_dir / "sample.csv")
    docs = get_loader_for_path(path).load(path)
    assert len(docs) == 3  # 3 data rows
    assert "Alice" in docs[0].page_content
    assert "name: Alice" in docs[0].page_content
    assert docs[0].metadata["row"] == 0
    assert docs[2].metadata["row"] == 2


def test_json_loader_with_jq_style_schema(fixtures_dir):
    path = str(fixtures_dir / "sample.json")
    docs = get_loader("json", jq_schema=".messages[].content").load(path)
    assert len(docs) == 2
    assert "vacation leave" in docs[0].page_content
    assert "printer" in docs[1].page_content


def test_json_loader_with_content_key_keeps_rest_as_metadata(fixtures_dir):
    path = str(fixtures_dir / "sample.json")
    # ".messages[]" (no trailing ".content") yields the message dicts
    # themselves, so content_key="content" can split each into
    # page_content="content" + metadata=the remaining fields (here: "author").
    docs = get_loader("json", jq_schema=".messages[]", content_key="content").load(path)
    assert docs[0].page_content == "Full time employees receive 20 days of paid vacation leave per year."
    assert docs[0].metadata["author"] == "hr"
    assert docs[1].metadata["author"] == "it"


@pytest.mark.parametrize(
    "schema,expected",
    [
        (".", None),  # whole document returned as-is
        (".messages[].content", ["Full time employees receive 20 days of paid vacation leave per year.", "To reset your printer, hold the power button for 10 seconds."]),
    ],
)
def test_jq_like_schema_evaluator(schema, expected):
    data = {
        "messages": [
            {"content": "Full time employees receive 20 days of paid vacation leave per year."},
            {"content": "To reset your printer, hold the power button for 10 seconds."},
        ]
    }
    result = _apply_jq_like_schema(data, schema)
    if expected is None:
        assert result == [data]
    else:
        assert result == expected


def test_pdf_loader_pypdf_engine(fixtures_dir):
    path = str(fixtures_dir / "sample.pdf")
    docs = get_loader("pdf", engine="pypdf").load(path)
    assert len(docs) == 2  # two pages
    assert "Leave Policy" in docs[0].page_content
    assert docs[0].metadata["page"] == 0
    assert "Sick leave" in docs[1].page_content


def test_pdf_loader_pymupdf_engine(fixtures_dir):
    path = str(fixtures_dir / "sample.pdf")
    docs = get_loader("pdf", engine="pymupdf").load(path)
    assert len(docs) == 2
    assert docs[0].metadata["total_pages"] == 2
    assert "Leave Policy" in docs[0].page_content


def test_pdf_loader_rejects_unknown_engine():
    with pytest.raises(ValueError):
        get_loader("pdf", engine="not-a-real-engine")


def test_docx_loader(fixtures_dir):
    path = str(fixtures_dir / "sample.docx")
    docs = get_loader_for_path(path).load(path)
    assert len(docs) == 1
    assert "Employee Handbook" in docs[0].page_content
    assert "vacation leave" in docs[0].page_content


def test_web_loader_strips_script_style_and_nav_noise():
    # Exercises the documented "clean vs raw" contrast without a network
    # call (see WebLoader's module docstring) — real fetch is verified
    # manually, not in the automated suite.
    from ragengine.loaders.web_loader import WebLoader

    html = (Path(__file__).parent / "fixtures" / "sample.html").read_text()
    cleaned = WebLoader.extract_clean_text(html)
    assert "Full time employees receive 20 days" in cleaned
    assert "console.log" not in cleaned
    assert "color:red" not in cleaned
    assert "Home | About" not in cleaned  # nav stripped
    assert "Copyright notice" not in cleaned  # footer stripped


def test_loader_name_for_path_guesses_from_extension():
    assert loader_name_for_path("report.pdf") == "pdf"
    assert loader_name_for_path("notes.md") == "markdown"
    assert loader_name_for_path("data.CSV") == "csv"  # case-insensitive


def test_loader_name_for_path_raises_on_unknown_extension():
    with pytest.raises(ValueError):
        loader_name_for_path("archive.zip")


def test_get_loader_raises_on_unknown_name():
    with pytest.raises(ValueError):
        get_loader("not-a-real-loader")
