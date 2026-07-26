"""CSV loader.

Mirrors LangChain's `CSVLoader`: one Document per row, with the row
rendered as `column: value` lines so column names stay attached to their
values in the embedded text, plus row/source metadata for traceability.
"""

from __future__ import annotations

import csv
from pathlib import Path

from ragengine.documents import Document
from ragengine.loaders.base import Loader


class CsvLoader(Loader):
    name = "csv"

    def __init__(self, source_column: str | None = None):
        # optional column whose value overrides the "source" metadata,
        # same option LangChain's CSVLoader exposes
        self.source_column = source_column

    def load(self, source: str) -> list[Document]:
        path = Path(source)
        docs = []
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                content = "\n".join(f"{col}: {val}" for col, val in row.items())
                row_source = row.get(self.source_column) if self.source_column else str(path)
                docs.append(
                    Document(
                        page_content=content,
                        metadata={"source": row_source or str(path), "row": i},
                    )
                )
        return docs
