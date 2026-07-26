"""ragengine — a modular, local-first Retrieval-Augmented Generation engine.

    from ragengine.pipeline import RagPipeline
    pipeline = RagPipeline()
    pipeline.ingest_file("handbook.pdf")
    result = pipeline.query("How many vacation days do employees get?")

See README.md for the full walkthrough, the available loaders/splitters/
embeddings/vector stores/retrievers, and the agentic RAG mode.
"""

from ragengine.pipeline import RagPipeline

__version__ = "0.1.0"
__all__ = ["RagPipeline", "__version__"]
