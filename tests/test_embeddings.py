import numpy as np
import pytest

from ragengine.embeddings import available_embedding_backends, get_embedding_model


def test_available_embedding_backends():
    assert available_embedding_backends() == ["fake", "sentence_transformers"]


def test_fake_embedding_is_deterministic():
    emb = get_embedding_model("fake", dimensions=64)
    v1 = emb.embed_query("hello world")
    v2 = emb.embed_query("hello world")
    assert v1 == v2


def test_fake_embedding_respects_requested_dimensions():
    emb = get_embedding_model("fake", dimensions=128)
    assert emb.dimensions == 128
    assert len(emb.embed_query("anything")) == 128


def test_fake_embedding_is_l2_normalized():
    emb = get_embedding_model("fake", dimensions=256)
    v = np.array(emb.embed_query("some reasonably long piece of text to embed"))
    assert np.isclose(np.linalg.norm(v), 1.0)


def test_fake_embedding_ranks_lexically_similar_text_higher():
    # At the class default dimensionality, shared-vocabulary texts should
    # score clearly higher than unrelated text — see embeddings/deterministic_fake.py
    # for why this "hashing trick" property is what makes fake embeddings a
    # meaningful stand-in for retrieval tests, not just noise.
    emb = get_embedding_model("fake", dimensions=256)
    query = emb.embed_query("how much paid leave do I get")
    related = emb.embed_documents(["Employees get extra paid leave after five years of tenure."])[0]
    unrelated = emb.embed_documents(["To reset your printer, hold the power button for 10 seconds."])[0]

    sim_related = float(np.dot(query, related))
    sim_unrelated = float(np.dot(query, unrelated))
    assert sim_related > sim_unrelated


def test_embed_documents_batch_matches_embed_query_for_same_text():
    emb = get_embedding_model("fake", dimensions=64)
    text = "a consistent piece of text"
    assert emb.embed_documents([text])[0] == emb.embed_query(text)


def test_get_embedding_model_raises_on_unknown_backend():
    with pytest.raises(ValueError):
        get_embedding_model("not-a-real-backend")


def test_sentence_transformer_embedding_import_is_lazy():
    # Importing the embeddings package must never require torch/sentence-transformers
    # to be installed - only instantiating SentenceTransformerEmbedding does.
    import ragengine.embeddings.sentence_transformer  # noqa: F401
