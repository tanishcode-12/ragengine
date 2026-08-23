import pytest

from ragengine.documents import Document
from ragengine.rerankers import available_rerankers, get_reranker
from ragengine.rerankers.lexical_overlap import LexicalOverlapReranker


def test_available_rerankers():
    assert available_rerankers() == ["cross_encoder", "lexical_overlap"]


def test_get_reranker_raises_on_unknown_name():
    with pytest.raises(ValueError):
        get_reranker("not-a-real-reranker")


def test_get_reranker_returns_configured_backend():
    reranker = get_reranker("lexical_overlap")
    assert isinstance(reranker, LexicalOverlapReranker)


def test_lexical_overlap_scores_stronger_term_matches_higher():
    reranker = LexicalOverlapReranker()
    docs = [
        Document(page_content="paid leave"),
        Document(page_content="Employees receive twenty days of paid leave every year, plus sick leave."),
        Document(page_content="The office kitchen was renovated last spring."),
    ]
    scores = reranker.score("how many days of paid leave do employees get", docs)
    assert len(scores) == 3
    # doc[1] shares far more query vocabulary than doc[0] or doc[2]
    assert scores[1] > scores[0]
    assert scores[1] > scores[2]


def test_lexical_overlap_scores_zero_for_no_shared_terms():
    reranker = LexicalOverlapReranker()
    scores = reranker.score("paid leave policy", [Document(page_content="unrelated content entirely")])
    assert scores == [0.0]


def test_lexical_overlap_handles_empty_document_list():
    reranker = LexicalOverlapReranker()
    assert reranker.score("anything", []) == []


def test_reranker_rerank_respects_k_and_sorts_descending():
    reranker = LexicalOverlapReranker()
    docs = [
        Document(page_content="paid leave amount and paid leave policy details"),
        Document(page_content="paid leave"),
        Document(page_content="nothing relevant here"),
    ]
    candidates = [(doc, 0.0) for doc in docs]  # first-pass scores irrelevant to a reranker
    reranked = reranker.rerank("paid leave policy", candidates, k=2)
    assert len(reranked) == 2
    assert reranked[0][1] >= reranked[1][1]
    assert reranked[0][0].page_content.startswith("paid leave amount")


def test_reranker_rerank_handles_empty_candidates():
    reranker = LexicalOverlapReranker()
    assert reranker.rerank("query", [], k=3) == []


def test_cross_encoder_reranker_requires_local_models_extra_when_unavailable():
    from ragengine.rerankers.cross_encoder import CrossEncoderReranker

    try:
        import sentence_transformers  # noqa: F401

        pytest.skip("sentence-transformers is installed; import-guard path not exercised")
    except ImportError:
        with pytest.raises(ImportError, match="local-models"):
            CrossEncoderReranker()
