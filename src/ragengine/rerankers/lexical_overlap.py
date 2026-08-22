"""Deterministic, dependency-free reranker: BM25 over the candidate pool.

Plays the same role here that `DeterministicHashEmbedding` plays for
embeddings — a real, usable-by-default implementation that needs no model
download and no network access, so tests, CI, and a first `docker compose
up` all get meaningful reranking behaviour with zero external dependencies.
Unlike the hash embedding (which is explicitly *not* semantic), this one
is a legitimate, if simple, information-retrieval technique: BM25 is a
strong classical baseline that real search engines used in production for
decades before cross-encoders existed, so this is a genuine default, not
only a test double.

Deliberately scoped to the candidate pool it's given (typically
`fetch_k` documents from a first-pass retriever), not the whole corpus:
IDF is computed from term frequency across just those candidates. That
makes it a *reranker* — it only ever sharpens an existing shortlist — and
keeps it cheap enough to run on every request with no precomputed index.
"""

from __future__ import annotations

import math
import re
from collections import Counter

from ragengine.documents import Document
from ragengine.rerankers.base import Reranker

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Standard BM25 free parameters: k1 controls term-frequency saturation,
# b controls how much document-length normalization matters.
_K1 = 1.5
_B = 0.75


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class LexicalOverlapReranker(Reranker):
    """BM25 scoring of `documents` against `query`, computed fresh per call."""

    name = "lexical_overlap"

    def score(self, query: str, documents: list[Document]) -> list[float]:
        if not documents:
            return []

        query_terms = _tokenize(query)
        doc_tokens = [_tokenize(doc.page_content) for doc in documents]
        doc_lengths = [len(toks) for toks in doc_tokens]
        avg_len = (sum(doc_lengths) / len(doc_lengths)) if doc_lengths else 0.0

        n_docs = len(documents)
        # Document frequency: how many candidate documents each query term
        # appears in at least once.
        df: Counter[str] = Counter()
        for toks in doc_tokens:
            present = set(toks)
            for term in set(query_terms):
                if term in present:
                    df[term] += 1

        scores: list[float] = []
        for toks, length in zip(doc_tokens, doc_lengths):
            term_counts = Counter(toks)
            score = 0.0
            for term in query_terms:
                tf = term_counts.get(term, 0)
                if tf == 0:
                    continue
                # Standard BM25 IDF, floored at 0 so a term present in
                # every candidate never contributes a negative weight.
                idf = math.log(1 + (n_docs - df[term] + 0.5) / (df[term] + 0.5))
                idf = max(idf, 0.0)
                denom = tf + _K1 * (1 - _B + _B * (length / avg_len if avg_len else 1.0))
                score += idf * (tf * (_K1 + 1)) / (denom if denom else 1.0)
            scores.append(score)
        return scores
