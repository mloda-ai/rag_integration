"""BM25 lexical retrieve connector, backed by the ``bm25s`` library.

Canonical concrete for the ``retrieve`` family: zero-download (no model, no
network), deterministic, and contract-canonical (bm25s ``retrieve`` returns
ranked indices with scores directly). MIT-licensed, numpy-only.
"""

from __future__ import annotations

from typing import List, Tuple

from mloda.provider import property_spec

from rag_integration.feature_groups.connectors.retrieve.base import BaseRetrieveConnector


class Bm25sRetriever(BaseRetrieveConnector):
    """Lexical BM25 retrieval over an inline corpus.

    Selected with ``retrieve_backend="bm25s"``. Builds an in-memory BM25 index
    per call (the corpus is per-call), so there is no shared state to cache and
    repeated calls are idempotent. Family rule: at most ``top_k`` passages come
    back and only those scoring positively, so a degenerate query (empty, or
    all out-of-vocabulary) yields no passages.
    """

    RETRIEVE_BACKENDS = {
        "bm25s": "BM25 lexical retrieval (bm25s)",
    }

    # Declarative option documentation; selection is via
    # ``match_feature_group_criteria`` (see BaseRetrieveConnector). The allowed
    # backend value is the single key of RETRIEVE_BACKENDS above.
    PROPERTY_MAPPING = {
        BaseRetrieveConnector.RETRIEVE_BACKEND: property_spec("Use 'bm25s' for BM25 lexical retrieval", context=False),
        BaseRetrieveConnector.QUERY_TEXT: property_spec("Raw text query to search the corpus", context=False),
        BaseRetrieveConnector.TOP_K: property_spec(
            f"Number of passages to return (default {BaseRetrieveConnector.DEFAULT_TOP_K})", context=False
        ),
        BaseRetrieveConnector.CORPUS: property_spec("Inline corpus: a list of {doc_id, text} dicts", context=False),
    }

    @classmethod
    def _rank(cls, query: str, texts: List[str], top_k: int) -> List[Tuple[int, float]]:
        import bm25s

        corpus_tokens = bm25s.tokenize(texts, stopwords="en", show_progress=False)
        # Degenerate corpus (e.g. every doc is only stopwords) tokenizes to an
        # empty vocabulary; bm25s would raise on retrieve. Nothing is rankable.
        if len(corpus_tokens.vocab) == 0:
            return []

        retriever = bm25s.BM25()
        retriever.index(corpus_tokens, show_progress=False)

        query_tokens = bm25s.tokenize([query], stopwords="en", show_progress=False)
        # index() is called WITHOUT a corpus, so retrieve() returns integer
        # corpus indices (not document objects); keep it that way so the int()
        # cast below stays valid.
        indices, scores = retriever.retrieve(query_tokens, k=top_k, show_progress=False)

        pairs = [(int(indices[0][rank]), float(scores[0][rank])) for rank in range(top_k)]
        # Family rule: only positively scoring passages are returned. Without
        # this filter a degenerate query (empty, or all out-of-vocabulary)
        # would pad the result with top_k zero-scored passages in arbitrary
        # order instead of returning nothing.
        return [(idx, score) for idx, score in pairs if score > 0.0]
