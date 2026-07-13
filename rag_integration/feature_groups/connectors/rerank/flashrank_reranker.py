"""FlashRank cross-encoder reranker.

Pedigree concrete for the ``rerank`` family: a real neural cross-encoder that
exercises the distinguishing semantics of reranking, kept light by FlashRank's
ONNX runtime (no torch). Behind the ``rerank`` extra. The model (~4 MB for the
default ``ms-marco-TinyBERT-L-2-v2``) downloads on first use and is cached, so
its contract test is skipped on CI (network) but runs locally; the zero-download
``LexicalReranker`` is the always-on CI anchor.
"""

from __future__ import annotations

import threading
from typing import Any, List, Tuple

from mloda.provider import property_spec

from rag_integration.feature_groups.connectors.rerank.base import BaseRerankConnector


class FlashRankReranker(BaseRerankConnector):
    """Cross-encoder reranking via FlashRank (``rerank_backend="flashrank"``).

    The default model is ``ms-marco-TinyBERT-L-2-v2`` (~4 MB, Apache-2.0). The
    ranker is cached at class level since constructing it loads the ONNX model;
    loading is guarded by a lock so concurrent callers do not build it twice.
    """

    # Fixed to the small default model. A configurable model option is omitted
    # deliberately: the base `_rank(query, texts, top_k)` contract passes no
    # options, so advertising a model option here would be a surface lie. If a
    # future need arises, plumb it through `calculate_feature`, not here.
    DEFAULT_MODEL = "ms-marco-TinyBERT-L-2-v2"

    RERANK_BACKENDS = {
        "flashrank": "Cross-encoder reranking (FlashRank, ONNX)",
    }

    PROPERTY_MAPPING = {
        BaseRerankConnector.RERANK_BACKEND: property_spec("Use 'flashrank' for cross-encoder reranking", context=False),
        BaseRerankConnector.QUERY_TEXT: property_spec("Query the candidates are reranked against", context=False),
        BaseRerankConnector.TOP_K: property_spec(
            f"Number of passages to return after reranking (default {BaseRerankConnector.DEFAULT_TOP_K})", context=False
        ),
        BaseRerankConnector.CANDIDATES: property_spec(
            "Candidate passages: a list of {doc_id, text} dicts", context=False
        ),
    }

    # Single-slot cache: only DEFAULT_MODEL is ever loaded.
    _ranker: Any | None = None
    _cache_lock = threading.Lock()

    @classmethod
    def _get_ranker(cls) -> Any:
        from flashrank import Ranker

        ranker = cls._ranker
        if ranker is not None:
            return ranker
        with cls._cache_lock:
            if cls._ranker is None:
                cls._ranker = Ranker(model_name=cls.DEFAULT_MODEL)
            return cls._ranker

    @classmethod
    def _rank(cls, query: str, texts: List[str], top_k: int) -> List[Tuple[int, float]]:
        from flashrank import RerankRequest

        ranker = cls._get_ranker()
        # Use the candidate's list index as the passage id so results map back
        # to positions regardless of how FlashRank reorders them.
        passages = [{"id": str(idx), "text": text} for idx, text in enumerate(texts)]
        ranked = ranker.rerank(RerankRequest(query=query, passages=passages))
        return [(int(item["id"]), float(item["score"])) for item in ranked[:top_k]]
