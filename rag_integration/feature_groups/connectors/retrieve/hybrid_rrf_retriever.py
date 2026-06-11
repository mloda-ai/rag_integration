"""Hybrid lexical + dense retrieve connector fused with RRF (issue #46).

The decision recorded here: fusion *mechanics* are cross-cutting and live in
``connectors/fusion.py``; family *exposure* is this thin backend, because
within one family fusion still serves the unchanged ``retrieve`` contract.
Cross-family blending (e.g. ``retrieve`` + ``graph_rag``) reuses
:func:`rrf_fuse` on ``doc_id``-keyed rankings instead of growing a new backend.
Requires the ``faiss`` extra (the dense component).
"""

from __future__ import annotations

from typing import List, Tuple

from rag_integration.feature_groups.connectors.fusion import rrf_fuse
from rag_integration.feature_groups.connectors.retrieve.base import BaseRetrieveConnector
from rag_integration.feature_groups.connectors.retrieve.bm25s_retriever import Bm25sRetriever
from rag_integration.feature_groups.connectors.retrieve.faiss_retriever import FaissDenseRetriever


class HybridRrfRetriever(BaseRetrieveConnector):
    """Hybrid retrieval over an inline corpus (``retrieve_backend="hybrid_rrf"``).

    Ranks the corpus twice, lexically (``bm25s``) and densely (``faiss``), and
    fuses the two rankings with reciprocal-rank fusion, so a doc both
    components agree on outranks a doc only one champions. Each component
    contributes its full positive-scoring ranking; the fused score is the RRF
    sum, positive by construction, so a query degenerate for both components
    yields no passages and the family rules hold unchanged.
    """

    _COMPONENTS: Tuple[type[BaseRetrieveConnector], ...] = (Bm25sRetriever, FaissDenseRetriever)

    RETRIEVE_BACKENDS = {
        "hybrid_rrf": "Hybrid lexical (bm25s) + dense (faiss) retrieval fused with reciprocal-rank fusion",
    }

    PROPERTY_MAPPING = {
        BaseRetrieveConnector.RETRIEVE_BACKEND: {"explanation": "Use 'hybrid_rrf' for RRF-fused lexical + dense"},
        BaseRetrieveConnector.QUERY_TEXT: {"explanation": "Raw text query to search the corpus"},
        BaseRetrieveConnector.TOP_K: {
            "explanation": f"Number of passages to return (default {BaseRetrieveConnector.DEFAULT_TOP_K})"
        },
        BaseRetrieveConnector.CORPUS: {"explanation": "Inline corpus: a list of {doc_id, text} dicts"},
    }

    @classmethod
    def _rank(cls, query: str, texts: List[str], top_k: int) -> List[Tuple[int, float]]:
        # Each component ranks the whole corpus (not just top_k) so fusion sees
        # full rankings; components already return only positive-scoring pairs.
        rankings = [[idx for idx, _ in component._rank(query, texts, len(texts))] for component in cls._COMPONENTS]
        return rrf_fuse(rankings, top_k)
