"""Dense FAISS retrieve connector (canonical dense backend, issue #36).

The stage pipeline's FAISS search folded in behind the family's inline-corpus
contract: embeds corpus and query per call with the deterministic
``HashEmbedder`` and searches an in-memory index. Requires the ``faiss`` extra.
Row-shape parity with the stage is pinned by
``tests/integration/test_stage_connector_parity.py``.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from rag_integration.feature_groups.connectors.retrieve.base import BaseRetrieveConnector
from rag_integration.feature_groups.rag_pipeline.embedding.hash_embed import HashEmbedder


class FaissDenseRetriever(BaseRetrieveConnector):
    """Dense FAISS retrieval over an inline corpus (``retrieve_backend="faiss"``).

    Ranks by inner product over unit-normalized hash embeddings, so scores are
    cosine similarities. Per-call index, no shared state, idempotent. Family
    rule: at most ``top_k`` passages, only positive scores, so a degenerate
    query yields no passages.
    """

    _EMBED_DIM = 384  # the hash embedder's own default width

    RETRIEVE_BACKENDS = {
        "faiss": "Dense FAISS retrieval (cosine over deterministic hash embeddings)",
    }

    PROPERTY_MAPPING = {
        BaseRetrieveConnector.RETRIEVE_BACKEND: {"explanation": "Use 'faiss' for dense FAISS retrieval"},
        BaseRetrieveConnector.QUERY_TEXT: {"explanation": "Raw text query to search the corpus"},
        BaseRetrieveConnector.TOP_K: {
            "explanation": f"Number of passages to return (default {BaseRetrieveConnector.DEFAULT_TOP_K})"
        },
        BaseRetrieveConnector.CORPUS: {"explanation": "Inline corpus: a list of {doc_id, text} dicts"},
    }

    @classmethod
    def _rank(cls, query: str, texts: List[str], top_k: int) -> List[Tuple[int, float]]:
        import faiss

        vectors = HashEmbedder._embed_texts(list(texts) + [query], cls._EMBED_DIM, "default")
        corpus_array = np.array(vectors[:-1], dtype=np.float32)
        query_array = np.array([vectors[-1]], dtype=np.float32)

        # Unit-length vectors: inner product == cosine, returned best-first.
        index = faiss.IndexFlatIP(cls._EMBED_DIM)
        index.add(corpus_array)
        scores, indices = index.search(query_array, top_k)

        pairs = [(int(idx), float(score)) for idx, score in zip(indices[0], scores[0])]
        # Family rule: only positive scores; a degenerate query yields no pairs.
        return [(idx, score) for idx, score in pairs if score > 0.0]
