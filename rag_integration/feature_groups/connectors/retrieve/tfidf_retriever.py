"""Vector-space TF-IDF retrieve connector (lexical).

Second concrete for the ``retrieve`` family: vectorizes the corpus and query
with the repo's deterministic :class:`TfidfEmbedder` (hashed TF-IDF, still a
lexical representation, not a learned dense one) and ranks documents by cosine
similarity. Zero-download (no model, no network), pure-Python, deterministic. A
vector-space counterpart to the probabilistic ``bm25s`` backend that anchors
the same contract suite from a different ranking mechanism, and with no new
dependency (it reuses the existing TF-IDF embedder).
"""

from __future__ import annotations

from typing import List, Tuple

from mloda.provider import property_spec

from rag_integration.feature_groups.connectors.retrieve.base import BaseRetrieveConnector
from rag_integration.feature_groups.rag_pipeline.embedding.tfidf import TfidfEmbedder


class TfidfRetriever(BaseRetrieveConnector):
    """Vector-space TF-IDF retrieval (lexical) over an inline corpus
    (``retrieve_backend="tfidf"``).

    Vectorizes the corpus and the query together so they share one
    IDF/vocabulary, then ranks documents by cosine similarity to the query. The
    embedder L2-normalizes every vector, so cosine reduces to a dot product.
    Ties are broken by corpus index, so the ordering is stable and
    deterministic. Family rule: at most ``top_k`` passages come back and only
    those scoring positively, so a degenerate query yields no passages.
    """

    # The embedder hashes terms into a fixed-width vector; 384 is its own default
    # and is ample for the small inline corpora this family serves. ``model_name``
    # is ignored by the TF-IDF embedder, so the default is passed verbatim.
    _TFIDF_DIM = 384

    RETRIEVE_BACKENDS = {
        "tfidf": "Vector-space TF-IDF retrieval (cosine over hashed TF-IDF vectors)",
    }

    PROPERTY_MAPPING = {
        BaseRetrieveConnector.RETRIEVE_BACKEND: property_spec(
            "Use 'tfidf' for vector-space TF-IDF retrieval", context=False
        ),
        BaseRetrieveConnector.QUERY_TEXT: property_spec("Raw text query to search the corpus", context=False),
        BaseRetrieveConnector.TOP_K: property_spec(
            f"Number of passages to return (default {BaseRetrieveConnector.DEFAULT_TOP_K})", context=False
        ),
        BaseRetrieveConnector.CORPUS: property_spec("Inline corpus: a list of {doc_id, text} dicts", context=False),
    }

    @staticmethod
    def _cosine(query_vector: List[float], doc_vector: List[float]) -> float:
        # Both vectors are L2-normalized by the embedder, so the dot product is
        # already the cosine similarity.
        return sum(q * d for q, d in zip(query_vector, doc_vector))

    @classmethod
    def _rank(cls, query: str, texts: List[str], top_k: int) -> List[Tuple[int, float]]:
        # ``_embed_texts`` is the embedder's deterministic raw-text vectorization
        # entry point; embedding the corpus and query in one batch shares a
        # single IDF/vocabulary so the query and documents live in one space.
        vectors = TfidfEmbedder._embed_texts(list(texts) + [query], cls._TFIDF_DIM, "default")
        query_vector = vectors[-1]
        doc_vectors = vectors[:-1]

        scored = [(idx, cls._cosine(query_vector, doc_vector)) for idx, doc_vector in enumerate(doc_vectors)]
        # Family rule: only positively scoring passages are returned. This also
        # covers the degenerate query (empty, or only tokens the embedder
        # drops): it embeds to an all-zero vector, every cosine is 0, and no
        # pair survives the filter.
        positive = [(idx, score) for idx, score in scored if score > 0.0]
        # Best score first; ties broken by original index for a stable order.
        positive.sort(key=lambda pair: (-pair[1], pair[0]))
        return positive[:top_k]
