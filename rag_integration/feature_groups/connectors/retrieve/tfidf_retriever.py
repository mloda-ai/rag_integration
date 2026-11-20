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

from collections import Counter
from typing import Any, ClassVar

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

    RETRIEVE_BACKENDS: ClassVar = {
        "tfidf": "Vector-space TF-IDF retrieval (cosine over hashed TF-IDF vectors)",
    }

    PROPERTY_MAPPING: ClassVar = {
        BaseRetrieveConnector.RETRIEVE_BACKEND: property_spec(
            "Use 'tfidf' for vector-space TF-IDF retrieval", context=False
        ),
        BaseRetrieveConnector.QUERY_TEXT: BaseRetrieveConnector.QUERY_TEXT_SPEC,
        BaseRetrieveConnector.TOP_K: BaseRetrieveConnector.TOP_K_SPEC,
        BaseRetrieveConnector.CORPUS: property_spec(
            "Inline corpus: a list of {doc_id, text} dicts", context=False, default=None
        ),
        BaseRetrieveConnector.CORPUS_SOURCE: BaseRetrieveConnector.CORPUS_SOURCE_SPEC,
    }

    @staticmethod
    def _cosine(query_vector: list[float], doc_vector: list[float]) -> float:
        # Both vectors are L2-normalized by the embedder, so the dot product is
        # already the cosine similarity.
        return sum(q * d for q, d in zip(query_vector, doc_vector))

    @classmethod
    def _build_index(cls, texts: tuple[str, ...]) -> Any:
        """Each document's hashed term frequencies and each term's document frequency."""
        docs = tuple(TfidfEmbedder._hashed_tf(TfidfEmbedder._tokenize(text), cls._TFIDF_DIM) for text in texts)
        return docs, Counter(word for terms in docs for word, _, _ in terms)

    @classmethod
    def _rank(cls, query: str, texts: list[str], top_k: int) -> list[tuple[int, float]]:
        # IDF spans the corpus plus the query (one shared space), so each query re-weighs every
        # document; tokenizing, term frequencies and hashing are reused.
        docs, doc_freq = cls._index(texts)
        query_terms = TfidfEmbedder._hashed_tf(TfidfEmbedder._tokenize(query), cls._TFIDF_DIM)
        query_words = {word for word, _, _ in query_terms}
        n_docs = len(docs) + 1
        idf = {
            word: TfidfEmbedder._idf(n_docs, doc_freq.get(word, 0) + (word in query_words))
            for word in doc_freq.keys() | query_words
        }
        query_vector = TfidfEmbedder._weigh(query_terms, idf, cls._TFIDF_DIM)
        doc_vectors = [TfidfEmbedder._weigh(terms, idf, cls._TFIDF_DIM) for terms in docs]

        scored = [(idx, cls._cosine(query_vector, doc_vector)) for idx, doc_vector in enumerate(doc_vectors)]
        # Family rule: only positively scoring passages are returned. This also
        # covers the degenerate query (empty, or only tokens the embedder
        # drops): it embeds to an all-zero vector, every cosine is 0, and no
        # pair survives the filter.
        positive = [(idx, score) for idx, score in scored if score > 0.0]
        # Best score first; ties broken by original index for a stable order.
        positive.sort(key=lambda pair: (-pair[1], pair[0]))
        return positive[:top_k]
