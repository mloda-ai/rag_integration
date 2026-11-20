"""TF-IDF based embedding."""

from __future__ import annotations

import math
from collections import Counter
from typing import ClassVar

from mloda.provider import DefaultOptionKeys, property_spec

from rag_integration.feature_groups.rag_pipeline.embedding.base import BaseEmbedder


class TfidfEmbedder(BaseEmbedder):
    """
    TF-IDF based embedder.

    Creates embeddings using Term Frequency-Inverse Document Frequency.
    Captures term importance relative to the corpus.

    Simple semantic representation without neural networks.

    Config-based matching:
        embedding_method="tfidf"
    """

    PROPERTY_MAPPING: ClassVar = {
        BaseEmbedder.EMBEDDING_METHOD: property_spec(
            "Algorithm used to embed text into vectors",
            strict=True,
            allowed_values={"tfidf": "TF-IDF based embeddings"},
            deferred_binding=True,
        ),
        BaseEmbedder.EMBEDDING_DIM: property_spec("Dimension of the embedding vectors", default=384),
        BaseEmbedder.MODEL_NAME: property_spec("Name of the embedding model", default="default"),
        DefaultOptionKeys.in_features: property_spec("Source feature containing text to embed"),
    }

    @classmethod
    def _tokenize(cls, text: str) -> list[str]:
        """Simple tokenization."""
        # Lowercase and split on non-alphanumeric
        import re

        words = re.findall(r"\b\w+\b", text.lower())
        # Filter short words
        return [w for w in words if len(w) > 2]

    @classmethod
    def _compute_tf(cls, tokens: list[str]) -> dict[str, float]:
        """Compute term frequency."""
        counter = Counter(tokens)
        total = len(tokens) if tokens else 1
        return {word: count / total for word, count in counter.items()}

    @classmethod
    def _compute_idf(cls, documents: list[list[str]], vocab: set[str]) -> dict[str, float]:
        """Compute inverse document frequency."""
        n_docs = len(documents)
        idf = {}

        for word in vocab:
            doc_count = sum(1 for doc in documents if word in doc)
            idf[word] = cls._idf(n_docs, doc_count)

        return idf

    @staticmethod
    def _idf(n_docs: int, doc_count: int) -> float:
        """Smoothed IDF of a term found in ``doc_count`` of ``n_docs`` documents."""
        return math.log((n_docs + 1) / (doc_count + 1)) + 1

    @classmethod
    def _embed_texts(
        cls,
        texts: list[str],
        embedding_dim: int,
        model_name: str,
    ) -> list[list[float]]:
        """
        Generate TF-IDF embeddings.

        Computes TF-IDF vectors and reduces to specified dimension
        using hashing to map words to dimensions.

        Args:
            texts: List of text strings to embed
            embedding_dim: Dimension of output vectors
            model_name: Ignored for TF-IDF embedder

        Returns:
            List of TF-IDF embedding vectors
        """
        # Tokenize all documents
        tokenized = [cls._tokenize(text) for text in texts]

        # Build vocabulary
        vocab: set[str] = set()
        for tokens in tokenized:
            vocab.update(tokens)

        # Compute IDF
        idf = cls._compute_idf(tokenized, vocab)

        # Generate embeddings
        embeddings = []
        for tokens in tokenized:
            embedding = cls._tfidf_embed(tokens, idf, embedding_dim)
            embeddings.append(embedding)

        return embeddings

    @classmethod
    def _tfidf_embed(cls, tokens: list[str], idf: dict[str, float], dim: int) -> list[float]:
        """
        Generate TF-IDF embedding for a single document.

        Args:
            tokens: Tokenized document
            idf: IDF scores for vocabulary
            dim: Embedding dimension

        Returns:
            Normalized TF-IDF vector
        """
        return cls._weigh(cls._hashed_tf(tokens, dim), idf, dim)

    @classmethod
    def _hashed_tf(cls, tokens: list[str], dim: int) -> tuple[tuple[str, float, int], ...]:
        """Each term's ``(word, tf, hashed position)``, in first-occurrence order (query-independent)."""
        import hashlib

        return tuple(
            (word, tf_score, int(hashlib.md5(word.encode("utf-8"), usedforsecurity=False).hexdigest(), 16) % dim)
            for word, tf_score in cls._compute_tf(tokens).items()
        )

    @staticmethod
    def _weigh(terms: tuple[tuple[str, float, int], ...], idf: dict[str, float], dim: int) -> list[float]:
        """The normalized TF-IDF vector of ``_hashed_tf`` terms."""
        embedding = [0.0] * dim
        for word, tf_score, position in terms:
            embedding[position] += tf_score * idf.get(word, 1.0)

        # Normalize
        magnitude = math.sqrt(sum(x * x for x in embedding))
        if magnitude > 0:
            embedding = [x / magnitude for x in embedding]

        return embedding
