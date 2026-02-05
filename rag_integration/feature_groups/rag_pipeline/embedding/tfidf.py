"""TF-IDF based embedding."""

from __future__ import annotations

import math
from collections import Counter
from typing import Dict, List, Set

from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys

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

    PROPERTY_MAPPING = {
        BaseEmbedder.EMBEDDING_METHOD: {
            "tfidf": "TF-IDF based embeddings",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: True,
        },
        BaseEmbedder.EMBEDDING_DIM: {
            "explanation": "Dimension of the embedding vectors",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.default: 384,
        },
        BaseEmbedder.MODEL_NAME: {
            "explanation": "Name of the embedding model",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.default: "default",
        },
        DefaultOptionKeys.in_features: {
            "explanation": "Source feature containing text to embed",
            DefaultOptionKeys.context: True,
        },
    }

    @classmethod
    def _tokenize(cls, text: str) -> List[str]:
        """Simple tokenization."""
        # Lowercase and split on non-alphanumeric
        import re

        words = re.findall(r"\b\w+\b", text.lower())
        # Filter short words
        return [w for w in words if len(w) > 2]

    @classmethod
    def _compute_tf(cls, tokens: List[str]) -> Dict[str, float]:
        """Compute term frequency."""
        counter = Counter(tokens)
        total = len(tokens) if tokens else 1
        return {word: count / total for word, count in counter.items()}

    @classmethod
    def _compute_idf(cls, documents: List[List[str]], vocab: Set[str]) -> Dict[str, float]:
        """Compute inverse document frequency."""
        n_docs = len(documents)
        idf = {}

        for word in vocab:
            doc_count = sum(1 for doc in documents if word in doc)
            idf[word] = math.log((n_docs + 1) / (doc_count + 1)) + 1

        return idf

    @classmethod
    def _embed_texts(
        cls,
        texts: List[str],
        embedding_dim: int,
        model_name: str,
    ) -> List[List[float]]:
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
        vocab: Set[str] = set()
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
    def _tfidf_embed(cls, tokens: List[str], idf: Dict[str, float], dim: int) -> List[float]:
        """
        Generate TF-IDF embedding for a single document.

        Args:
            tokens: Tokenized document
            idf: IDF scores for vocabulary
            dim: Embedding dimension

        Returns:
            Normalized TF-IDF vector
        """
        import hashlib

        embedding = [0.0] * dim

        if not tokens:
            return embedding

        # Compute TF
        tf = cls._compute_tf(tokens)

        # Compute TF-IDF and hash to position
        for word, tf_score in tf.items():
            tfidf_score = tf_score * idf.get(word, 1.0)

            # Hash word to position
            word_hash = int(hashlib.md5(word.encode("utf-8"), usedforsecurity=False).hexdigest(), 16)
            position = word_hash % dim

            embedding[position] += tfidf_score

        # Normalize
        magnitude = math.sqrt(sum(x * x for x in embedding))
        if magnitude > 0:
            embedding = [x / magnitude for x in embedding]

        return embedding
