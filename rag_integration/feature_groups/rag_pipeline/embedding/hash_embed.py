"""Deterministic hash-based embedding."""

from __future__ import annotations

import hashlib
import math
from typing import List

from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys

from rag_integration.feature_groups.rag_pipeline.embedding.base import BaseEmbedder


class HashEmbedder(BaseEmbedder):
    """
    Deterministic hash-based embedder.

    Generates consistent embeddings using feature hashing.
    Same text always produces the same embedding.

    Useful for reproducible pipelines and testing.
    Does not capture semantic similarity.

    Config-based matching:
        embedding_method="hash"
    """

    PROPERTY_MAPPING = {
        BaseEmbedder.EMBEDDING_METHOD: {
            "hash": "Feature hashing based embeddings",
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
    def _embed_texts(
        cls,
        texts: List[str],
        embedding_dim: int,
        model_name: str,
    ) -> List[List[float]]:
        """
        Generate hash-based embeddings.

        Uses feature hashing (hashing trick) to create embeddings.
        Words are hashed to positions in the embedding vector.

        Args:
            texts: List of text strings to embed
            embedding_dim: Dimension of output vectors
            model_name: Ignored for hash embedder

        Returns:
            List of deterministic embedding vectors
        """
        embeddings = []

        for text in texts:
            embedding = cls._hash_embed(text, embedding_dim)
            embeddings.append(embedding)

        return embeddings

    @classmethod
    def _hash_embed(cls, text: str, dim: int) -> List[float]:
        """
        Generate a hash-based embedding for text.

        Uses the hashing trick: each word hashes to a position,
        and the sign is determined by a second hash.

        Args:
            text: Text to embed
            dim: Dimension of the embedding

        Returns:
            Normalized embedding vector
        """
        embedding = [0.0] * dim

        # Tokenize (simple whitespace split)
        words = text.lower().split()

        for word in words:
            # Hash to get position
            pos_hash = int(hashlib.md5(word.encode("utf-8"), usedforsecurity=False).hexdigest(), 16)
            position = pos_hash % dim

            # Hash to get sign
            sign_hash = int(hashlib.sha256(word.encode("utf-8")).hexdigest(), 16)
            sign = 1 if sign_hash % 2 == 0 else -1

            embedding[position] += sign

        # Normalize to unit length
        magnitude = math.sqrt(sum(x * x for x in embedding))
        if magnitude > 0:
            embedding = [x / magnitude for x in embedding]

        return embedding
