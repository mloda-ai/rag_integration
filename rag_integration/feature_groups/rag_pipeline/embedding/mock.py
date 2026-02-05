"""Mock embedding for testing."""

from __future__ import annotations

import hashlib
import math
from typing import List

from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys

from rag_integration.feature_groups.rag_pipeline.embedding.base import BaseEmbedder


class MockEmbedder(BaseEmbedder):
    """
    Mock embedder for testing and development.

    Generates deterministic pseudo-random vectors based on text hash.
    Useful for testing pipeline integration without real embedding models.

    The embeddings are deterministic: same text always produces same vector.

    Config-based matching:
        embedding_method="mock"
    """

    PROPERTY_MAPPING = {
        BaseEmbedder.EMBEDDING_METHOD: {
            "mock": "Deterministic mock embeddings for testing",
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
        Generate mock embeddings based on text hash.

        Args:
            texts: List of text strings to embed
            embedding_dim: Dimension of output vectors
            model_name: Ignored for mock embedder

        Returns:
            List of deterministic pseudo-random vectors
        """
        embeddings = []

        for text in texts:
            embedding = cls._generate_mock_embedding(text, embedding_dim)
            embeddings.append(embedding)

        return embeddings

    @classmethod
    def _generate_mock_embedding(cls, text: str, dim: int) -> List[float]:
        """
        Generate a deterministic mock embedding for a text.

        Uses the text hash as a seed to generate reproducible values.
        The embedding is normalized to unit length.

        Args:
            text: Text to embed
            dim: Dimension of the embedding

        Returns:
            List of floats (normalized to unit length)
        """
        # Use hash as seed for reproducibility
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

        # Generate values from hash chunks
        embedding = []
        hash_index = 0

        for i in range(dim):
            # Get 8 hex chars (4 bytes) for each dimension
            if hash_index + 8 > len(text_hash):
                # Extend hash if needed
                text_hash += hashlib.sha256((text + str(hash_index)).encode("utf-8")).hexdigest()

            hex_chunk = text_hash[hash_index : hash_index + 8]
            hash_index += 8

            # Convert to float in range [-1, 1]
            int_value = int(hex_chunk, 16)
            float_value = (int_value / 0xFFFFFFFF) * 2 - 1
            embedding.append(float_value)

        # Normalize to unit length
        magnitude = math.sqrt(sum(x * x for x in embedding))
        if magnitude > 0:
            embedding = [x / magnitude for x in embedding]

        return embedding
