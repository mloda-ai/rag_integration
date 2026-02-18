"""Mock image embedding for testing."""

from __future__ import annotations

import hashlib
import math
from typing import List

from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys

from rag_integration.feature_groups.image_pipeline.embedding.base import BaseImageEmbedder


class MockImageEmbedder(BaseImageEmbedder):
    """
    Mock image embedder for testing and development.

    Generates deterministic pseudo-random vectors based on image data hash.
    Useful for testing pipeline integration without real embedding models.

    The embeddings are deterministic: same image always produces same vector.

    Config-based matching:
        image_embedding_method="mock"
    """

    PROPERTY_MAPPING = {
        BaseImageEmbedder.IMAGE_EMBEDDING_METHOD: {
            "mock": "Deterministic mock image embeddings for testing",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: True,
        },
        BaseImageEmbedder.EMBEDDING_DIM: {
            "explanation": "Dimension of the embedding vectors",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.default: 512,
        },
        BaseImageEmbedder.MODEL_NAME: {
            "explanation": "Name of the embedding model",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.default: "default",
        },
        DefaultOptionKeys.in_features: {
            "explanation": "Source feature containing images to embed",
            DefaultOptionKeys.context: True,
        },
    }

    @classmethod
    def _embed_image(
        cls,
        image_data: bytes,
        embedding_dim: int,
        model_name: str,
    ) -> List[float]:
        """
        Generate mock embedding based on image data hash.

        Args:
            image_data: Raw image bytes
            embedding_dim: Dimension of output vector
            model_name: Ignored for mock embedder

        Returns:
            Deterministic pseudo-random vector, normalized to unit length
        """
        return cls._generate_mock_embedding(image_data, embedding_dim)

    @classmethod
    def _generate_mock_embedding(cls, image_data: bytes, dim: int) -> List[float]:
        """
        Generate a deterministic mock embedding for image data.

        Uses the image data hash as a seed to generate reproducible values.
        The embedding is normalized to unit length.

        Args:
            image_data: Image bytes
            dim: Dimension of the embedding

        Returns:
            List of floats (normalized to unit length)
        """
        data_hash = hashlib.sha256(image_data).hexdigest()

        embedding = []
        hash_index = 0

        for i in range(dim):
            if hash_index + 8 > len(data_hash):
                data_hash += hashlib.sha256((data_hash + str(hash_index)).encode("utf-8")).hexdigest()

            hex_chunk = data_hash[hash_index : hash_index + 8]
            hash_index += 8

            int_value = int(hex_chunk, 16)
            float_value = (int_value / 0xFFFFFFFF) * 2 - 1
            embedding.append(float_value)

        # Normalize to unit length
        magnitude = math.sqrt(sum(x * x for x in embedding))
        if magnitude > 0:
            embedding = [x / magnitude for x in embedding]

        return embedding
