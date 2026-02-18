"""Deterministic hash-based image embedding."""

from __future__ import annotations

import hashlib
import math
from typing import List

from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys

from rag_integration.feature_groups.image_pipeline.embedding.base import BaseImageEmbedder


class HashImageEmbedder(BaseImageEmbedder):
    """
    Deterministic hash-based image embedder.

    Generates consistent embeddings using feature hashing on image pixel data.
    Converts image to grayscale pixels and hashes pixel blocks to positions.

    Same image always produces the same embedding.
    Does not capture visual semantics.

    Requires Pillow.

    Config-based matching:
        image_embedding_method="hash"
    """

    PROPERTY_MAPPING = {
        BaseImageEmbedder.IMAGE_EMBEDDING_METHOD: {
            "hash": "Feature hashing based image embeddings",
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
        Generate hash-based embedding from image pixel data.

        Args:
            image_data: Raw image bytes
            embedding_dim: Dimension of output vector
            model_name: Ignored for hash embedder

        Returns:
            Normalized embedding vector
        """
        try:
            from PIL import Image
        except ImportError:
            raise ImportError("Pillow is required for HashImageEmbedder. Install with: pip install Pillow")

        import io

        if not image_data:
            # Return zero vector for empty images
            return [0.0] * embedding_dim

        img = Image.open(io.BytesIO(image_data))
        # Convert to grayscale and resize to manageable size
        img = img.convert("L").resize((64, 64), Image.LANCZOS)
        pixels = list(img.get_flattened_data())

        embedding = [0.0] * embedding_dim

        # Hash pixel blocks to positions
        block_size = max(1, len(pixels) // embedding_dim)
        for i in range(0, len(pixels), block_size):
            block = pixels[i : i + block_size]
            block_bytes = bytes(block)

            pos_hash = int(hashlib.md5(block_bytes, usedforsecurity=False).hexdigest(), 16)
            position = pos_hash % embedding_dim

            sign_hash = int(hashlib.sha256(block_bytes).hexdigest(), 16)
            sign = 1 if sign_hash % 2 == 0 else -1

            embedding[position] += sign * (sum(block) / len(block) / 255.0)

        # Normalize to unit length
        magnitude = math.sqrt(sum(x * x for x in embedding))
        if magnitude > 0:
            embedding = [x / magnitude for x in embedding]

        return embedding
