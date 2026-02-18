"""CLIP model image embedding."""

from __future__ import annotations

import math
from typing import List

from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys

from rag_integration.feature_groups.image_pipeline.embedding.base import BaseImageEmbedder


class CLIPImageEmbedder(BaseImageEmbedder):
    """
    CLIP model image embedder.

    Uses OpenAI's CLIP (Contrastive Language-Image Pre-Training) model
    to generate semantically meaningful image embeddings that exist in
    a shared text-image embedding space.

    Requires: transformers, torch, Pillow

    Config-based matching:
        image_embedding_method="clip"
    """

    PROPERTY_MAPPING = {
        BaseImageEmbedder.IMAGE_EMBEDDING_METHOD: {
            "clip": "CLIP model image embeddings",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: True,
        },
        BaseImageEmbedder.EMBEDDING_DIM: {
            "explanation": "Dimension of the embedding vectors (CLIP default: 512)",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.default: 512,
        },
        BaseImageEmbedder.MODEL_NAME: {
            "explanation": "CLIP model name (default: openai/clip-vit-base-patch32)",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.default: "openai/clip-vit-base-patch32",
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
        Generate CLIP embedding for an image.

        Args:
            image_data: Raw image bytes
            embedding_dim: Dimension of output vector (used for validation)
            model_name: CLIP model identifier

        Returns:
            CLIP embedding vector, normalized to unit length
        """
        try:
            from transformers import CLIPProcessor, CLIPModel
            from PIL import Image
            import torch
        except ImportError:
            raise ImportError(
                "transformers, torch, and Pillow are required for CLIPImageEmbedder. "
                "Install with: pip install transformers torch Pillow"
            )

        import io

        if not image_data:
            return [0.0] * embedding_dim

        img = Image.open(io.BytesIO(image_data)).convert("RGB")

        model = CLIPModel.from_pretrained(model_name)
        processor = CLIPProcessor.from_pretrained(model_name)

        inputs = processor(images=img, return_tensors="pt")

        with torch.no_grad():
            outputs = model.get_image_features(**inputs)

        embedding = outputs[0].tolist()

        # Normalize to unit length
        magnitude = math.sqrt(sum(x * x for x in embedding))
        if magnitude > 0:
            embedding = [x / magnitude for x in embedding]

        return embedding
