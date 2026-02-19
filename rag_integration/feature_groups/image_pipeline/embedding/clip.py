"""CLIP model image embedding."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, List

from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys

from rag_integration.feature_groups.image_pipeline.embedding.base import BaseImageEmbedder

# Default local model path (saved via save_pretrained)
_DEFAULT_LOCAL_MODEL = str(Path(__file__).resolve().parents[4] / "models" / "clip-vit-base-patch32")
# HuggingFace fallback
_HF_MODEL_ID = "openai/clip-vit-base-patch32"


class CLIPImageEmbedder(BaseImageEmbedder):
    """
    CLIP model image embedder.

    Uses OpenAI's CLIP (Contrastive Language-Image Pre-Training) model
    to generate semantically meaningful image embeddings that exist in
    a shared text-image embedding space.

    Loads from a local directory if available, otherwise downloads from HuggingFace.

    Requires: transformers, torch, Pillow

    Config-based matching:
        image_embedding_method="clip"
    """

    _model_cache: dict[str, Any] = {}

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
            "explanation": "Local path or HuggingFace model ID",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.default: _HF_MODEL_ID,
        },
        DefaultOptionKeys.in_features: {
            "explanation": "Source feature containing images to embed",
            DefaultOptionKeys.context: True,
        },
    }

    @classmethod
    def _resolve_model_path(cls, model_name: str) -> str:
        """Resolve model name: use local path if it exists, otherwise use as-is (HuggingFace ID)."""
        if os.path.isdir(model_name):
            return model_name
        if os.path.isdir(_DEFAULT_LOCAL_MODEL):
            return _DEFAULT_LOCAL_MODEL
        return model_name

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

        resolved = cls._resolve_model_path(model_name)
        if resolved not in cls._model_cache:
            cls._model_cache[resolved] = (
                CLIPModel.from_pretrained(resolved),  # nosec B615
                CLIPProcessor.from_pretrained(resolved),  # nosec B615
            )
        model, processor = cls._model_cache[resolved]

        inputs = processor(images=img, return_tensors="pt")

        with torch.no_grad():
            vision_outputs = model.vision_model(**{k: v for k, v in inputs.items() if k == "pixel_values"})
            image_features = model.visual_projection(vision_outputs.pooler_output)

        embedding = image_features.squeeze(0).tolist()

        # Normalize to unit length
        magnitude = math.sqrt(sum(x * x for x in embedding))
        if magnitude > 0:
            embedding = [x / magnitude for x in embedding]

        return embedding  # type: ignore[no-any-return]
