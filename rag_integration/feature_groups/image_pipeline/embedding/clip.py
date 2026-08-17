"""CLIP model image and text embedding.

CLIPImageEmbedder handles mixed image+text rows: corpus rows are embedded via
CLIP's vision encoder; query rows that have a ``caption`` field but no
``image_data`` are embedded via CLIP's text encoder.  Both encoders project
into the same 512-d shared embedding space, enabling cross-modal similarity.
"""

from __future__ import annotations

import math
import os
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from mloda.provider import DefaultOptionKeys, FeatureSet, property_spec

if TYPE_CHECKING:
    from mloda.user import Feature

from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_utils import columnar_to_rows

from rag_integration.feature_groups.image_pipeline.embedding.base import BaseImageEmbedder

# Default local model path: looks two levels above the git repo root (mloda/models/)
# clip.py lives at rag_integration/rag_integration/feature_groups/image_pipeline/embedding/clip.py
# parents[4] = rag_integration repo root, parents[5] = mloda/ directory
_DEFAULT_LOCAL_MODEL = str(Path(__file__).resolve().parents[5] / "models" / "clip-vit-base-patch32")
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

    Note: Caches (model, processor) pairs per resolved path at class level.
    Initialization is guarded by a lock so concurrent callers do not build the
    same model twice.
    """

    _model_cache: ClassVar[dict[str, Any]] = {}
    _model_lock = threading.Lock()

    # Matches the PROPERTY_MAPPING default below; used when a caller invokes
    # calculate_feature directly and skips mloda's option-default materialization.
    DEFAULT_MODEL = _HF_MODEL_ID

    PROPERTY_MAPPING: ClassVar = {
        BaseImageEmbedder.IMAGE_EMBEDDING_METHOD: property_spec(
            "Algorithm used to embed images into vectors",
            strict=True,
            allowed_values={"clip": "CLIP model image embeddings"},
            deferred_binding=True,
        ),
        BaseImageEmbedder.EMBEDDING_DIM: property_spec(
            "Dimension of the embedding vectors (CLIP default: 512)", default=512
        ),
        BaseImageEmbedder.MODEL_NAME: property_spec("Local path or HuggingFace model ID", default=_HF_MODEL_ID),
        DefaultOptionKeys.in_features: property_spec("Source feature containing images to embed"),
    }

    @classmethod
    def _get_model_name(cls, feature: Feature) -> str:
        """Get model name from feature options, defaulting to the HuggingFace CLIP model ID."""
        name = feature.options.get(cls.MODEL_NAME)
        return str(name) if name is not None else cls.DEFAULT_MODEL

    @classmethod
    def _resolve_model_path(cls, model_name: str) -> str:
        """Resolve model name: use local path if it exists, otherwise use as-is (HuggingFace ID)."""
        if os.path.isdir(model_name):
            return model_name
        if os.path.isdir(_DEFAULT_LOCAL_MODEL):
            return _DEFAULT_LOCAL_MODEL
        return model_name

    @classmethod
    def _get_model_and_processor(cls, resolved: str) -> tuple[Any, Any]:
        """Get or create the (model, processor) pair for a resolved path (thread-safe)."""
        # Fast path: single atomic dict read.
        cached = cls._model_cache.get(resolved)
        if cached is not None:
            return cached  # type: ignore[no-any-return]

        with cls._model_lock:
            # Re-check inside the lock: another thread may have built it.
            cached = cls._model_cache.get(resolved)
            if cached is None:
                try:
                    from transformers import CLIPModel, CLIPProcessor
                except ImportError as e:
                    raise ImportError(
                        "transformers is required for CLIPImageEmbedder. Install with: pip install transformers"
                    ) from e

                cached = (
                    CLIPModel.from_pretrained(resolved),  # nosec B615
                    CLIPProcessor.from_pretrained(resolved),  # nosec B615
                )
                cls._model_cache[resolved] = cached
            return cached

    @classmethod
    def _embed_image(
        cls,
        image_data: bytes,
        embedding_dim: int,
        model_name: str,
    ) -> list[float]:
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
            import torch
            from PIL import Image
        except ImportError:
            raise ImportError(
                "torch and Pillow are required for CLIPImageEmbedder. Install with: pip install torch Pillow"
            )

        import io

        if not image_data:
            return [0.0] * embedding_dim

        img = Image.open(io.BytesIO(image_data)).convert("RGB")

        resolved = cls._resolve_model_path(model_name)
        model, processor = cls._get_model_and_processor(resolved)

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

    @classmethod
    def _embed_text(
        cls,
        text: str,
        embedding_dim: int,
        model_name: str,
    ) -> list[float]:
        """
        Generate CLIP embedding for a text string using the text encoder.

        Projects the caption into the same 512-d space as the image embeddings,
        enabling cross-modal cosine similarity (text-to-image retrieval).

        Args:
            text: Caption or query string
            embedding_dim: Expected output dimension (used for zero fallback)
            model_name: CLIP model identifier

        Returns:
            Unit-normalised CLIP text embedding vector
        """
        try:
            import torch
        except ImportError:
            raise ImportError("torch is required for CLIPImageEmbedder. Install with: pip install torch")

        if not text:
            return [0.0] * embedding_dim

        resolved = cls._resolve_model_path(model_name)
        model, processor = cls._get_model_and_processor(resolved)

        inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True)

        with torch.no_grad():
            text_outputs = model.text_model(**{k: v for k, v in inputs.items() if k != "pixel_values"})
            text_features = model.text_projection(text_outputs.pooler_output)

        embedding = text_features.squeeze(0).tolist()

        # Normalize to unit length
        magnitude = math.sqrt(sum(x * x for x in embedding))
        if magnitude > 0:
            embedding = [x / magnitude for x in embedding]

        return embedding  # type: ignore[no-any-return]

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> list[dict[str, Any]]:
        """
        Embed each row using CLIP's vision or text encoder based on row content.

        - Corpus rows (``image_data`` present): embedded via CLIP vision encoder.
        - Query rows (``image_data`` is None, ``caption`` present): embedded via
          CLIP text encoder, projecting into the same shared embedding space.

        This cross-modal dispatch is what makes text-to-image Recall@K meaningful.
        """
        # mloda 0.9.0 passes columnar data; pivot to rows for row-wise reading.
        rows = columnar_to_rows(data)
        for feature in features.features:
            embedding_dim = cls._get_embedding_dim(feature)
            model_name = cls._get_model_name(feature)
            feature_name = feature.name

            for row in rows:
                image_data = row.get("image_data")
                caption = row.get("caption")

                if image_data:
                    if not isinstance(image_data, bytes):
                        image_data = bytes(image_data)
                    embedding = cls._embed_image(image_data, embedding_dim, model_name)
                elif caption:
                    embedding = cls._embed_text(str(caption), embedding_dim, model_name)
                else:
                    embedding = [0.0] * embedding_dim

                row[feature_name] = embedding
                row["embedding_dim"] = len(embedding)
                row["embedding_model"] = model_name

        return rows
