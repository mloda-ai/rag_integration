"""Normalize image preprocessing."""

from __future__ import annotations

from typing import List

from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys

from rag_integration.feature_groups.image_pipeline.preprocessing.base import BaseImagePreprocessor


class NormalizePreprocessor(BaseImagePreprocessor):
    """
    Normalize image preprocessor.

    Converts images to RGB, resizes to target dimensions, and normalizes
    pixel values. This is the standard preprocessing for most vision models.

    The normalization converts to a consistent format (RGB, target size)
    while keeping the image as bytes for the pipeline.

    Configuration:
        target_size: [width, height] in pixels (default: [224, 224])

    Config-based matching:
        preprocessing_method="normalize"
    """

    PROPERTY_MAPPING = {
        BaseImagePreprocessor.PREPROCESSING_METHOD: {
            "normalize": "Normalize pixel values to [0, 1] range",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: True,
        },
        BaseImagePreprocessor.TARGET_SIZE: {
            "explanation": "Target size as [width, height] in pixels",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.default: [224, 224],
        },
        DefaultOptionKeys.in_features: {
            "explanation": "Source feature containing images to preprocess",
            DefaultOptionKeys.context: True,
        },
    }

    @classmethod
    def _preprocess_image(
        cls,
        image_data: bytes,
        image_format: str,
        target_size: List[int],
    ) -> bytes:
        """
        Normalize image: convert to RGB, resize, and standardize.

        Args:
            image_data: Raw image bytes
            image_format: Image format (png, jpeg, etc.)
            target_size: [width, height] in pixels

        Returns:
            Normalized image as PNG bytes (always outputs PNG for lossless)
        """
        try:
            from PIL import Image
        except ImportError:
            raise ImportError("Pillow is required for NormalizePreprocessor. Install with: pip install Pillow")

        import io

        img = Image.open(io.BytesIO(image_data))

        # Convert to RGB (handles RGBA, grayscale, etc.)
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Resize to target dimensions
        width, height = target_size[0], target_size[1]
        img = img.resize((width, height), Image.LANCZOS)

        # Output as PNG for lossless standardization
        output = io.BytesIO()
        img.save(output, format="PNG")
        return output.getvalue()
