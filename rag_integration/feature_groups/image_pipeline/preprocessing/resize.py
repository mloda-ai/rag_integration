"""Resize image preprocessing."""

from __future__ import annotations

from typing import List

from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys

from rag_integration.feature_groups.image_pipeline.preprocessing.base import BaseImagePreprocessor


class ResizePreprocessor(BaseImagePreprocessor):
    """
    Resize image preprocessor.

    Resizes images to exact target dimensions. Does not preserve
    aspect ratio — stretches or shrinks to fit.

    Configuration:
        target_size: [width, height] in pixels (default: [224, 224])

    Config-based matching:
        preprocessing_method="resize"
    """

    PROPERTY_MAPPING = {
        BaseImagePreprocessor.PREPROCESSING_METHOD: {
            "resize": "Resize images to target dimensions",
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
        Resize image to exact target dimensions.

        Args:
            image_data: Raw image bytes
            image_format: Image format (png, jpeg, etc.)
            target_size: [width, height] in pixels

        Returns:
            Resized image as bytes
        """
        try:
            from PIL import Image
        except ImportError:
            raise ImportError("Pillow is required for ResizePreprocessor. Install with: pip install Pillow")

        import io

        img = Image.open(io.BytesIO(image_data))
        width, height = target_size[0], target_size[1]
        resized = img.resize((width, height), Image.LANCZOS)

        output = io.BytesIO()
        save_format = "PNG" if image_format.lower() == "png" else "JPEG"
        resized.save(output, format=save_format)
        return output.getvalue()
