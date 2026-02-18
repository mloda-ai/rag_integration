"""Thumbnail image preprocessing."""

from __future__ import annotations

from typing import List

from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys

from rag_integration.feature_groups.image_pipeline.preprocessing.base import BaseImagePreprocessor


class ThumbnailPreprocessor(BaseImagePreprocessor):
    """
    Thumbnail image preprocessor.

    Generates thumbnails that preserve aspect ratio by fitting within
    the target dimensions. The resulting image may be smaller than
    target_size in one dimension.

    Configuration:
        target_size: Maximum [width, height] in pixels (default: [224, 224])

    Config-based matching:
        preprocessing_method="thumbnail"
    """

    PROPERTY_MAPPING = {
        BaseImagePreprocessor.PREPROCESSING_METHOD: {
            "thumbnail": "Generate thumbnail preserving aspect ratio",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: True,
        },
        BaseImagePreprocessor.TARGET_SIZE: {
            "explanation": "Maximum size as [width, height] in pixels",
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
        Generate thumbnail preserving aspect ratio.

        Args:
            image_data: Raw image bytes
            image_format: Image format (png, jpeg, etc.)
            target_size: Maximum [width, height] in pixels

        Returns:
            Thumbnail image as bytes
        """
        try:
            from PIL import Image
        except ImportError:
            raise ImportError("Pillow is required for ThumbnailPreprocessor. Install with: pip install Pillow")

        import io

        img = Image.open(io.BytesIO(image_data))

        # thumbnail() modifies in place and preserves aspect ratio
        width, height = target_size[0], target_size[1]
        img.thumbnail((width, height), Image.LANCZOS)

        output = io.BytesIO()
        save_format = "PNG" if image_format.lower() == "png" else "JPEG"
        img.save(output, format=save_format)
        return output.getvalue()
