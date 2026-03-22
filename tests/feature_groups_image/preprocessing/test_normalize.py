"""Tests for NormalizePreprocessor."""

import io
from typing import List, Type

import pytest

from rag_integration.feature_groups.image_pipeline.preprocessing import NormalizePreprocessor
from rag_integration.feature_groups.image_pipeline.preprocessing.base import BaseImagePreprocessor
from tests.feature_groups_image.preprocessing.image_preprocessing_test_base import (
    ImagePreprocessingTestBase,
    can_import_pillow,
    create_test_image,
)


@pytest.mark.skipif(not can_import_pillow(), reason="Pillow required")
class TestNormalizePreprocessor(ImagePreprocessingTestBase):
    """Tests for NormalizePreprocessor."""

    @property
    def preprocessor_class(self) -> Type[BaseImagePreprocessor]:
        return NormalizePreprocessor

    @property
    def target_size(self) -> List[int]:
        return [64, 64]

    @property
    def feature_match_name(self) -> str:
        return "image_docs__preprocessed"

    @property
    def feature_reject_name(self) -> str:
        return "image_docs__embedded"

    def test_converts_rgba_to_rgb(self) -> None:
        """Should convert RGBA images to RGB."""
        from PIL import Image

        image_data = create_test_image(100, 100, mode="RGBA", color=(128, 64, 32, 200))
        result = NormalizePreprocessor._preprocess_image(image_data, "png", [100, 100])
        img = Image.open(io.BytesIO(result))
        assert img.mode == "RGB"

    def test_output_is_png(self) -> None:
        """Normalized output should always be PNG."""
        from PIL import Image

        image_data = create_test_image(100, 100, mode="RGBA", color=(128, 64, 32, 200))
        result = NormalizePreprocessor._preprocess_image(image_data, "jpeg", [100, 100])
        img = Image.open(io.BytesIO(result))
        assert img.format == "PNG"
