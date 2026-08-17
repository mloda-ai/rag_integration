"""Tests for ResizePreprocessor."""

import io

import pytest

from rag_integration.feature_groups.image_pipeline.preprocessing import ResizePreprocessor
from rag_integration.feature_groups.image_pipeline.preprocessing.base import BaseImagePreprocessor
from tests.feature_groups_image.preprocessing.image_preprocessing_test_base import (
    ImagePreprocessingTestBase,
    can_import_pillow,
    create_test_image,
)


@pytest.mark.skipif(not can_import_pillow(), reason="Pillow required")
class TestResizePreprocessor(ImagePreprocessingTestBase):
    """Tests for ResizePreprocessor."""

    @property
    def preprocessor_class(self) -> type[BaseImagePreprocessor]:
        return ResizePreprocessor

    @property
    def target_size(self) -> list[int]:
        return [100, 100]

    @property
    def feature_match_name(self) -> str:
        return "image_docs__pii_redacted__preprocessed"

    @property
    def feature_reject_name(self) -> str:
        return "image_docs__pii_redacted"

    def test_resize_upscale(self) -> None:
        """Should upscale smaller images."""
        from PIL import Image

        image_data = create_test_image(50, 50)
        result = ResizePreprocessor._preprocess_image(image_data, "png", [224, 224])
        img = Image.open(io.BytesIO(result))
        assert img.size == (224, 224)
