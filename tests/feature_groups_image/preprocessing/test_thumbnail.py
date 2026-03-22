"""Tests for ThumbnailPreprocessor."""

import io
from typing import List, Type

import pytest

from rag_integration.feature_groups.image_pipeline.preprocessing import ThumbnailPreprocessor
from rag_integration.feature_groups.image_pipeline.preprocessing.base import BaseImagePreprocessor
from tests.feature_groups_image.preprocessing.image_preprocessing_test_base import (
    ImagePreprocessingTestBase,
    can_import_pillow,
    create_test_image,
)


@pytest.mark.skipif(not can_import_pillow(), reason="Pillow required")
class TestThumbnailPreprocessor(ImagePreprocessingTestBase):
    """Tests for ThumbnailPreprocessor."""

    @property
    def preprocessor_class(self) -> Type[BaseImagePreprocessor]:
        return ThumbnailPreprocessor

    @property
    def target_size(self) -> List[int]:
        return [100, 100]

    @property
    def feature_match_name(self) -> str:
        return "image_docs__pii_redacted__preprocessed"

    @property
    def feature_reject_name(self) -> str:
        return "image_docs__pii_redacted"

    def test_preserves_aspect_ratio(self) -> None:
        """Thumbnail should preserve aspect ratio, fitting within target."""
        from PIL import Image

        image_data = create_test_image(400, 200)
        result = ThumbnailPreprocessor._preprocess_image(image_data, "png", [100, 100])
        img = Image.open(io.BytesIO(result))
        # 400x200 (2:1 ratio) fit into 100x100 -> 100x50
        assert img.size == (100, 50)
