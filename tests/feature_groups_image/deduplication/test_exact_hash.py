"""Tests for ExactHashImageDeduplicator."""

from typing import List, Optional, Type

from rag_integration.feature_groups.image_pipeline.deduplication import ExactHashImageDeduplicator
from rag_integration.feature_groups.image_pipeline.deduplication.base import BaseImageDeduplicator
from tests.feature_groups_image.deduplication.image_dedup_test_base import ImageDeduplicationTestBase


class TestExactHashImageDeduplicator(ImageDeduplicationTestBase):
    """Tests for ExactHashImageDeduplicator."""

    @property
    def deduplicator_class(self) -> Type[BaseImageDeduplicator]:
        return ExactHashImageDeduplicator

    @property
    def duplicate_images(self) -> List[bytes]:
        return [b"image_a", b"image_b", b"image_a", b"image_c"]

    @property
    def duplicate_expected_indices(self) -> List[Optional[int]]:
        return [None, None, 0, None]

    @property
    def unique_images(self) -> List[bytes]:
        return [b"img_1", b"img_2", b"img_3"]

    @property
    def default_threshold(self) -> float:
        return 1.0

    def test_all_same(self) -> None:
        """Should detect all-same images."""
        images = [b"same", b"same", b"same"]
        result = ExactHashImageDeduplicator._find_duplicates(images, 1.0)
        assert result[0] is None
        assert result[1] == 0
        assert result[2] == 0
