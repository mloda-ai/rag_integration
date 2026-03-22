"""Tests for DifferenceHashImageDeduplicator."""

from typing import List, Optional, Type

import pytest

from rag_integration.feature_groups.image_pipeline.deduplication import DifferenceHashImageDeduplicator
from rag_integration.feature_groups.image_pipeline.deduplication.base import BaseImageDeduplicator
from tests.feature_groups_image.deduplication.image_dedup_test_base import ImageDeduplicationTestBase, can_import_pillow


@pytest.mark.skipif(not can_import_pillow(), reason="Pillow required")
class TestDifferenceHashImageDeduplicator(ImageDeduplicationTestBase):
    """Tests for DifferenceHashImageDeduplicator."""

    @property
    def deduplicator_class(self) -> Type[BaseImageDeduplicator]:
        return DifferenceHashImageDeduplicator

    @property
    def duplicate_images(self) -> List[bytes]:
        img = self.create_test_image((255, 0, 0))
        return [img, img]

    @property
    def duplicate_expected_indices(self) -> List[Optional[int]]:
        return [None, 0]

    @property
    def unique_images(self) -> List[bytes]:
        return [
            self.create_patterned_image("left_half"),
            self.create_patterned_image("top_half"),
            self.create_patterned_image("diagonal"),
        ]

    @property
    def default_threshold(self) -> float:
        return 0.9

    def test_hamming_distance(self) -> None:
        """Hamming distance should be computed correctly."""
        assert DifferenceHashImageDeduplicator._hamming_distance(0b1100, 0b1010) == 2
        assert DifferenceHashImageDeduplicator._hamming_distance(0, 0) == 0
