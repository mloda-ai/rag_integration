"""Tests for PerceptualHashImageDeduplicator."""

import io
from typing import List, Optional, Type

import pytest

from rag_integration.feature_groups.image_pipeline.deduplication import PerceptualHashImageDeduplicator
from rag_integration.feature_groups.image_pipeline.deduplication.base import BaseImageDeduplicator
from tests.feature_groups_image.deduplication.image_dedup_test_base import ImageDeduplicationTestBase, can_import_pillow


@pytest.mark.skipif(not can_import_pillow(), reason="Pillow required")
class TestPerceptualHashImageDeduplicator(ImageDeduplicationTestBase):
    """Tests for PerceptualHashImageDeduplicator."""

    @property
    def deduplicator_class(self) -> Type[BaseImageDeduplicator]:
        return PerceptualHashImageDeduplicator

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

    def test_different_images_not_duplicates(self) -> None:
        """Structurally different images should not be duplicates."""
        from PIL import Image, ImageDraw

        img1 = Image.new("RGB", (64, 64), color=(255, 255, 255))
        draw1 = ImageDraw.Draw(img1)
        draw1.rectangle([0, 0, 32, 64], fill=(0, 0, 0))

        img2 = Image.new("RGB", (64, 64), color=(255, 255, 255))
        draw2 = ImageDraw.Draw(img2)
        draw2.rectangle([0, 0, 64, 32], fill=(0, 0, 0))

        buf1, buf2 = io.BytesIO(), io.BytesIO()
        img1.save(buf1, format="PNG")
        img2.save(buf2, format="PNG")

        images = [buf1.getvalue(), buf2.getvalue()]
        result = PerceptualHashImageDeduplicator._find_duplicates(images, 1.0)
        assert result[0] is None
        assert result[1] is None

    def test_hamming_distance(self) -> None:
        """Hamming distance should be computed correctly."""
        assert PerceptualHashImageDeduplicator._hamming_distance(0b1100, 0b1010) == 2
        assert PerceptualHashImageDeduplicator._hamming_distance(0, 0) == 0
