"""Base test class for image deduplication feature groups."""

from __future__ import annotations

import io
from abc import ABC, abstractmethod
from typing import List, Optional, Type

from mloda.user import Options

from rag_integration.feature_groups.image_pipeline.deduplication.base import BaseImageDeduplicator


def can_import_pillow() -> bool:
    """Check if Pillow is available."""
    try:
        import PIL  # noqa: F401

        return True
    except ImportError:
        return False


class ImageDeduplicationTestBase(ABC):
    """Abstract base providing shared tests for all image deduplication implementations."""

    @property
    @abstractmethod
    def deduplicator_class(self) -> Type[BaseImageDeduplicator]: ...

    @property
    @abstractmethod
    def duplicate_images(self) -> List[bytes]: ...

    @property
    @abstractmethod
    def duplicate_expected_indices(self) -> List[Optional[int]]: ...

    @property
    @abstractmethod
    def unique_images(self) -> List[bytes]: ...

    @property
    @abstractmethod
    def default_threshold(self) -> float: ...

    @staticmethod
    def create_test_image(color: tuple[int, int, int] = (255, 0, 0), size: tuple[int, int] = (64, 64)) -> bytes:
        """Create a simple solid-color test image."""
        from PIL import Image

        img = Image.new("RGB", size, color=color)
        output = io.BytesIO()
        img.save(output, format="PNG")
        return output.getvalue()

    @staticmethod
    def create_patterned_image(pattern: str, size: tuple[int, int] = (64, 64)) -> bytes:
        """Create a test image with distinct internal structure."""
        from PIL import Image, ImageDraw

        img = Image.new("RGB", size, color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        if pattern == "left_half":
            draw.rectangle([0, 0, size[0] // 2, size[1]], fill=(0, 0, 0))
        elif pattern == "top_half":
            draw.rectangle([0, 0, size[0], size[1] // 2], fill=(0, 0, 0))
        elif pattern == "diagonal":
            for i in range(size[0]):
                draw.rectangle([i, i, i + 1, i + 1], fill=(0, 0, 0))
        output = io.BytesIO()
        img.save(output, format="PNG")
        return output.getvalue()

    def test_find_duplicates(self) -> None:
        """Should detect duplicates in the provided test data."""
        result = self.deduplicator_class._find_duplicates(self.duplicate_images, self.default_threshold)
        assert result == self.duplicate_expected_indices

    def test_no_duplicates(self) -> None:
        """Should return None for all unique images."""
        result = self.deduplicator_class._find_duplicates(self.unique_images, self.default_threshold)
        assert all(r is None for r in result)

    def test_feature_matching_pattern(self) -> None:
        """Should match deduped features and reject non-deduped."""
        assert self.deduplicator_class.match_feature_group_criteria("image_docs__preprocessed__deduped", Options())
        assert not self.deduplicator_class.match_feature_group_criteria("image_docs__preprocessed", Options())
