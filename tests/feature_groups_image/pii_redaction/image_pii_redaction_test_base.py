"""Base test class for image PII redaction feature groups."""

from __future__ import annotations

import io
from abc import ABC, abstractmethod
from typing import Type

from mloda.user import Options

from rag_integration.feature_groups.image_pipeline.pii_redaction.base import BaseImagePIIRedactor


def can_import_pillow() -> bool:
    """Check if Pillow is available."""
    try:
        import PIL  # noqa: F401

        return True
    except ImportError:
        return False


def create_test_image(width: int = 100, height: int = 100) -> bytes:
    """Create a test PNG image with visual variation (gradient pattern)."""
    from PIL import Image

    img = Image.new("RGB", (width, height))
    for x in range(width):
        for y in range(height):
            img.putpixel((x, y), (x * 255 // max(width - 1, 1), y * 255 // max(height - 1, 1), 128))
    output = io.BytesIO()
    img.save(output, format="PNG")
    return output.getvalue()


class ImagePIIRedactionTestBase(ABC):
    """Abstract base providing shared tests for all image PII redaction implementations."""

    @property
    @abstractmethod
    def redactor_class(self) -> Type[BaseImagePIIRedactor]: ...

    def test_redact_single_region(self) -> None:
        """Should redact a single region."""
        image_data = create_test_image()
        regions = [{"bbox": [10, 10, 50, 50], "type": "FACE"}]
        result = self.redactor_class._redact_region(image_data, "png", regions)
        assert isinstance(result, bytes)
        assert len(result) > 0
        assert result != image_data

    def test_redact_multiple_regions(self) -> None:
        """Should redact multiple regions."""
        image_data = create_test_image()
        regions = [
            {"bbox": [10, 10, 30, 30], "type": "FACE"},
            {"bbox": [50, 50, 80, 80], "type": "TEXT"},
        ]
        result = self.redactor_class._redact_region(image_data, "png", regions)
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_empty_regions(self) -> None:
        """Should return image when no regions provided."""
        image_data = create_test_image()
        result = self.redactor_class._redact_region(image_data, "png", [])
        assert isinstance(result, bytes)

    def test_invalid_bbox_skipped(self) -> None:
        """Should skip regions with invalid bboxes."""
        image_data = create_test_image()
        regions = [{"bbox": [10], "type": "FACE"}]
        result = self.redactor_class._redact_region(image_data, "png", regions)
        assert isinstance(result, bytes)

    def test_feature_matching_pattern(self) -> None:
        """Should match pii_redacted features and reject others."""
        assert self.redactor_class.match_feature_group_criteria("image_docs__pii_redacted", Options())
        assert not self.redactor_class.match_feature_group_criteria("image_docs__preprocessed", Options())
