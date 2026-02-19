"""Tests for SolidFillPIIRedactor."""

import pytest

from rag_integration.feature_groups.image_pipeline.pii_redaction import SolidFillPIIRedactor


def _can_import_pillow() -> bool:
    try:
        from PIL import Image  # noqa: F401

        return True
    except ImportError:
        return False


def _create_test_image(width: int = 100, height: int = 100) -> bytes:
    """Create a simple test PNG image using Pillow."""
    from PIL import Image
    import io

    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    output = io.BytesIO()
    img.save(output, format="PNG")
    return output.getvalue()


@pytest.mark.skipif(
    not _can_import_pillow(),
    reason="Pillow required",
)
class TestSolidFillPIIRedactor:
    """Tests for SolidFillPIIRedactor."""

    def test_redact_single_region(self) -> None:
        """Should fill a single region with solid color."""
        image_data = _create_test_image()
        regions = [{"bbox": [10, 10, 50, 50], "type": "FACE"}]

        result = SolidFillPIIRedactor._redact_region(image_data, "png", regions)
        assert isinstance(result, bytes)
        assert len(result) > 0
        # Redacted image should differ from original
        assert result != image_data

    def test_redact_multiple_regions(self) -> None:
        """Should fill multiple regions."""
        image_data = _create_test_image()
        regions = [
            {"bbox": [10, 10, 30, 30], "type": "FACE"},
            {"bbox": [50, 50, 80, 80], "type": "TEXT"},
        ]

        result = SolidFillPIIRedactor._redact_region(image_data, "png", regions)
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_empty_regions(self) -> None:
        """Should return original image when no regions provided."""
        image_data = _create_test_image()

        result = SolidFillPIIRedactor._redact_region(image_data, "png", [])
        assert isinstance(result, bytes)

    def test_invalid_bbox_skipped(self) -> None:
        """Should skip regions with invalid bboxes."""
        image_data = _create_test_image()
        regions = [{"bbox": [10], "type": "FACE"}]  # Invalid bbox

        result = SolidFillPIIRedactor._redact_region(image_data, "png", regions)
        assert isinstance(result, bytes)

    def test_feature_matching_pattern(self) -> None:
        """Should match pii_redacted features."""
        from mloda.user import Options

        assert SolidFillPIIRedactor.match_feature_group_criteria("image_docs__pii_redacted", Options())
        assert not SolidFillPIIRedactor.match_feature_group_criteria("image_docs__preprocessed", Options())
