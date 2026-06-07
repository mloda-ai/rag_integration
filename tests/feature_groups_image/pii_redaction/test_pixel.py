"""Tests for PixelPIIRedactor."""

from typing import Type
from unittest.mock import patch

import pytest
from mloda.user import Feature, Options

from rag_integration.feature_groups.image_pipeline.pii_redaction import PixelPIIRedactor
from rag_integration.feature_groups.image_pipeline.pii_redaction.base import BaseImagePIIRedactor
from tests.feature_groups_image.pii_redaction.image_pii_redaction_test_base import (
    ImagePIIRedactionTestBase,
    can_import_pillow,
    create_test_image,
)


@pytest.mark.skipif(not can_import_pillow(), reason="Pillow required")
class TestPixelPIIRedactor(ImagePIIRedactionTestBase):
    """Tests for PixelPIIRedactor."""

    @property
    def redactor_class(self) -> Type[BaseImagePIIRedactor]:
        return PixelPIIRedactor


class TestPixelSizeOption:
    """Tests for the configurable pixel size (issue: promote hardcoded values)."""

    def test_pixel_size_default(self) -> None:
        feature = Feature("image_docs__pii_redacted", options=Options())
        assert PixelPIIRedactor._get_pixel_size(feature) == PixelPIIRedactor.DEFAULT_PIXEL_SIZE

    def test_pixel_size_from_options(self) -> None:
        feature = Feature("image_docs__pii_redacted", options=Options(context={"pixel_size": 4}))
        assert PixelPIIRedactor._get_pixel_size(feature) == 4

    def test_redact_region_for_feature_threads_size(self) -> None:
        feature = Feature("image_docs__pii_redacted", options=Options(context={"pixel_size": 4}))
        with patch.object(PixelPIIRedactor, "_redact_region", return_value=b"out") as mock_redact:
            result = PixelPIIRedactor._redact_region_for_feature(b"data", "png", [{"bbox": [0, 0, 1, 1]}], feature)

        assert result == b"out"
        mock_redact.assert_called_once_with(b"data", "png", [{"bbox": [0, 0, 1, 1]}], pixel_size=4)

    @pytest.mark.skipif(not can_import_pillow(), reason="Pillow required")
    def test_custom_size_produces_valid_image(self) -> None:
        image_data = create_test_image()
        regions = [{"bbox": [10, 10, 50, 50], "type": "FACE"}]
        result = PixelPIIRedactor._redact_region(image_data, "png", regions, pixel_size=4)
        assert isinstance(result, bytes)
        assert len(result) > 0

    @pytest.mark.skipif(not can_import_pillow(), reason="Pillow required")
    def test_zero_size_does_not_raise(self) -> None:
        """A zero pixel size is clamped to 1 instead of dividing by zero."""
        image_data = create_test_image()
        regions = [{"bbox": [10, 10, 50, 50], "type": "FACE"}]
        result = PixelPIIRedactor._redact_region(image_data, "png", regions, pixel_size=0)
        assert isinstance(result, bytes)
        assert len(result) > 0
