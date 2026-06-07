"""Tests for SolidFillPIIRedactor."""

from typing import Type
from unittest.mock import patch

import pytest
from mloda.user import Feature, Options

from rag_integration.feature_groups.image_pipeline.pii_redaction import SolidFillPIIRedactor
from rag_integration.feature_groups.image_pipeline.pii_redaction.base import BaseImagePIIRedactor
from tests.feature_groups_image.pii_redaction.image_pii_redaction_test_base import (
    ImagePIIRedactionTestBase,
    can_import_pillow,
    create_test_image,
)


@pytest.mark.skipif(not can_import_pillow(), reason="Pillow required")
class TestSolidFillPIIRedactor(ImagePIIRedactionTestBase):
    """Tests for SolidFillPIIRedactor."""

    @property
    def redactor_class(self) -> Type[BaseImagePIIRedactor]:
        return SolidFillPIIRedactor


class TestFillColorOption:
    """Tests for the configurable fill color (issue: promote hardcoded values)."""

    def test_fill_color_default(self) -> None:
        feature = Feature("image_docs__pii_redacted", options=Options())
        assert SolidFillPIIRedactor._get_fill_color(feature) == SolidFillPIIRedactor.DEFAULT_FILL_COLOR

    def test_fill_color_from_options(self) -> None:
        feature = Feature("image_docs__pii_redacted", options=Options(context={"fill_color": [255, 0, 0]}))
        assert SolidFillPIIRedactor._get_fill_color(feature) == (255, 0, 0)

    def test_fill_color_wrong_length_raises(self) -> None:
        feature = Feature("image_docs__pii_redacted", options=Options(context={"fill_color": [255, 0]}))
        with pytest.raises(ValueError, match="RGB triple"):
            SolidFillPIIRedactor._get_fill_color(feature)

    def test_redact_region_for_feature_threads_color(self) -> None:
        feature = Feature("image_docs__pii_redacted", options=Options(context={"fill_color": [255, 0, 0]}))
        with patch.object(SolidFillPIIRedactor, "_redact_region", return_value=b"out") as mock_redact:
            result = SolidFillPIIRedactor._redact_region_for_feature(b"data", "png", [{"bbox": [0, 0, 1, 1]}], feature)

        assert result == b"out"
        mock_redact.assert_called_once_with(b"data", "png", [{"bbox": [0, 0, 1, 1]}], fill_color=(255, 0, 0))

    @pytest.mark.skipif(not can_import_pillow(), reason="Pillow required")
    def test_custom_color_produces_valid_image(self) -> None:
        image_data = create_test_image()
        regions = [{"bbox": [10, 10, 50, 50], "type": "FACE"}]
        result = SolidFillPIIRedactor._redact_region(image_data, "png", regions, fill_color=(255, 0, 0))
        assert isinstance(result, bytes)
        assert len(result) > 0
