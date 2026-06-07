"""Tests for BlurPIIRedactor."""

from typing import Type
from unittest.mock import patch

import pytest
from mloda.user import Feature, Options

from rag_integration.feature_groups.image_pipeline.pii_redaction import BlurPIIRedactor
from rag_integration.feature_groups.image_pipeline.pii_redaction.base import BaseImagePIIRedactor
from tests.feature_groups_image.pii_redaction.image_pii_redaction_test_base import (
    ImagePIIRedactionTestBase,
    can_import_pillow,
    create_test_image,
)


@pytest.mark.skipif(not can_import_pillow(), reason="Pillow required")
class TestBlurPIIRedactor(ImagePIIRedactionTestBase):
    """Tests for BlurPIIRedactor."""

    @property
    def redactor_class(self) -> Type[BaseImagePIIRedactor]:
        return BlurPIIRedactor


class TestBlurRadiusOption:
    """Tests for the configurable blur radius (issue: promote hardcoded values)."""

    def test_blur_radius_default(self) -> None:
        feature = Feature("image_docs__pii_redacted", options=Options())
        assert BlurPIIRedactor._get_blur_radius(feature) == BlurPIIRedactor.DEFAULT_BLUR_RADIUS

    def test_blur_radius_from_options(self) -> None:
        feature = Feature("image_docs__pii_redacted", options=Options(context={"blur_radius": 7}))
        assert BlurPIIRedactor._get_blur_radius(feature) == 7

    def test_redact_region_for_feature_threads_radius(self) -> None:
        feature = Feature("image_docs__pii_redacted", options=Options(context={"blur_radius": 7}))
        with patch.object(BlurPIIRedactor, "_redact_region", return_value=b"out") as mock_redact:
            result = BlurPIIRedactor._redact_region_for_feature(b"data", "png", [{"bbox": [0, 0, 1, 1]}], feature)

        assert result == b"out"
        mock_redact.assert_called_once_with(b"data", "png", [{"bbox": [0, 0, 1, 1]}], blur_radius=7)

    @pytest.mark.skipif(not can_import_pillow(), reason="Pillow required")
    def test_custom_radius_produces_valid_image(self) -> None:
        image_data = create_test_image()
        regions = [{"bbox": [10, 10, 50, 50], "type": "FACE"}]
        result = BlurPIIRedactor._redact_region(image_data, "png", regions, blur_radius=3)
        assert isinstance(result, bytes)
        assert len(result) > 0
