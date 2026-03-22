"""Tests for PixelPIIRedactor."""

from typing import Type

import pytest

from rag_integration.feature_groups.image_pipeline.pii_redaction import PixelPIIRedactor
from rag_integration.feature_groups.image_pipeline.pii_redaction.base import BaseImagePIIRedactor
from tests.feature_groups_image.pii_redaction.image_pii_redaction_test_base import (
    ImagePIIRedactionTestBase,
    can_import_pillow,
)


@pytest.mark.skipif(not can_import_pillow(), reason="Pillow required")
class TestPixelPIIRedactor(ImagePIIRedactionTestBase):
    """Tests for PixelPIIRedactor."""

    @property
    def redactor_class(self) -> Type[BaseImagePIIRedactor]:
        return PixelPIIRedactor
