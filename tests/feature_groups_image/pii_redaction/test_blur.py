"""Tests for BlurPIIRedactor."""

from typing import Type

import pytest

from rag_integration.feature_groups.image_pipeline.pii_redaction import BlurPIIRedactor
from rag_integration.feature_groups.image_pipeline.pii_redaction.base import BaseImagePIIRedactor
from tests.feature_groups_image.pii_redaction.image_pii_redaction_test_base import (
    ImagePIIRedactionTestBase,
    can_import_pillow,
)


@pytest.mark.skipif(not can_import_pillow(), reason="Pillow required")
class TestBlurPIIRedactor(ImagePIIRedactionTestBase):
    """Tests for BlurPIIRedactor."""

    @property
    def redactor_class(self) -> Type[BaseImagePIIRedactor]:
        return BlurPIIRedactor
