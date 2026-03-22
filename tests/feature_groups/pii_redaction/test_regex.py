"""Tests for RegexPIIRedactor."""

from typing import Type

from rag_integration.feature_groups.rag_pipeline.pii_redaction import RegexPIIRedactor
from rag_integration.feature_groups.rag_pipeline.pii_redaction.base import BasePIIRedactor
from tests.feature_groups.pii_redaction.text_pii_redaction_test_base import TextPIIRedactionTestBase


class TestRegexPIIRedactor(TextPIIRedactionTestBase):
    """Tests for RegexPIIRedactor."""

    @property
    def redactor_class(self) -> Type[BasePIIRedactor]:
        return RegexPIIRedactor

    def test_redact_all(self) -> None:
        """Should redact all PII types when ALL specified."""
        texts = ["Contact john@test.com at 555-111-2222, SSN: 111-22-3333"]
        result = RegexPIIRedactor._redact_pii(texts, ["ALL"], "mask")
        assert "@" not in result[0]
        assert "555-111-2222" not in result[0]
        assert "111-22-3333" not in result[0]
