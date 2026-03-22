"""Tests for SimplePIIRedactor."""

from typing import Type

from rag_integration.feature_groups.rag_pipeline.pii_redaction import SimplePIIRedactor
from rag_integration.feature_groups.rag_pipeline.pii_redaction.base import BasePIIRedactor
from tests.feature_groups.pii_redaction.text_pii_redaction_test_base import TextPIIRedactionTestBase


class TestSimplePIIRedactor(TextPIIRedactionTestBase):
    """Tests for SimplePIIRedactor."""

    @property
    def redactor_class(self) -> Type[BasePIIRedactor]:
        return SimplePIIRedactor

    def test_redact_common_first_name(self) -> None:
        """Should redact common first names."""
        texts = ["Hello John, how are you?"]
        result = SimplePIIRedactor._redact_pii(texts, ["NAME"], "mask")
        assert "John" not in result[0]
        assert "[REDACTED]" in result[0]

    def test_redact_full_name(self) -> None:
        """Should redact full names (first + last)."""
        texts = ["Contact John Smith for details."]
        result = SimplePIIRedactor._redact_pii(texts, ["NAME"], "mask")
        assert "John" not in result[0]
        assert "Smith" not in result[0]
        assert result[0].count("[REDACTED]") == 1

    def test_redact_all(self) -> None:
        """Should redact all PII types when ALL specified."""
        texts = ["John Smith: 555-111-2222, SSN 111-22-3333, test@example.com"]
        result = SimplePIIRedactor._redact_pii(texts, ["ALL"], "mask")
        assert "John" not in result[0]
        assert "555-111-2222" not in result[0]
        assert "111-22-3333" not in result[0]
        assert "@" not in result[0]
