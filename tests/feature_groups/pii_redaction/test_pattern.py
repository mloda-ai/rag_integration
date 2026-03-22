"""Tests for PatternPIIRedactor."""

from typing import Type

from rag_integration.feature_groups.rag_pipeline.pii_redaction import PatternPIIRedactor
from rag_integration.feature_groups.rag_pipeline.pii_redaction.base import BasePIIRedactor
from tests.feature_groups.pii_redaction.text_pii_redaction_test_base import TextPIIRedactionTestBase


class TestPatternPIIRedactor(TextPIIRedactionTestBase):
    """Tests for PatternPIIRedactor."""

    @property
    def redactor_class(self) -> Type[BasePIIRedactor]:
        return PatternPIIRedactor

    def test_credit_card_redaction(self) -> None:
        """Should redact credit card numbers."""
        texts = ["Card: 1234-5678-9012-3456"]
        result = PatternPIIRedactor._redact_pii(texts, ["CREDIT_CARD"], "mask")
        assert "1234-5678-9012-3456" not in result[0]
        assert "[REDACTED]" in result[0]

    def test_credit_card_no_dashes(self) -> None:
        """Should redact credit card numbers without dashes."""
        texts = ["Card: 1234567890123456"]
        result = PatternPIIRedactor._redact_pii(texts, ["CREDIT_CARD"], "mask")
        assert "1234567890123456" not in result[0]

    def test_ip_address_redaction(self) -> None:
        """Should redact IP addresses."""
        texts = ["Server at 192.168.1.100"]
        result = PatternPIIRedactor._redact_pii(texts, ["IP_ADDRESS"], "mask")
        assert "192.168.1.100" not in result[0]
        assert "[REDACTED]" in result[0]

    def test_redact_all(self) -> None:
        """Should redact all PII types when ALL specified."""
        texts = ["Email: a@b.com, Phone: 555-111-2222, Card: 1234-5678-9012-3456, IP: 1.2.3.4"]
        result = PatternPIIRedactor._redact_pii(texts, ["ALL"], "mask")
        assert "@" not in result[0]
        assert "555-111-2222" not in result[0]
        assert "1234-5678-9012-3456" not in result[0]
        assert "1.2.3.4" not in result[0]
