"""Tests for PatternPIIRedactor."""

from rag_integration.feature_groups.rag_pipeline.pii_redaction import PatternPIIRedactor


class TestPatternPIIRedactor:
    """Tests for PatternPIIRedactor."""

    def test_default_patterns_email(self) -> None:
        """Should redact email with default pattern."""
        texts = ["Contact user@example.com"]
        result = PatternPIIRedactor._redact_pii(texts, ["EMAIL"], "mask")
        assert "user@example.com" not in result[0]
        assert "[REDACTED]" in result[0]

    def test_default_patterns_phone(self) -> None:
        """Should redact phone with default pattern."""
        texts = ["Call 555-123-4567"]
        result = PatternPIIRedactor._redact_pii(texts, ["PHONE"], "mask")
        assert "555-123-4567" not in result[0]
        assert "[REDACTED]" in result[0]

    def test_default_patterns_ssn(self) -> None:
        """Should redact SSN with default pattern."""
        texts = ["SSN: 123-45-6789"]
        result = PatternPIIRedactor._redact_pii(texts, ["SSN"], "mask")
        assert "123-45-6789" not in result[0]
        assert "[REDACTED]" in result[0]

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

    def test_type_label_replacement(self) -> None:
        """Should use type labels when configured."""
        texts = ["Email: test@example.com, IP: 10.0.0.1"]
        result = PatternPIIRedactor._redact_pii(texts, ["EMAIL", "IP_ADDRESS"], "type_label")
        assert "[EMAIL]" in result[0]
        assert "[IP_ADDRESS]" in result[0]

    def test_redact_all(self) -> None:
        """Should redact all PII types when ALL specified."""
        texts = ["Email: a@b.com, Phone: 555-111-2222, Card: 1234-5678-9012-3456, IP: 1.2.3.4"]
        result = PatternPIIRedactor._redact_pii(texts, ["ALL"], "mask")
        assert "@" not in result[0]
        assert "555-111-2222" not in result[0]
        assert "1234-5678-9012-3456" not in result[0]
        assert "1.2.3.4" not in result[0]

    def test_feature_matching_pattern(self) -> None:
        """Should match pii_redacted features."""
        from mloda.user import Options

        assert PatternPIIRedactor.match_feature_group_criteria("docs__pii_redacted", Options())
        assert PatternPIIRedactor.match_feature_group_criteria("text__pii_redacted", Options())
        assert not PatternPIIRedactor.match_feature_group_criteria("docs__chunked", Options())
