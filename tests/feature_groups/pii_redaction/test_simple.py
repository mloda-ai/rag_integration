"""Tests for SimplePIIRedactor."""

from rag_integration.feature_groups.rag_pipeline.pii_redaction import SimplePIIRedactor


class TestSimplePIIRedactor:
    """Tests for SimplePIIRedactor."""

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
        # Full name replaced with single redaction
        assert result[0].count("[REDACTED]") == 1

    def test_redact_email(self) -> None:
        """Should redact email addresses."""
        texts = ["Email me at alice@example.com please."]
        result = SimplePIIRedactor._redact_pii(texts, ["EMAIL"], "mask")
        assert "alice@example.com" not in result[0]
        assert "[REDACTED]" in result[0]

    def test_redact_phone(self) -> None:
        """Should redact phone numbers."""
        texts = ["Call 555-123-4567 for info."]
        result = SimplePIIRedactor._redact_pii(texts, ["PHONE"], "mask")
        assert "555-123-4567" not in result[0]
        assert "[REDACTED]" in result[0]

    def test_redact_ssn(self) -> None:
        """Should redact SSN."""
        texts = ["SSN: 123-45-6789"]
        result = SimplePIIRedactor._redact_pii(texts, ["SSN"], "mask")
        assert "123-45-6789" not in result[0]
        assert "[REDACTED]" in result[0]

    def test_type_label_replacement(self) -> None:
        """Should use type labels when configured."""
        texts = ["Contact John at john@example.com"]
        result = SimplePIIRedactor._redact_pii(texts, ["NAME", "EMAIL"], "type_label")
        assert "[NAME]" in result[0]
        assert "[EMAIL]" in result[0]

    def test_redact_all(self) -> None:
        """Should redact all PII types when ALL specified."""
        texts = ["John Smith: 555-111-2222, SSN 111-22-3333, test@example.com"]
        result = SimplePIIRedactor._redact_pii(texts, ["ALL"], "mask")
        assert "John" not in result[0]
        assert "555-111-2222" not in result[0]
        assert "111-22-3333" not in result[0]
        assert "@" not in result[0]

    def test_feature_matching_pattern(self) -> None:
        """Should match pii_redacted features."""
        from mloda.user import Options

        assert SimplePIIRedactor.match_feature_group_criteria("docs__pii_redacted", Options())
        assert SimplePIIRedactor.match_feature_group_criteria("text__pii_redacted", Options())
        assert not SimplePIIRedactor.match_feature_group_criteria("docs__chunked", Options())
