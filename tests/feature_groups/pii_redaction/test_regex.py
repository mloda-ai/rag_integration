"""Tests for RegexPIIRedactor."""

from rag_integration.feature_groups.rag_pipeline.pii_redaction import RegexPIIRedactor


class TestRegexPIIRedactor:
    """Tests for RegexPIIRedactor."""

    def test_redact_email(self) -> None:
        """Should redact email addresses."""
        texts = ["Contact me at john@example.com for details."]
        result = RegexPIIRedactor._redact_pii(texts, ["EMAIL"], "mask")
        assert "john@example.com" not in result[0]
        assert "[REDACTED]" in result[0]

    def test_redact_phone(self) -> None:
        """Should redact phone numbers."""
        texts = ["Call me at 555-123-4567."]
        result = RegexPIIRedactor._redact_pii(texts, ["PHONE"], "mask")
        assert "555-123-4567" not in result[0]
        assert "[REDACTED]" in result[0]

    def test_redact_ssn(self) -> None:
        """Should redact SSN."""
        texts = ["My SSN is 123-45-6789."]
        result = RegexPIIRedactor._redact_pii(texts, ["SSN"], "mask")
        assert "123-45-6789" not in result[0]
        assert "[REDACTED]" in result[0]

    def test_type_label_replacement(self) -> None:
        """Should use type labels when configured."""
        texts = ["Email: test@example.com"]
        result = RegexPIIRedactor._redact_pii(texts, ["EMAIL"], "type_label")
        assert "[EMAIL]" in result[0]

    def test_redact_all(self) -> None:
        """Should redact all PII types when ALL specified."""
        texts = ["Contact john@test.com at 555-111-2222, SSN: 111-22-3333"]
        result = RegexPIIRedactor._redact_pii(texts, ["ALL"], "mask")
        assert "@" not in result[0]
        assert "555-111-2222" not in result[0]
        assert "111-22-3333" not in result[0]

    def test_feature_matching_pattern(self) -> None:
        """Should match pii_redacted features."""
        from mloda.user import Options

        assert RegexPIIRedactor.match_feature_group_criteria("docs__pii_redacted", Options())
        assert RegexPIIRedactor.match_feature_group_criteria("text__pii_redacted", Options())
        assert not RegexPIIRedactor.match_feature_group_criteria("docs__chunked", Options())
