"""Tests for PatternPIIRedactor."""

from mloda.provider import FeatureSet
from mloda.user import Feature, Options

from rag_integration.feature_groups.rag_pipeline.pii_redaction import PatternPIIRedactor
from rag_integration.feature_groups.rag_pipeline.pii_redaction.base import BasePIIRedactor
from tests.feature_groups.pii_redaction.text_pii_redaction_test_base import TextPIIRedactionTestBase


class TestPatternPIIRedactor(TextPIIRedactionTestBase):
    """Tests for PatternPIIRedactor."""

    @property
    def redactor_class(self) -> type[BasePIIRedactor]:
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

    def test_custom_patterns_are_per_feature(self) -> None:
        """Features in one FeatureSet each use their own custom_patterns, and no state leaks afterwards."""

        def feature(name: str, pii_type: str, regex: str) -> Feature:
            context = {
                "redaction_method": "pattern",
                "pii_types": [pii_type],
                "replacement_strategy": "type_label",
                "custom_patterns": {pii_type: regex},
            }
            return Feature(name, options=Options(context=context))

        features = FeatureSet(
            [
                feature("a__pii_redacted", "EMPLOYEE_ID", r"EMP-\d{6}"),
                feature("b__pii_redacted", "PROJECT", r"PRJ-[A-Z]{3}"),
            ]
        )
        text = "EMP-123456 on PRJ-ABC"
        rows = PatternPIIRedactor.calculate_feature({"a": [text], "b": [text]}, features)

        assert rows[0]["a__pii_redacted"] == "[EMPLOYEE_ID] on PRJ-ABC"
        assert rows[0]["b__pii_redacted"] == "EMP-123456 on [PROJECT]"
        assert PatternPIIRedactor._redact_pii(["EMP-123456"], ["ALL"], "mask") == ["EMP-123456"]
