"""Base test class for text PII redaction feature groups."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Type

from mloda.user import Options

from rag_integration.feature_groups.rag_pipeline.pii_redaction.base import BasePIIRedactor


class TextPIIRedactionTestBase(ABC):
    """Abstract base providing shared tests for all text PII redaction implementations."""

    @property
    @abstractmethod
    def redactor_class(self) -> Type[BasePIIRedactor]: ...

    def test_redact_email(self) -> None:
        """Should redact email addresses."""
        result = self.redactor_class._redact_pii(["Contact me at test@example.com for details."], ["EMAIL"], "mask")
        assert "test@example.com" not in result[0]
        assert "[REDACTED]" in result[0]

    def test_redact_phone(self) -> None:
        """Should redact phone numbers."""
        result = self.redactor_class._redact_pii(["Call me at 555-123-4567."], ["PHONE"], "mask")
        assert "555-123-4567" not in result[0]
        assert "[REDACTED]" in result[0]

    def test_redact_ssn(self) -> None:
        """Should redact SSN."""
        result = self.redactor_class._redact_pii(["My SSN is 123-45-6789."], ["SSN"], "mask")
        assert "123-45-6789" not in result[0]
        assert "[REDACTED]" in result[0]

    def test_type_label_replacement(self) -> None:
        """Should use type labels when configured."""
        result = self.redactor_class._redact_pii(["Email: test@example.com"], ["EMAIL"], "type_label")
        assert "[EMAIL]" in result[0]

    def test_feature_matching_pattern(self) -> None:
        """Should match pii_redacted features and reject others."""
        assert self.redactor_class.match_feature_group_criteria("docs__pii_redacted", Options())
        assert self.redactor_class.match_feature_group_criteria("text__pii_redacted", Options())
        assert not self.redactor_class.match_feature_group_criteria("docs__chunked", Options())
