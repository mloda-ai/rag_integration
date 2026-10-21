"""Configurable pattern-based PII redaction."""

from __future__ import annotations

import re
from collections.abc import Mapping
from re import Pattern
from typing import Any, ClassVar

from mloda.provider import DefaultOptionKeys, property_spec
from mloda.user import Feature

from rag_integration.feature_groups.rag_pipeline.pii_redaction.base import BasePIIRedactor


class PatternPIIRedactor(BasePIIRedactor):
    """
    Configurable pattern-based PII redactor.

    Allows users to define custom regex patterns for PII detection.
    Useful for domain-specific PII types.

    Configuration:
        custom_patterns: Dict mapping pattern names to regex strings
            Example: {"EMPLOYEE_ID": r"EMP-\\d{6}"}

    Usage:
        Feature("docs__pii_redacted", Options(context={
            "custom_patterns": {
                "EMPLOYEE_ID": r"EMP-\\d{6}",
                "ACCOUNT_NUM": r"ACC[0-9]{10}",
            }
        }))

    Config-based matching:
        redaction_method="pattern"
    """

    CUSTOM_PATTERNS = "custom_patterns"

    PROPERTY_MAPPING: ClassVar = {
        BasePIIRedactor.REDACTION_METHOD: property_spec(
            "Detector used to locate PII in text",
            strict=True,
            allowed_values={"pattern": "Custom pattern based detection"},
            deferred_binding=True,
        ),
        BasePIIRedactor.PII_TYPES: property_spec(
            "List of PII types to redact (EMAIL, PHONE, SSN, NAME, ALL)", default=["ALL"]
        ),
        BasePIIRedactor.REPLACEMENT_STRATEGY: property_spec(
            "How detected PII is replaced in the redacted text",
            allowed_values=BasePIIRedactor.REPLACEMENT_STRATEGIES,
            default="mask",
        ),
        CUSTOM_PATTERNS: property_spec(
            "Extra PII types as a dict of type name to regex string, added to the defaults", default=None
        ),
        DefaultOptionKeys.in_features: property_spec("Source feature containing text to redact"),
    }

    # Default patterns (can be extended via options)
    DEFAULT_PATTERNS: ClassVar[dict[str, str]] = {
        "EMAIL": BasePIIRedactor.EMAIL_REGEX,
        "PHONE": BasePIIRedactor.PHONE_REGEX,
        "SSN": BasePIIRedactor.SSN_REGEX,
        "CREDIT_CARD": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
        "IP_ADDRESS": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
    }

    @classmethod
    def _get_patterns(cls, feature: Feature) -> dict[str, Pattern[str]]:
        """Get compiled patterns including custom ones from feature options."""
        patterns: dict[str, Pattern[str]] = {}

        for name, pattern_str in cls.DEFAULT_PATTERNS.items():
            patterns[name] = re.compile(pattern_str)

        custom_patterns: Any = feature.options.get(cls.CUSTOM_PATTERNS)
        if custom_patterns is None:
            custom_patterns = {}
        if not isinstance(custom_patterns, Mapping):
            raise TypeError(
                f"{cls.CUSTOM_PATTERNS} must be a dict of type name to regex string, "
                f"got {type(custom_patterns).__name__}."
            )
        for name, pattern_str in custom_patterns.items():
            if not isinstance(name, str) or not isinstance(pattern_str, str):
                raise TypeError(f"{cls.CUSTOM_PATTERNS} entry {name!r} must map a string name to a regex string.")
            try:
                patterns[name] = re.compile(pattern_str)
            except re.error as e:
                raise ValueError(f"{cls.CUSTOM_PATTERNS} entry {name!r} is not a valid regex: {e}") from e

        return patterns

    @classmethod
    def _known_pii_types(cls, feature: Feature) -> set[str]:
        return set(cls._get_patterns(feature)) | {"ALL"}

    @classmethod
    def _redact_texts_for_feature(cls, texts: list[str], feature: Feature) -> list[str]:
        """Redact texts using the per-feature custom patterns."""
        return cls._redact_pii(
            texts,
            cls._get_pii_types(feature),
            cls._get_replacement_strategy(feature),
            patterns=cls._get_patterns(feature),
        )

    @classmethod
    def _redact_pii(
        cls,
        texts: list[str],
        pii_types: list[str],
        replacement_strategy: str,
        patterns: dict[str, Pattern[str]] | None = None,
    ) -> list[str]:
        """Redact PII using the given patterns, or the defaults when none are given."""
        if patterns is None:
            patterns = {name: re.compile(p) for name, p in cls.DEFAULT_PATTERNS.items()}

        if "ALL" in pii_types:
            active_types = list(patterns.keys())
        else:
            active_types = [t for t in pii_types if t in patterns]

        result = []
        for text in texts:
            redacted = text
            for pii_type in active_types:
                pattern = patterns[pii_type]
                replacement = cls._get_replacement(pii_type, replacement_strategy)
                redacted = pattern.sub(replacement, redacted)
            result.append(redacted)

        return result
