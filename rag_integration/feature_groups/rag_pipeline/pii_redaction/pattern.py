"""Configurable pattern-based PII redaction."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Pattern

from mloda.user import Feature
from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys

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

    PROPERTY_MAPPING = {
        BasePIIRedactor.REDACTION_METHOD: {
            "pattern": "Custom pattern based detection",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: True,
        },
        BasePIIRedactor.PII_TYPES: {
            "explanation": "List of PII types to redact (EMAIL, PHONE, SSN, NAME, ALL)",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.default: ["ALL"],
        },
        BasePIIRedactor.REPLACEMENT_STRATEGY: {
            **BasePIIRedactor.REPLACEMENT_STRATEGIES,
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.default: "mask",
        },
        DefaultOptionKeys.in_features: {
            "explanation": "Source feature containing text to redact",
            DefaultOptionKeys.context: True,
        },
    }

    # Default patterns (can be extended via options)
    DEFAULT_PATTERNS: Dict[str, str] = {
        "EMAIL": BasePIIRedactor.EMAIL_REGEX,
        "PHONE": BasePIIRedactor.PHONE_REGEX,
        "SSN": BasePIIRedactor.SSN_REGEX,
        "CREDIT_CARD": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
        "IP_ADDRESS": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
    }

    _active_patterns: Dict[str, Pattern[str]] | None = None

    @classmethod
    def _get_patterns(cls, feature: Feature) -> Dict[str, Pattern[str]]:
        """Get compiled patterns including custom ones from feature options."""
        patterns: Dict[str, Pattern[str]] = {}

        for name, pattern_str in cls.DEFAULT_PATTERNS.items():
            patterns[name] = re.compile(pattern_str)

        custom_patterns: Any = feature.options.get("custom_patterns")
        if custom_patterns is None:
            custom_patterns = {}
        for name, pattern_str in custom_patterns.items():
            patterns[name] = re.compile(pattern_str)

        return patterns

    @classmethod
    def calculate_feature(cls, data: List[Dict[str, Any]], features: Any) -> List[Dict[str, Any]]:
        """Extract custom patterns from feature options before redacting."""
        for feature in features.features:
            cls._active_patterns = cls._get_patterns(feature)
        return super().calculate_feature(data, features)

    @classmethod
    def _redact_pii(
        cls,
        texts: List[str],
        pii_types: List[str],
        replacement_strategy: str,
    ) -> List[str]:
        """Redact PII using configurable patterns (default + custom)."""
        if cls._active_patterns is not None:
            patterns = cls._active_patterns
        else:
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
