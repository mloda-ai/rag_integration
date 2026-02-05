"""Configurable pattern-based PII redaction."""

from __future__ import annotations

import re
from typing import Dict, List, Pattern, Any

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
        "EMAIL": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "PHONE": r"(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}",
        "SSN": r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b",
        "CREDIT_CARD": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
        "IP_ADDRESS": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
    }

    @classmethod
    def _get_patterns(cls, feature: Feature) -> Dict[str, Pattern[str]]:
        """Get compiled patterns including custom ones."""
        patterns: Dict[str, Pattern[str]] = {}

        # Add default patterns
        for name, pattern_str in cls.DEFAULT_PATTERNS.items():
            patterns[name] = re.compile(pattern_str)

        # Add custom patterns from options
        custom_patterns: Any = feature.options.get("custom_patterns")
        if custom_patterns is None:
            custom_patterns = {}
        for name, pattern_str in custom_patterns.items():
            try:
                patterns[name] = re.compile(pattern_str)
            except re.error:
                pass  # Skip invalid patterns

        return patterns

    @classmethod
    def _get_replacement(cls, pii_type: str, replacement_strategy: str) -> str:
        """Get the replacement string based on strategy."""
        if replacement_strategy == "type_label":
            return f"[{pii_type}]"
        return "[REDACTED]"

    @classmethod
    def _redact_pii(
        cls,
        texts: List[str],
        pii_types: List[str],
        replacement_strategy: str,
    ) -> List[str]:
        """
        Redact PII using configurable patterns.

        Note: This implementation uses the default patterns.
        Custom patterns are added via calculate_feature override.
        """
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
