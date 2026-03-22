"""Regex-based PII redaction."""

from __future__ import annotations

import re
from typing import List

from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys

from rag_integration.feature_groups.rag_pipeline.pii_redaction.base import BasePIIRedactor


class RegexPIIRedactor(BasePIIRedactor):
    """
    Regex-based PII redactor.

    Uses regular expressions to detect and redact common PII patterns.
    No external dependencies required.

    Supported PII Types:
        - EMAIL: Standard email format (user@domain.tld)
        - PHONE: US phone formats (555-123-4567, (555) 123-4567, etc.)
        - SSN: Social Security Numbers (123-45-6789)
        - NAME: Common name patterns (requires word boundaries)

    Config-based matching:
        redaction_method="regex"
    """

    PROPERTY_MAPPING = {
        BasePIIRedactor.REDACTION_METHOD: {
            "regex": "Regex-based PII detection",
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

    # Regex patterns for different PII types
    PATTERNS = {
        "EMAIL": re.compile(BasePIIRedactor.EMAIL_REGEX),
        "PHONE": re.compile(BasePIIRedactor.PHONE_REGEX),
        "SSN": re.compile(BasePIIRedactor.SSN_REGEX),
        "NAME": re.compile(
            r"\b(?:Mr\.|Mrs\.|Ms\.|Dr\.)?\s*"
            r"[A-Z][a-z]+\s+"  # First name
            r"(?:[A-Z]\.?\s+)?"  # Optional middle initial
            r"[A-Z][a-z]+\b"  # Last name
        ),
    }

    @classmethod
    def _redact_pii(
        cls,
        texts: List[str],
        pii_types: List[str],
        replacement_strategy: str,
    ) -> List[str]:
        """
        Redact PII using regex patterns.

        Args:
            texts: List of text strings to process
            pii_types: List of PII types to redact
            replacement_strategy: How to replace detected PII

        Returns:
            List of redacted text strings
        """
        # Determine which patterns to apply
        if "ALL" in pii_types:
            active_types = list(cls.PATTERNS.keys())
        else:
            active_types = [t for t in pii_types if t in cls.PATTERNS]

        result = []
        for text in texts:
            redacted = text
            for pii_type in active_types:
                pattern = cls.PATTERNS[pii_type]
                replacement = cls._get_replacement(pii_type, replacement_strategy)
                redacted = pattern.sub(replacement, redacted)
            result.append(redacted)

        return result
