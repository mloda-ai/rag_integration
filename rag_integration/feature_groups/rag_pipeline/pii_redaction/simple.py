"""Simple word-list based PII redaction."""

from __future__ import annotations

import re
from typing import List, Set

from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys

from rag_integration.feature_groups.rag_pipeline.pii_redaction.base import BasePIIRedactor


class SimplePIIRedactor(BasePIIRedactor):
    """
    Simple word-list based PII redactor.

    Uses predefined lists of common names to detect person names.
    Combines with basic regex for other PII types.

    Lightweight alternative when full NER is not needed.

    Config-based matching:
        redaction_method="simple"
    """

    PROPERTY_MAPPING = {
        BasePIIRedactor.REDACTION_METHOD: {
            "simple": "Simple word-list based detection",
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

    # Common first names (subset for demonstration)
    COMMON_FIRST_NAMES: Set[str] = {
        "james",
        "john",
        "robert",
        "michael",
        "william",
        "david",
        "richard",
        "joseph",
        "thomas",
        "charles",
        "mary",
        "patricia",
        "jennifer",
        "linda",
        "elizabeth",
        "barbara",
        "susan",
        "jessica",
        "sarah",
        "karen",
        "nancy",
        "alice",
        "bob",
        "jane",
        "joe",
        "jim",
        "tom",
        "bill",
        "mike",
        "steve",
    }

    # Common last names (subset for demonstration)
    COMMON_LAST_NAMES: Set[str] = {
        "smith",
        "johnson",
        "williams",
        "brown",
        "jones",
        "garcia",
        "miller",
        "davis",
        "rodriguez",
        "martinez",
        "wilson",
        "anderson",
        "taylor",
        "thomas",
        "moore",
        "jackson",
        "martin",
        "lee",
        "thompson",
        "white",
        "doe",
        "black",
        "green",
        "king",
        "scott",
        "young",
        "allen",
        "hill",
    }

    # Simple patterns for non-name PII
    EMAIL_PATTERN = re.compile(BasePIIRedactor.EMAIL_REGEX)
    PHONE_PATTERN = re.compile(BasePIIRedactor.PHONE_REGEX)
    SSN_PATTERN = re.compile(BasePIIRedactor.SSN_REGEX)

    @classmethod
    def _redact_names(cls, text: str, replacement: str) -> str:
        """Redact names using word list matching."""
        words = text.split()
        result = []
        i = 0

        while i < len(words):
            word = words[i]
            word_lower = word.lower().strip(".,!?;:'\"")

            # Check for first name followed by last name
            if word_lower in cls.COMMON_FIRST_NAMES and i + 1 < len(words):
                next_word = words[i + 1].lower().strip(".,!?;:'\"")
                if next_word in cls.COMMON_LAST_NAMES:
                    result.append(replacement)
                    i += 2
                    continue

            # Check for standalone common name (with capital letter)
            if word_lower in cls.COMMON_FIRST_NAMES and word[0].isupper():
                result.append(replacement)
                i += 1
                continue

            result.append(word)
            i += 1

        return " ".join(result)

    @classmethod
    def _redact_pii(
        cls,
        texts: List[str],
        pii_types: List[str],
        replacement_strategy: str,
    ) -> List[str]:
        """
        Redact PII using word lists and simple patterns.

        Args:
            texts: List of text strings to process
            pii_types: List of PII types to redact
            replacement_strategy: How to replace detected PII

        Returns:
            List of redacted text strings
        """
        active_types = pii_types if "ALL" not in pii_types else ["EMAIL", "PHONE", "SSN", "NAME"]

        result = []
        for text in texts:
            redacted = text

            if "EMAIL" in active_types:
                replacement = cls._get_replacement("EMAIL", replacement_strategy)
                redacted = cls.EMAIL_PATTERN.sub(replacement, redacted)

            if "PHONE" in active_types:
                replacement = cls._get_replacement("PHONE", replacement_strategy)
                redacted = cls.PHONE_PATTERN.sub(replacement, redacted)

            if "SSN" in active_types:
                replacement = cls._get_replacement("SSN", replacement_strategy)
                redacted = cls.SSN_PATTERN.sub(replacement, redacted)

            if "NAME" in active_types:
                replacement = cls._get_replacement("NAME", replacement_strategy)
                redacted = cls._redact_names(redacted, replacement)

            result.append(redacted)

        return result
