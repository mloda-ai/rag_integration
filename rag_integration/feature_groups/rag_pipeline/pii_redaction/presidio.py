"""Presidio-based PII redaction using Microsoft's Presidio library."""

from __future__ import annotations

from typing import List, Optional

from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys

from rag_integration.feature_groups.rag_pipeline.pii_redaction.base import BasePIIRedactor


class PresidioPIIRedactor(BasePIIRedactor):
    """
    Presidio-based PII redactor using Microsoft's Presidio library.

    Uses Presidio Analyzer for advanced PII detection with support for
    multiple languages, custom recognizers, and configurable entity types.

    Requires: pip install presidio-analyzer

    Supported PII Types (maps to Presidio entity types):
        - EMAIL: EMAIL_ADDRESS
        - PHONE: PHONE_NUMBER
        - SSN: US_SSN
        - NAME: PERSON
        - CREDIT_CARD: CREDIT_CARD
        - IP_ADDRESS: IP_ADDRESS
        - ALL: All supported entity types

    Config-based matching:
        redaction_method="presidio"

    Note: Caches the analyzer at class level for performance. Not thread-safe.
    """

    PROPERTY_MAPPING = {
        BasePIIRedactor.REDACTION_METHOD: {
            "presidio": "Microsoft Presidio-based PII detection",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: True,
        },
        BasePIIRedactor.PII_TYPES: {
            "explanation": "List of PII types to redact (EMAIL, PHONE, SSN, NAME, CREDIT_CARD, IP_ADDRESS, ALL)",
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

    # Map our PII types to Presidio entity types
    PII_TYPE_MAPPING = {
        "EMAIL": "EMAIL_ADDRESS",
        "PHONE": "PHONE_NUMBER",
        "SSN": "US_SSN",
        "NAME": "PERSON",
        "CREDIT_CARD": "CREDIT_CARD",
        "IP_ADDRESS": "IP_ADDRESS",
    }

    # All Presidio entities we support
    ALL_ENTITIES = list(PII_TYPE_MAPPING.values())

    _analyzer: Optional[object] = None

    @classmethod
    def _get_analyzer(cls) -> object:
        """Get or create the Presidio analyzer instance."""
        if cls._analyzer is None:
            try:
                from presidio_analyzer import AnalyzerEngine
            except ImportError as e:
                raise ImportError(
                    "presidio-analyzer is required for PresidioPIIRedactor. Install with: pip install presidio-analyzer"
                ) from e

            cls._analyzer = AnalyzerEngine()
        return cls._analyzer

    @classmethod
    def _get_presidio_entities(cls, pii_types: List[str]) -> List[str]:
        """Convert our PII types to Presidio entity types."""
        if "ALL" in pii_types:
            return cls.ALL_ENTITIES

        entities = []
        for pii_type in pii_types:
            if pii_type in cls.PII_TYPE_MAPPING:
                entities.append(cls.PII_TYPE_MAPPING[pii_type])
        return entities if entities else cls.ALL_ENTITIES

    @classmethod
    def _get_replacement(cls, entity_type: str, replacement_strategy: str) -> str:
        """Get the replacement string based on strategy."""
        if replacement_strategy == "type_label":
            # Convert Presidio entity type back to our format
            for our_type, presidio_type in cls.PII_TYPE_MAPPING.items():
                if presidio_type == entity_type:
                    return f"[{our_type}]"
            return f"[{entity_type}]"
        return "[REDACTED]"

    @classmethod
    def _redact_pii(
        cls,
        texts: List[str],
        pii_types: List[str],
        replacement_strategy: str,
    ) -> List[str]:
        """
        Redact PII using Presidio Analyzer.

        Args:
            texts: List of text strings to process
            pii_types: List of PII types to redact
            replacement_strategy: How to replace detected PII

        Returns:
            List of redacted text strings
        """
        analyzer = cls._get_analyzer()
        entities = cls._get_presidio_entities(pii_types)

        result: List[str] = []
        for text in texts:
            if not text:
                result.append(text)
                continue

            # Analyze text for PII
            results = analyzer.analyze(  # type: ignore[attr-defined]
                text=text,
                entities=entities,
                language="en",
            )

            # Sort results by start position (descending) to replace from end
            sorted_results = sorted(results, key=lambda x: x.start, reverse=True)

            # Replace each detected entity
            redacted = text
            for detection in sorted_results:
                replacement = cls._get_replacement(detection.entity_type, replacement_strategy)
                redacted = redacted[: detection.start] + replacement + redacted[detection.end :]

            result.append(redacted)

        return result
