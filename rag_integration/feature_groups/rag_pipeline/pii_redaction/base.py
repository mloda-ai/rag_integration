"""Base class for PII redaction feature groups."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, List, Set, Type, Union

from mloda.provider import FeatureGroup, ComputeFramework, FeatureSet
from mloda.provider import FeatureChainParserMixin
from mloda.user import Feature
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
    PythonDictFramework,
)
from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys


class BasePIIRedactor(FeatureChainParserMixin, FeatureGroup):
    """
    Base class for PII redaction feature groups.

    Redacts personally identifiable information from text documents.

    Feature Naming Pattern:
        {in_feature}__pii_redacted

    Examples:
        - docs__pii_redacted
        - raw_text__pii_redacted

    Supported PII Types:
        - EMAIL: Email addresses
        - PHONE: Phone numbers
        - SSN: Social security numbers
        - NAME: Person names
        - ALL: All supported PII types

    ## Configuration-Based Creation

    Uses Options with proper group/context parameter separation:

    ```python
    feature = Feature(
        name="my_redacted",
        options=Options(
            context={
                "redaction_method": "regex",
                DefaultOptionKeys.in_features: "docs",
            }
        )
    )
    ```
    """

    # Configuration keys
    PII_TYPES = "pii_types"
    REPLACEMENT_STRATEGY = "replacement_strategy"

    # Discriminator key for config-based feature matching
    REDACTION_METHOD = "redaction_method"

    # Supported redaction methods (implementations must define which they handle)
    REDACTION_METHODS = {
        "regex": "Regex-based PII detection",
        "simple": "Simple word-list based detection",
        "pattern": "Custom pattern based detection",
    }

    # Supported PII types
    SUPPORTED_PII_TYPES = {
        "EMAIL": "Email addresses",
        "PHONE": "Phone numbers",
        "SSN": "Social security numbers",
        "NAME": "Person names",
        "ALL": "All supported PII types",
    }

    # Replacement strategies
    REPLACEMENT_STRATEGIES = {
        "mask": "Replace with [REDACTED]",
        "type_label": "Replace with [PII_TYPE]",
    }

    # Shared regex pattern strings for common PII types
    EMAIL_REGEX = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
    PHONE_REGEX = r"(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}"
    SSN_REGEX = r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b"

    @classmethod
    def _get_replacement(cls, pii_type: str, replacement_strategy: str) -> str:
        """Get the replacement string based on strategy."""
        if replacement_strategy == "type_label":
            return f"[{pii_type}]"
        return "[REDACTED]"

    # Pattern for feature chain parsing
    PREFIX_PATTERN = r".*__pii_redacted$"

    MIN_IN_FEATURES = 1
    MAX_IN_FEATURES = 1

    PROPERTY_MAPPING = {
        REDACTION_METHOD: {
            **REDACTION_METHODS,
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: True,
        },
        PII_TYPES: {
            "explanation": "List of PII types to redact (EMAIL, PHONE, SSN, NAME, ALL)",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.default: ["ALL"],
        },
        REPLACEMENT_STRATEGY: {
            **REPLACEMENT_STRATEGIES,
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.default: "mask",
        },
        DefaultOptionKeys.in_features: {
            "explanation": "Source feature containing text to redact",
            DefaultOptionKeys.context: True,
        },
    }

    @classmethod
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFramework]]]:
        return {PythonDictFramework}

    @classmethod
    def _get_source_feature_name(cls, feature: Feature) -> str:
        """Extract source feature name from the feature."""
        source_features = cls._extract_source_features(feature)
        return source_features[0]

    @classmethod
    def _get_pii_types(cls, feature: Feature) -> List[str]:
        """Get PII types to redact from feature options."""
        pii_types = feature.options.get(cls.PII_TYPES)
        if pii_types is None:
            pii_types = ["ALL"]
        if not isinstance(pii_types, list):
            pii_types = [pii_types]
        return pii_types

    @classmethod
    def _get_replacement_strategy(cls, feature: Feature) -> str:
        """Get replacement strategy from feature options."""
        strategy = feature.options.get(cls.REPLACEMENT_STRATEGY)
        return str(strategy) if strategy is not None else "mask"

    @classmethod
    @abstractmethod
    def _redact_pii(
        cls,
        texts: List[str],
        pii_types: List[str],
        replacement_strategy: str,
    ) -> List[str]:
        """
        Redact PII from a list of texts.

        Args:
            texts: List of text strings to process
            pii_types: List of PII types to redact
            replacement_strategy: How to replace detected PII

        Returns:
            List of redacted text strings
        """
        ...

    @classmethod
    def calculate_feature(cls, data: List[Dict[str, Any]], features: FeatureSet) -> List[Dict[str, Any]]:
        """Perform PII redaction on the source feature."""
        for feature in features.features:
            source_feature = cls._get_source_feature_name(feature)
            pii_types = cls._get_pii_types(feature)
            replacement_strategy = cls._get_replacement_strategy(feature)

            # Extract texts from source feature (use 'text' field if source is root)
            texts = []
            for row in data:
                if source_feature in row:
                    texts.append(str(row[source_feature]))
                elif "text" in row:
                    texts.append(str(row["text"]))
                else:
                    texts.append("")

            # Redact PII
            redacted_texts = cls._redact_pii(texts, pii_types, replacement_strategy)

            # Add results to data
            for i, row in enumerate(data):
                row[feature.get_name()] = redacted_texts[i]

        return data
