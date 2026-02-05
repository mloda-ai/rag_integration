"""Base class for deduplication feature groups."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, List, Optional, Set, Type, Union

from mloda.provider import FeatureGroup, ComputeFramework, FeatureSet
from mloda.provider import FeatureChainParserMixin
from mloda.user import Feature, FeatureName, Options
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
    PythonDictFramework,
)
from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys


class BaseDeduplicator(FeatureChainParserMixin, FeatureGroup):
    """
    Base class for deduplication feature groups.

    Removes duplicate or near-duplicate text chunks.

    Feature Naming Pattern:
        {in_feature}__deduped

    Examples:
        - docs__pii_redacted__chunked__deduped

    Note: Deduplication can reduce the number of rows by removing duplicates.
    It adds metadata to track duplicate relationships.

    ## Configuration-Based Creation

    Uses Options with proper group/context parameter separation:

    ```python
    feature = Feature(
        name="my_deduped",
        options=Options(
            context={
                "deduplication_method": "exact_hash",
                DefaultOptionKeys.in_features: "docs",
            }
        )
    )
    ```
    """

    # Configuration keys
    SIMILARITY_THRESHOLD = "similarity_threshold"
    KEEP_STRATEGY = "keep_strategy"

    # Discriminator key for config-based feature matching
    DEDUPLICATION_METHOD = "deduplication_method"

    # Supported deduplication methods (implementations must define which they handle)
    DEDUPLICATION_METHODS = {
        "exact_hash": "MD5 hash-based exact duplicate detection",
        "normalized": "Normalized text hash-based detection",
        "ngram": "N-gram Jaccard similarity based detection",
    }

    # Keep strategies
    KEEP_STRATEGIES = {
        "first": "Keep the first occurrence",
        "longest": "Keep the longest text",
        "all_unique": "Mark duplicates but keep all rows",
    }

    PREFIX_PATTERN = r".*__deduped$"

    MIN_IN_FEATURES = 1
    MAX_IN_FEATURES = 1

    PROPERTY_MAPPING = {
        DEDUPLICATION_METHOD: {
            **DEDUPLICATION_METHODS,
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: True,
        },
        SIMILARITY_THRESHOLD: {
            "explanation": "Threshold for considering texts as duplicates (0.0-1.0)",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.default: 1.0,  # Exact match by default
        },
        KEEP_STRATEGY: {
            **KEEP_STRATEGIES,
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.default: "first",
        },
        DefaultOptionKeys.in_features: {
            "explanation": "Source feature containing text to deduplicate",
            DefaultOptionKeys.context: True,
        },
    }

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: str | FeatureName,
        options: Options,
        data_access_collection: Any = None,
    ) -> bool:
        """
        Match feature using mixin logic, catching ValueError for strict validation.

        When multiple subclasses have different discriminator values in their
        PROPERTY_MAPPING, strict validation raises ValueError for non-matching
        subclasses. We catch this to return False, allowing other subclasses to match.
        """
        try:
            return bool(super().match_feature_group_criteria(feature_name, options, data_access_collection))
        except ValueError:
            return False

    @classmethod
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFramework]]]:
        return {PythonDictFramework}

    @classmethod
    def _get_source_feature_name(cls, feature: Feature) -> str:
        """Extract source feature name from the feature."""
        source_features = cls._extract_source_features(feature)
        return source_features[0]

    @classmethod
    def _get_similarity_threshold(cls, feature: Feature) -> float:
        """Get similarity threshold from feature options."""
        threshold = feature.options.get(cls.SIMILARITY_THRESHOLD)
        return float(threshold) if threshold is not None else 1.0

    @classmethod
    def _get_keep_strategy(cls, feature: Feature) -> str:
        """Get keep strategy from feature options."""
        strategy = feature.options.get(cls.KEEP_STRATEGY)
        return str(strategy) if strategy is not None else "first"

    @classmethod
    @abstractmethod
    def _find_duplicates(
        cls,
        texts: List[str],
        threshold: float,
    ) -> List[Optional[int]]:
        """
        Find duplicates in a list of texts.

        Args:
            texts: List of text strings
            threshold: Similarity threshold (0.0-1.0)

        Returns:
            List where each element is either None (not a duplicate) or
            the index of the text it duplicates.
        """
        ...

    @classmethod
    def calculate_feature(cls, data: List[Dict[str, Any]], features: FeatureSet) -> List[Dict[str, Any]]:
        """Perform deduplication on the source feature."""
        for feature in features.features:
            source_feature = cls._get_source_feature_name(feature)
            threshold = cls._get_similarity_threshold(feature)
            keep_strategy = cls._get_keep_strategy(feature)
            feature_name = feature.get_name()

            # Extract texts from source feature
            texts = []
            for row in data:
                if source_feature in row:
                    texts.append(str(row[source_feature]))
                elif "text" in row:
                    texts.append(str(row["text"]))
                else:
                    texts.append("")

            # Find duplicates
            duplicate_of = cls._find_duplicates(texts, threshold)

            # Add metadata and filter based on keep strategy
            result = []
            for i, row in enumerate(data):
                new_row = row.copy()
                new_row["is_duplicate"] = duplicate_of[i] is not None
                new_row["duplicate_of"] = duplicate_of[i]
                new_row[feature_name] = texts[i]

                if keep_strategy == "all_unique":
                    result.append(new_row)
                elif keep_strategy == "first" and duplicate_of[i] is None:
                    result.append(new_row)
                elif keep_strategy == "longest":
                    result.append(new_row)

            if keep_strategy == "longest":
                result = cls._keep_longest(result, feature_name)

            return result

        return data

    @classmethod
    def _keep_longest(cls, data: List[Dict[str, Any]], feature_name: str) -> List[Dict[str, Any]]:
        """Keep only the longest text in each duplicate group."""
        # Group by duplicate relationship
        groups: Dict[int, List[int]] = {}
        for i, row in enumerate(data):
            dup_of = row.get("duplicate_of")
            key = dup_of if dup_of is not None else i
            if key not in groups:
                groups[key] = []
            groups[key].append(i)

        # Keep longest from each group
        keep_indices = set()
        for indices in groups.values():
            longest_idx = max(indices, key=lambda i: len(str(data[i].get(feature_name, ""))))
            keep_indices.add(longest_idx)

        return [data[i] for i in sorted(keep_indices)]
