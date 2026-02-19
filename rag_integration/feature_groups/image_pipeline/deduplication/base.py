"""Base class for image deduplication feature groups."""

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


class BaseImageDeduplicator(FeatureChainParserMixin, FeatureGroup):
    """
    Base class for image deduplication feature groups.

    Removes duplicate or near-duplicate images from the pipeline.

    Feature Naming Pattern:
        {in_feature}__deduped

    Examples:
        - image_docs__pii_redacted__preprocessed__deduped

    Note: Deduplication can reduce the number of rows by removing duplicates.
    It adds metadata to track duplicate relationships.

    ## Configuration-Based Creation

    Uses Options with proper group/context parameter separation:

    ```python
    feature = Feature(
        name="my_deduped",
        options=Options(
            context={
                "image_deduplication_method": "exact_hash",
                DefaultOptionKeys.in_features: "image_docs",
            }
        )
    )
    ```
    """

    # Configuration keys
    SIMILARITY_THRESHOLD = "similarity_threshold"
    KEEP_STRATEGY = "keep_strategy"

    # Discriminator key for config-based feature matching
    IMAGE_DEDUPLICATION_METHOD = "image_deduplication_method"

    # Supported deduplication methods
    DEDUPLICATION_METHODS = {
        "exact_hash": "MD5 hash-based exact duplicate detection",
        "phash": "Perceptual hash-based near-duplicate detection",
        "dhash": "Difference hash-based near-duplicate detection",
    }

    # Keep strategies
    KEEP_STRATEGIES = {
        "first": "Keep the first occurrence",
        "largest": "Keep the largest file size",
        "all_unique": "Mark duplicates but keep all rows",
    }

    PREFIX_PATTERN = r".*__deduped$"

    MIN_IN_FEATURES = 1
    MAX_IN_FEATURES = 1

    PROPERTY_MAPPING = {
        IMAGE_DEDUPLICATION_METHOD: {
            **DEDUPLICATION_METHODS,
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: True,
        },
        SIMILARITY_THRESHOLD: {
            "explanation": "Threshold for considering images as duplicates (0.0-1.0)",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.default: 1.0,
        },
        KEEP_STRATEGY: {
            **KEEP_STRATEGIES,
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.default: "first",
        },
        DefaultOptionKeys.in_features: {
            "explanation": "Source feature containing images to deduplicate",
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
        """Match feature using mixin logic, catching ValueError for strict validation."""
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
        image_data_list: List[bytes],
        threshold: float,
    ) -> List[Optional[int]]:
        """
        Find duplicates in a list of images.

        Args:
            image_data_list: List of image bytes
            threshold: Similarity threshold (0.0-1.0)

        Returns:
            List where each element is either None (not a duplicate) or
            the index of the image it duplicates.
        """
        ...

    @classmethod
    def calculate_feature(cls, data: List[Dict[str, Any]], features: FeatureSet) -> List[Dict[str, Any]]:
        """Perform deduplication on images."""
        for feature in features.features:
            cls._get_source_feature_name(feature)
            threshold = cls._get_similarity_threshold(feature)
            keep_strategy = cls._get_keep_strategy(feature)
            feature_name = feature.get_name()

            # Extract image data
            image_data_list = []
            for row in data:
                image_data = row.get("image_data", b"")
                if not isinstance(image_data, bytes):
                    image_data = bytes(image_data) if image_data else b""
                image_data_list.append(image_data)

            # Find duplicates
            duplicate_of = cls._find_duplicates(image_data_list, threshold)

            # Add metadata and filter based on keep strategy
            result = []
            for i, row in enumerate(data):
                new_row = row.copy()
                new_row["is_duplicate"] = duplicate_of[i] is not None
                new_row["duplicate_of"] = duplicate_of[i]
                new_row[feature_name] = image_data_list[i]

                if keep_strategy == "all_unique":
                    result.append(new_row)
                elif keep_strategy == "first" and duplicate_of[i] is None:
                    result.append(new_row)
                elif keep_strategy == "largest":
                    result.append(new_row)

            if keep_strategy == "largest":
                result = cls._keep_largest(result)

            return result

        return data

    @classmethod
    def _keep_largest(cls, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Keep only the largest file in each duplicate group."""
        groups: Dict[int, List[int]] = {}
        for i, row in enumerate(data):
            dup_of = row.get("duplicate_of")
            key = dup_of if dup_of is not None else i
            if key not in groups:
                groups[key] = []
            groups[key].append(i)

        keep_indices = set()
        for indices in groups.values():
            largest_idx = max(indices, key=lambda idx: len(data[idx].get("image_data", b"")))
            keep_indices.add(largest_idx)

        return [data[i] for i in sorted(keep_indices)]
