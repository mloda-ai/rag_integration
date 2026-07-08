"""Base class for image deduplication feature groups."""

from __future__ import annotations

from typing import Any, Dict, List

from mloda.user import Feature, FeatureName, Options
from mloda.provider import DefaultOptionKeys

from rag_integration.feature_groups.deduplication_base import BaseRowDeduplicator


class BaseImageDeduplicator(BaseRowDeduplicator):
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

    # Image deduplication keeps the largest file from each duplicate group.
    KEEP_LARGEST_STRATEGY = "largest"

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
    def _extract_items(cls, data: List[Dict[str, Any]], feature: Feature) -> List[Any]:
        """Extract image bytes from each row's 'image_data' field."""
        items: List[Any] = []
        for row in data:
            image_data = row.get("image_data", b"")
            if not isinstance(image_data, bytes):
                image_data = bytes(image_data) if image_data else b""
            items.append(image_data)
        return items
