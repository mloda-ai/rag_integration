"""Base class for deduplication feature groups."""

from __future__ import annotations

from typing import Any, Dict, List

from mloda.user import Feature
from mloda.provider import DefaultOptionKeys, property_spec

from rag_integration.feature_groups.deduplication_base import BaseRowDeduplicator


class BaseDeduplicator(BaseRowDeduplicator):
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
        DEDUPLICATION_METHOD: property_spec(
            "Algorithm used to detect duplicate texts", strict=True, allowed_values=DEDUPLICATION_METHODS
        ),
        SIMILARITY_THRESHOLD: property_spec("Threshold for considering texts as duplicates (0.0-1.0)", default=1.0),
        KEEP_STRATEGY: property_spec(
            "How a group of detected duplicates is resolved", allowed_values=KEEP_STRATEGIES, default="first"
        ),
        DefaultOptionKeys.in_features: property_spec("Source feature containing text to deduplicate"),
    }

    @classmethod
    def _extract_items(cls, data: List[Dict[str, Any]], feature: Feature) -> List[Any]:
        """Extract text from each row's source feature, falling back to the 'text' field."""
        source_feature = cls._get_source_feature_name(feature)
        items: List[Any] = []
        for row in data:
            if source_feature in row:
                items.append(str(row[source_feature]))
            elif "text" in row:
                items.append(str(row["text"]))
            else:
                items.append("")
        return items
