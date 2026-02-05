"""Exact hash-based deduplication."""

from __future__ import annotations

import hashlib
from typing import Dict, List, Optional

from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys

from rag_integration.feature_groups.rag_pipeline.deduplication.base import BaseDeduplicator


class ExactHashDeduplicator(BaseDeduplicator):
    """
    Exact hash-based deduplicator.

    Uses MD5 hashing for fast exact duplicate detection.
    Only detects exact matches (ignores similarity_threshold).

    Efficient for large datasets where exact duplicates are expected.

    Config-based matching:
        deduplication_method="exact_hash"
    """

    PROPERTY_MAPPING = {
        BaseDeduplicator.DEDUPLICATION_METHOD: {
            "exact_hash": "MD5 hash-based exact duplicate detection",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: True,
        },
        BaseDeduplicator.SIMILARITY_THRESHOLD: {
            "explanation": "Threshold for considering texts as duplicates (0.0-1.0)",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.default: 1.0,
        },
        BaseDeduplicator.KEEP_STRATEGY: {
            **BaseDeduplicator.KEEP_STRATEGIES,
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.default: "first",
        },
        DefaultOptionKeys.in_features: {
            "explanation": "Source feature containing text to deduplicate",
            DefaultOptionKeys.context: True,
        },
    }

    @classmethod
    def _find_duplicates(
        cls,
        texts: List[str],
        threshold: float,
    ) -> List[Optional[int]]:
        """
        Find exact duplicates using MD5 hashing.

        Args:
            texts: List of text strings
            threshold: Ignored for exact hash matching

        Returns:
            List where each element is either None (first occurrence)
            or the index of the first occurrence it duplicates.
        """
        # Map hash -> first index
        hash_to_index: Dict[str, int] = {}
        result: List[Optional[int]] = []

        for i, text in enumerate(texts):
            # Compute hash
            text_hash = hashlib.md5(text.encode("utf-8"), usedforsecurity=False).hexdigest()

            if text_hash in hash_to_index:
                # This is a duplicate
                result.append(hash_to_index[text_hash])
            else:
                # First occurrence
                hash_to_index[text_hash] = i
                result.append(None)

        return result
