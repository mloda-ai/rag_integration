"""Normalized text deduplication."""

from __future__ import annotations

import hashlib
import re
from typing import Dict, List, Optional

from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys

from rag_integration.feature_groups.rag_pipeline.deduplication.base import BaseDeduplicator


class NormalizedDeduplicator(BaseDeduplicator):
    """
    Normalized text deduplicator.

    Normalizes whitespace and case before comparison.
    Detects near-exact duplicates that differ only in formatting.

    Good for detecting duplicates with minor whitespace differences.

    Config-based matching:
        deduplication_method="normalized"
    """

    PROPERTY_MAPPING = {
        BaseDeduplicator.DEDUPLICATION_METHOD: {
            "normalized": "Normalized text hash-based detection",
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
    def _normalize_text(cls, text: str) -> str:
        """Normalize text for comparison."""
        # Lowercase
        normalized = text.lower()
        # Normalize whitespace (collapse multiple spaces, strip)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized

    @classmethod
    def _find_duplicates(
        cls,
        texts: List[str],
        threshold: float,
    ) -> List[Optional[int]]:
        """
        Find duplicates after normalizing text.

        Args:
            texts: List of text strings
            threshold: Ignored for normalized matching

        Returns:
            List where each element is either None (first occurrence)
            or the index of the first occurrence it duplicates.
        """
        # Map normalized hash -> first index
        hash_to_index: Dict[str, int] = {}
        result: List[Optional[int]] = []

        for i, text in enumerate(texts):
            # Normalize and hash
            normalized = cls._normalize_text(text)
            text_hash = hashlib.md5(normalized.encode("utf-8"), usedforsecurity=False).hexdigest()

            if text_hash in hash_to_index:
                result.append(hash_to_index[text_hash])
            else:
                hash_to_index[text_hash] = i
                result.append(None)

        return result
