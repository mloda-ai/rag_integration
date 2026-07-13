"""Normalized text deduplication."""

from __future__ import annotations

import hashlib
import re
from typing import Dict, List, Optional

from mloda.provider import DefaultOptionKeys, property_spec

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
        BaseDeduplicator.DEDUPLICATION_METHOD: property_spec(
            "Algorithm used to detect duplicate texts",
            strict=True,
            allowed_values={"normalized": "Normalized text hash-based detection"},
        ),
        BaseDeduplicator.SIMILARITY_THRESHOLD: property_spec(
            "Threshold for considering texts as duplicates (0.0-1.0)", default=1.0
        ),
        BaseDeduplicator.KEEP_STRATEGY: property_spec(
            "How a group of detected duplicates is resolved",
            allowed_values=BaseDeduplicator.KEEP_STRATEGIES,
            default="first",
        ),
        DefaultOptionKeys.in_features: property_spec("Source feature containing text to deduplicate"),
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
