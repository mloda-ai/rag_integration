"""N-gram based fuzzy deduplication."""

from __future__ import annotations

from typing import List, Optional, Set

from mloda.provider import DefaultOptionKeys, property_spec

from rag_integration.feature_groups.rag_pipeline.deduplication.base import BaseDeduplicator


class NGramDeduplicator(BaseDeduplicator):
    """
    N-gram based fuzzy deduplicator.

    Uses character n-gram Jaccard similarity to detect near-duplicates.
    Configurable similarity threshold.

    Good for detecting paraphrased or slightly modified duplicates.

    Config-based matching:
        deduplication_method="ngram"
    """

    PROPERTY_MAPPING = {
        BaseDeduplicator.DEDUPLICATION_METHOD: property_spec(
            "Algorithm used to detect duplicate texts",
            strict=True,
            allowed_values={"ngram": "N-gram Jaccard similarity based detection"},
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

    # Default n-gram size
    NGRAM_SIZE = 3

    @classmethod
    def _get_ngrams(cls, text: str, n: int = 3) -> Set[str]:
        """Extract character n-grams from text."""
        text = text.lower().strip()
        if len(text) < n:
            return {text}
        return {text[i : i + n] for i in range(len(text) - n + 1)}

    @classmethod
    def _jaccard_similarity(cls, set1: Set[str], set2: Set[str]) -> float:
        """Compute Jaccard similarity between two sets."""
        if not set1 or not set2:
            return 0.0
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0

    @classmethod
    def _find_duplicates(
        cls,
        texts: List[str],
        threshold: float,
    ) -> List[Optional[int]]:
        """
        Find near-duplicates using n-gram Jaccard similarity.

        Args:
            texts: List of text strings
            threshold: Similarity threshold (0.0-1.0)

        Returns:
            List where each element is either None (not a duplicate)
            or the index of the text it duplicates.
        """
        # Precompute n-grams for all texts
        ngrams_list: List[Set[str]] = [cls._get_ngrams(text, cls.NGRAM_SIZE) for text in texts]

        # Track canonical texts (first occurrence of each unique text)
        canonical_indices: List[int] = []
        result: List[Optional[int]] = []

        for i, ngrams in enumerate(ngrams_list):
            found_duplicate = False

            # Compare against all canonical texts
            for canonical_idx in canonical_indices:
                canonical_ngrams = ngrams_list[canonical_idx]
                similarity = cls._jaccard_similarity(ngrams, canonical_ngrams)

                if similarity >= threshold:
                    result.append(canonical_idx)
                    found_duplicate = True
                    break

            if not found_duplicate:
                canonical_indices.append(i)
                result.append(None)

        return result
