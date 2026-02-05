"""Deduplication feature groups."""

from rag_integration.feature_groups.rag_pipeline.deduplication.base import BaseDeduplicator
from rag_integration.feature_groups.rag_pipeline.deduplication.exact_hash import ExactHashDeduplicator
from rag_integration.feature_groups.rag_pipeline.deduplication.normalized import NormalizedDeduplicator
from rag_integration.feature_groups.rag_pipeline.deduplication.ngram import NGramDeduplicator

__all__ = [
    "BaseDeduplicator",
    "ExactHashDeduplicator",
    "NormalizedDeduplicator",
    "NGramDeduplicator",
]
