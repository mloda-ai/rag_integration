"""Image deduplication feature groups."""

from rag_integration.feature_groups.image_pipeline.deduplication.base import BaseImageDeduplicator
from rag_integration.feature_groups.image_pipeline.deduplication.exact_hash import ExactHashImageDeduplicator
from rag_integration.feature_groups.image_pipeline.deduplication.phash import PerceptualHashImageDeduplicator
from rag_integration.feature_groups.image_pipeline.deduplication.dhash import DifferenceHashImageDeduplicator

__all__ = [
    "BaseImageDeduplicator",
    "ExactHashImageDeduplicator",
    "PerceptualHashImageDeduplicator",
    "DifferenceHashImageDeduplicator",
]
