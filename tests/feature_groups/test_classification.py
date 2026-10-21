"""Pins which feature groups declare a data classification level and which lower it by masking."""

from __future__ import annotations

from mloda.provider import FeatureGroup

from rag_integration.feature_groups.classification import (
    CLASSIFICATION_KEY,
    PassThroughClassification,
    RedactedClassification,
)
from rag_integration.feature_groups.deduplication_base import BaseRowDeduplicator
from rag_integration.feature_groups.image_pipeline.embedding.base import BaseImageEmbedder
from rag_integration.feature_groups.image_pipeline.pii_redaction.base import BaseImagePIIRedactor
from rag_integration.feature_groups.image_pipeline.preprocessing.base import BaseImagePreprocessor
from rag_integration.feature_groups.rag_pipeline.chunking.base import BaseChunker
from rag_integration.feature_groups.rag_pipeline.embedding.base import BaseEmbedder
from rag_integration.feature_groups.rag_pipeline.pii_redaction.base import BasePIIRedactor
from rag_integration.feature_groups.rag_pipeline.vector_store.base import BaseVectorStore
from tests.feature_groups.test_property_mapping_defaults import _all_feature_groups

# The registry extenders' levels, least to most restrictive.
LEVELS = ("public", "internal", "confidential", "restricted", "pii")

MASKING_BASES: set[type[FeatureGroup]] = {BasePIIRedactor}
PASS_THROUGH_BASES: set[type[FeatureGroup]] = {
    BaseImagePIIRedactor,
    BaseChunker,
    BaseRowDeduplicator,
    BaseEmbedder,
    BaseImageEmbedder,
    BaseImagePreprocessor,
    BaseVectorStore,
}


def test_stage_bases_declare_their_classification() -> None:
    assert all(issubclass(base, RedactedClassification) for base in MASKING_BASES)
    assert all(issubclass(base, PassThroughClassification) for base in PASS_THROUGH_BASES)
    assert not getattr(BaseImagePIIRedactor, "masking", False)


def test_every_feature_group_declares_a_valid_level_or_none() -> None:
    """Read like the extenders do, with ``features=None``; sources, retrievers and connectors stay undeclared."""
    for group in _all_feature_groups():
        declared = group.declared_attributes(None)
        level = declared.get(CLASSIFICATION_KEY)
        if issubclass(group, RedactedClassification):
            assert getattr(group, "masking", None) is True, group
            assert level == "confidential", group
        elif issubclass(group, PassThroughClassification):
            assert not getattr(group, "masking", False), group
            assert level == "public", group
        else:
            assert level is None, group
        assert level is None or level in LEVELS, group
