"""Pins which feature groups declare a data classification level and which lower it by masking."""

from __future__ import annotations

import pytest
from mloda.provider import FeatureGroup
from mloda.steward import resolve_feature
from mloda.user import Options, PluginCollector

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
from rag_integration.feature_groups.rag_pipeline.pii_redaction import (
    BasePIIRedactor,
    FullCoveragePatternPIIRedactor,
    FullCoveragePresidioPIIRedactor,
    FullCoverageRegexPIIRedactor,
    FullCoverageSimplePIIRedactor,
    RegexPIIRedactor,
)
from rag_integration.feature_groups.rag_pipeline.vector_store.base import BaseVectorStore
from tests.feature_groups.test_property_mapping_defaults import _all_feature_groups

MASKING_BASES: set[type[FeatureGroup]] = {
    FullCoveragePatternPIIRedactor,
    FullCoveragePresidioPIIRedactor,
    FullCoverageRegexPIIRedactor,
    FullCoverageSimplePIIRedactor,
}
PASS_THROUGH_BASES: set[type[FeatureGroup]] = {
    BasePIIRedactor,
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
    assert not getattr(BasePIIRedactor, "masking", False)
    assert BaseVectorStore.declared_attributes(None) == {CLASSIFICATION_KEY: "public"}


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


@pytest.mark.parametrize(
    ("pii_types", "expected"),
    [
        (None, FullCoverageRegexPIIRedactor),
        (["ALL"], FullCoverageRegexPIIRedactor),
        ("ALL", FullCoverageRegexPIIRedactor),
        (["ALL", "EMAIL"], FullCoverageRegexPIIRedactor),
        (["EMAIL"], RegexPIIRedactor),
        (["EMAIL", "PHONE"], RegexPIIRedactor),
    ],
)
def test_only_full_pii_coverage_resolves_to_the_masking_redactor(pii_types: object, expected: type) -> None:
    """A narrowed ``pii_types`` leaves residual PII, so it must not reach the group that lowers the level."""
    context: dict[str, object] = {"redaction_method": "regex", "in_features": "docs"}
    if pii_types is not None:
        context["pii_types"] = pii_types
    resolved = resolve_feature(
        "docs__pii_redacted",
        options=Options(context=context),
        plugin_collector=PluginCollector.enabled_feature_groups({RegexPIIRedactor, FullCoverageRegexPIIRedactor}),
    )
    assert resolved.error is None, resolved.error
    assert resolved.feature_group is expected
