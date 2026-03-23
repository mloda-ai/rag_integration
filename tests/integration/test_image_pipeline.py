"""
Integration tests for the Image Pipeline.

Mirrors the RAG pipeline integration test pattern using mlodaAPI.run_all()
with multiple features per call for efficiency.
"""

from __future__ import annotations

import math

from typing import Any, Dict, List, Set, Type

from mloda.user import mlodaAPI, PluginCollector, Domain, Feature, Options
from mloda.provider import DataCreator, FeatureGroup
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
    PythonDictFramework,
)
from tests.integration.helpers import flatten_result, get_results_by_feature

from rag_integration.feature_groups.image_pipeline import (
    SolidFillPIIRedactor,
    ResizePreprocessor,
    NormalizePreprocessor,
    ThumbnailPreprocessor,
    ExactHashImageDeduplicator,
    MockImageEmbedder,
    HashImageEmbedder,
)


# =============================================================================
# Test Data
# =============================================================================


def _create_test_image_bytes(color: tuple[int, int, int] = (255, 0, 0), size: tuple[int, int] = (64, 64)) -> bytes:
    """Create a test PNG image as bytes."""
    try:
        from PIL import Image
    except ImportError:
        # Fallback: return synthetic bytes that differ by color
        return bytes([color[0], color[1], color[2]] * (size[0] * size[1]))

    import io

    img = Image.new("RGB", size, color=color)
    output = io.BytesIO()
    img.save(output, format="PNG")
    return output.getvalue()


SAMPLE_IMAGES = [
    {
        "image_id": "img_001",
        "image_data": _create_test_image_bytes((255, 0, 0)),
        "format": "png",
    },
    {
        "image_id": "img_002",
        "image_data": _create_test_image_bytes((0, 255, 0)),
        "format": "png",
    },
    {
        "image_id": "img_003",
        "image_data": _create_test_image_bytes((0, 0, 255)),
        "format": "png",
    },
    {
        "image_id": "img_004",
        "image_data": _create_test_image_bytes((255, 0, 0)),  # Exact duplicate of img_001
        "format": "png",
    },
]


# =============================================================================
# DataCreator for test data
# =============================================================================


class MockImageDataCreator(FeatureGroup):
    """DataCreator that provides mock images for testing."""

    @classmethod
    def input_data(cls) -> DataCreator:
        return DataCreator({"image_docs"})

    @classmethod
    def match_feature_group_criteria(cls, feature_name: Any, options: Any, data_access_collection: Any = None) -> bool:
        return str(feature_name) == "image_docs"

    @classmethod
    def compute_framework_rule(cls) -> Set[Type[Any]]:
        return {PythonDictFramework}

    @classmethod
    def calculate_feature(cls, data: Any, features: Any) -> List[Dict[str, Any]]:
        return [
            {
                "image_docs": img["image_data"],
                "image_id": img["image_id"],
                "image_data": img["image_data"],
                "format": img["format"],
            }
            for img in SAMPLE_IMAGES
        ]


def get_test_providers() -> Set[Type[FeatureGroup]]:
    return {
        MockImageDataCreator,
        SolidFillPIIRedactor,
        ResizePreprocessor,
        ExactHashImageDeduplicator,
        MockImageEmbedder,
    }


# =============================================================================
# Integration Test: Full Pipeline with Multiple Features
# =============================================================================


class TestFullImagePipelineIntegration:
    """Test the complete image pipeline requesting multiple features at once."""

    def test_pipeline_stages_in_single_call(self) -> None:
        """
        Request all pipeline stages in one call and verify each stage.

        Tests:
        - image_docs: raw images loaded (4 images)
        - image_docs__pii_redacted: PII regions redacted (row count preserved)
        - image_docs__pii_redacted__preprocessed: images preprocessed (row count preserved)
        - image_docs__pii_redacted__preprocessed__deduped: duplicates removed (row count decreases)
        - image_docs__pii_redacted__preprocessed__deduped__embedded: embeddings generated (unit normalized)
        """
        feature_names = [
            "image_docs",
            "image_docs__pii_redacted",
            "image_docs__pii_redacted__preprocessed",
            "image_docs__pii_redacted__preprocessed__deduped",
            "image_docs__pii_redacted__preprocessed__deduped__embedded",
        ]

        raw_result = mlodaAPI.run_all(
            features=list(feature_names),
            compute_frameworks={PythonDictFramework},
            plugin_collector=PluginCollector.enabled_feature_groups(get_test_providers()),
        )

        results = get_results_by_feature(raw_result, feature_names)

        # 1. image_docs: should have raw images
        docs_rows = results["image_docs"]
        assert len(docs_rows) == 4, f"Should have 4 images, got {len(docs_rows)}"

        # 2. image_docs__pii_redacted: should preserve row count
        pii_rows = results["image_docs__pii_redacted"]
        assert len(pii_rows) == 4, "PII redaction should preserve row count"

        # 3. image_docs__pii_redacted__preprocessed: should preserve row count
        preproc_rows = results["image_docs__pii_redacted__preprocessed"]
        assert len(preproc_rows) == 4, "Preprocessing should preserve row count"

        # 4. image_docs__pii_redacted__preprocessed__deduped: dedup reduces rows
        deduped_rows = results["image_docs__pii_redacted__preprocessed__deduped"]
        # We have img_001 and img_004 as duplicates, so dedup should reduce count
        assert len(deduped_rows) < len(preproc_rows), "Dedup should reduce row count"

        # 5. image_docs__pii_redacted__preprocessed__deduped__embedded: embeddings
        embedded_rows = results["image_docs__pii_redacted__preprocessed__deduped__embedded"]
        assert len(embedded_rows) == len(deduped_rows), "Embedding should preserve row count"

        for row in embedded_rows:
            embedding = row.get("image_docs__pii_redacted__preprocessed__deduped__embedded")
            assert isinstance(embedding, list), "Embedding should be a list"
            assert len(embedding) > 0, "Embedding should have dimensions"
            assert all(isinstance(x, float) for x in embedding), "Embedding values should be floats"

            # Check unit normalization
            magnitude = math.sqrt(sum(x * x for x in embedding))
            assert abs(magnitude - 1.0) < 0.001, f"Embedding should be unit length, got {magnitude}"


# =============================================================================
# Domain-specific provider sets for testing multiple configs in one API call
# =============================================================================


def make_image_domain_providers(
    domain_name: str,
    pii_redactor: type,
    preprocessor: type,
    deduplicator: type,
    embedder: type,
) -> Set[Type[FeatureGroup]]:
    """Factory to create a complete image provider set with a specific domain."""

    class DomainDataCreator(MockImageDataCreator):
        @classmethod
        def get_domain(cls) -> Domain:
            return Domain(domain_name)

    class DomainPIIRedactor(pii_redactor):  # type: ignore[misc]
        @classmethod
        def get_domain(cls) -> Domain:
            return Domain(domain_name)

    class DomainPreprocessor(preprocessor):  # type: ignore[misc]
        @classmethod
        def get_domain(cls) -> Domain:
            return Domain(domain_name)

    class DomainDeduplicator(deduplicator):  # type: ignore[misc]
        @classmethod
        def get_domain(cls) -> Domain:
            return Domain(domain_name)

    class DomainEmbedder(embedder):  # type: ignore[misc]
        @classmethod
        def get_domain(cls) -> Domain:
            return Domain(domain_name)

    return {DomainDataCreator, DomainPIIRedactor, DomainPreprocessor, DomainDeduplicator, DomainEmbedder}


def make_config_based_image_pipeline_feature(
    output_name: str,
    domain_name: str,
    redaction_method: str = "solid",
    preprocessing_method: str = "resize",
    deduplication_method: str = "exact_hash",
    embedding_method: str = "mock",
) -> Feature:
    """
    Create an image pipeline feature using configuration-based syntax.

    Chain: image_docs -> pii_redact -> preprocess -> dedupe -> embed

    Args:
        output_name: Name for the final embedded feature
        domain_name: Domain to use for all features in the chain
        redaction_method: One of: blur, pixel, solid
        preprocessing_method: One of: resize, normalize, thumbnail
        deduplication_method: One of: exact_hash, phash, dhash
        embedding_method: One of: mock, hash, clip
    """
    docs_feature = Feature("image_docs", domain=domain_name)

    pii_feature = Feature(
        f"{output_name}_pii",
        domain=domain_name,
        options=Options(context={"image_redaction_method": redaction_method, "in_features": docs_feature}),
    )

    preproc_feature = Feature(
        f"{output_name}_preprocessed",
        domain=domain_name,
        options=Options(context={"preprocessing_method": preprocessing_method, "in_features": pii_feature}),
    )

    deduped_feature = Feature(
        f"{output_name}_deduped",
        domain=domain_name,
        options=Options(context={"image_deduplication_method": deduplication_method, "in_features": preproc_feature}),
    )

    embedded_feature = Feature(
        output_name,
        domain=domain_name,
        options=Options(context={"image_embedding_method": embedding_method, "in_features": deduped_feature}),
    )

    return embedded_feature


class TestAlternativeImageProviders:
    """Test alternative provider implementations work correctly."""

    def test_multiple_provider_combinations(self) -> None:
        """
        Test 3 different provider combinations using config-based features:
        1. set1: SolidFillPIIRedactor, ResizePreprocessor, ExactHashDedup, MockEmbedder
        2. set2: SolidFillPIIRedactor, NormalizePreprocessor, ExactHashDedup, HashEmbedder
        3. set3: SolidFillPIIRedactor, ThumbnailPreprocessor, ExactHashDedup, MockEmbedder
        """
        set1 = make_image_domain_providers(
            "imgset1", SolidFillPIIRedactor, ResizePreprocessor, ExactHashImageDeduplicator, MockImageEmbedder
        )
        set2 = make_image_domain_providers(
            "imgset2", SolidFillPIIRedactor, NormalizePreprocessor, ExactHashImageDeduplicator, HashImageEmbedder
        )
        set3 = make_image_domain_providers(
            "imgset3", SolidFillPIIRedactor, ThumbnailPreprocessor, ExactHashImageDeduplicator, MockImageEmbedder
        )

        all_providers = set1 | set2 | set3

        feature1 = make_config_based_image_pipeline_feature(
            "embedded_imgset1",
            "imgset1",
            redaction_method="solid",
            preprocessing_method="resize",
            deduplication_method="exact_hash",
            embedding_method="mock",
        )
        feature2 = make_config_based_image_pipeline_feature(
            "embedded_imgset2",
            "imgset2",
            redaction_method="solid",
            preprocessing_method="normalize",
            deduplication_method="exact_hash",
            embedding_method="hash",
        )
        feature3 = make_config_based_image_pipeline_feature(
            "embedded_imgset3",
            "imgset3",
            redaction_method="solid",
            preprocessing_method="thumbnail",
            deduplication_method="exact_hash",
            embedding_method="mock",
        )

        raw_result = mlodaAPI.run_all(
            features=[feature1, feature2, feature3],
            compute_frameworks={PythonDictFramework},
            plugin_collector=PluginCollector.enabled_feature_groups(all_providers),
        )

        assert len(raw_result) == 3, f"Should have 3 result sets, got {len(raw_result)}"

        feature_names = ["embedded_imgset1", "embedded_imgset2", "embedded_imgset3"]
        for i, result_set in enumerate(raw_result):
            result = flatten_result([result_set])
            assert len(result) > 0, f"Set {i + 1}: Should produce results"

            for row in result:
                embedding = row.get(feature_names[i])
                assert isinstance(embedding, list), f"Set {i + 1}: Embedding should be a list"
                magnitude = math.sqrt(sum(x * x for x in embedding))
                assert abs(magnitude - 1.0) < 0.001, f"Set {i + 1}: Embedding should be unit length, got {magnitude}"
