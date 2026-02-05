"""
Integration tests for the RAG pipeline.

Uses mlodaAPI.run_all() with multiple features per call for efficiency.
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Set, Type

from mloda.user import mlodaAPI, mloda, PluginCollector, Domain, Feature, Options
from mloda.provider import DataCreator, FeatureGroup
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
    PythonDictFramework,
)

from rag_integration.feature_groups.rag_pipeline import (
    RegexPIIRedactor,
    PresidioPIIRedactor,
    FixedSizeChunker,
    SentenceChunker,
    SemanticChunker,
    ExactHashDeduplicator,
    NormalizedDeduplicator,
    MockEmbedder,
    HashEmbedder,
    TfidfEmbedder,
    SentenceTransformerEmbedder,
)
from rag_integration.feature_groups.rag_pipeline.embedding import EmbeddingArtifact


# =============================================================================
# Test Data
# =============================================================================

SAMPLE_DOCUMENTS = [
    {"doc_id": "doc_001", "text": "Contact john@example.com or call 555-123-4567."},
    {"doc_id": "doc_002", "text": "Meeting with jane@test.org at 800-555-0199."},
    {"doc_id": "doc_003", "text": "Duplicate content here."},
    {"doc_id": "doc_004", "text": "Duplicate content here."},  # Exact duplicate
]


# =============================================================================
# DataCreator for test data
# =============================================================================


class MockDocumentDataCreator(FeatureGroup):
    """DataCreator that provides mock documents for testing."""

    @classmethod
    def input_data(cls) -> DataCreator:
        return DataCreator({"docs"})

    @classmethod
    def match_feature_group_criteria(cls, feature_name: Any, options: Any, data_access_collection: Any = None) -> bool:
        return str(feature_name) == "docs"

    @classmethod
    def compute_framework_rule(cls) -> Set[Type[Any]]:
        return {PythonDictFramework}

    @classmethod
    def calculate_feature(cls, data: Any, features: Any) -> List[Dict[str, Any]]:
        return [{"docs": doc["text"], "doc_id": doc["doc_id"]} for doc in SAMPLE_DOCUMENTS]


# =============================================================================
# Helpers
# =============================================================================


def flatten_result(result: List[Any]) -> List[Dict[str, Any]]:
    """Flatten nested mlodaAPI result."""
    if result and isinstance(result[0], list):
        return result[0]
    return result


def get_results_by_feature(raw_result: List[Any], feature_names: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Map mlodaAPI results to feature names.

    mlodaAPI.run_all() returns a list where each element corresponds to each requested feature.
    This helper creates a dict mapping feature_name -> result_rows.
    """
    return {name: flatten_result([raw_result[i]]) for i, name in enumerate(feature_names)}


def get_test_providers() -> Set[Type[FeatureGroup]]:
    return {MockDocumentDataCreator, RegexPIIRedactor, FixedSizeChunker, ExactHashDeduplicator, MockEmbedder}


# =============================================================================
# Integration Test: Full Pipeline with Multiple Features
# =============================================================================


class TestFullPipelineIntegration:
    """Test the complete pipeline requesting multiple features at once."""

    def test_pipeline_stages_in_single_call(self) -> None:
        """
        Request all pipeline stages in one call and verify each stage.

        Tests:
        - docs: raw documents loaded (4 docs)
        - docs__pii_redacted: PII removed (emails redacted)
        - docs__pii_redacted__chunked: text chunked (row count may increase)
        - docs__pii_redacted__chunked__deduped: duplicates removed (row count decreases)
        - docs__pii_redacted__chunked__deduped__embedded: embeddings generated (unit normalized)
        """
        feature_names = [
            "docs",
            "docs__pii_redacted",
            "docs__pii_redacted__chunked",
            "docs__pii_redacted__chunked__deduped",
            "docs__pii_redacted__chunked__deduped__embedded",
        ]

        raw_result = mlodaAPI.run_all(
            features=list(feature_names),
            compute_frameworks={PythonDictFramework},
            plugin_collector=PluginCollector.enabled_feature_groups(get_test_providers()),
        )

        results = get_results_by_feature(raw_result, feature_names)

        # 1. docs: should have raw documents
        docs_rows = results["docs"]
        assert len(docs_rows) == 4, f"Should have 4 docs, got {len(docs_rows)}"

        # 2. docs__pii_redacted: should have redacted text (no raw emails)
        pii_rows = results["docs__pii_redacted"]
        assert len(pii_rows) == 4, "PII redaction should preserve row count"
        for row in pii_rows:
            redacted = row.get("docs__pii_redacted", "")
            # Check no raw emails remain
            assert "@" not in redacted or "[EMAIL]" in redacted, f"PII should be redacted: {redacted}"

        # 3. docs__pii_redacted__chunked: chunking may expand rows
        chunked_rows = results["docs__pii_redacted__chunked"]
        assert len(chunked_rows) >= 4, "Chunking should produce at least as many rows"

        # 4. docs__pii_redacted__chunked__deduped: dedup reduces rows (we have duplicates)
        deduped_rows = results["docs__pii_redacted__chunked__deduped"]
        # We have 2 duplicate docs, so dedup should reduce count
        assert len(deduped_rows) < len(chunked_rows), "Dedup should reduce row count"

        # 5. docs__pii_redacted__chunked__deduped__embedded: embeddings
        embedded_rows = results["docs__pii_redacted__chunked__deduped__embedded"]
        assert len(embedded_rows) == len(deduped_rows), "Embedding should preserve row count"

        for row in embedded_rows:
            embedding = row.get("docs__pii_redacted__chunked__deduped__embedded")
            assert isinstance(embedding, list), "Embedding should be a list"
            assert len(embedding) > 0, "Embedding should have dimensions"
            assert all(isinstance(x, float) for x in embedding), "Embedding values should be floats"

            # Check unit normalization
            magnitude = math.sqrt(sum(x * x for x in embedding))
            assert abs(magnitude - 1.0) < 0.001, f"Embedding should be unit length, got {magnitude}"


# =============================================================================
# Domain-specific provider sets for testing multiple configs in one API call
# =============================================================================


def make_domain_providers(
    domain_name: str,
    chunker: type,
    deduplicator: type,
    embedder: type,
    pii_redactor: type = RegexPIIRedactor,
) -> Set[Type[FeatureGroup]]:
    """
    Factory to create a complete provider set with a specific domain.
    All providers in the chain get the same domain so feature matching works.
    """

    class DomainDataCreator(MockDocumentDataCreator):
        @classmethod
        def get_domain(cls) -> Domain:
            return Domain(domain_name)

    class DomainPIIRedactor(pii_redactor):  # type: ignore[misc]
        @classmethod
        def get_domain(cls) -> Domain:
            return Domain(domain_name)

    class DomainChunker(chunker):  # type: ignore[misc]
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

    return {DomainDataCreator, DomainPIIRedactor, DomainChunker, DomainDeduplicator, DomainEmbedder}


def make_config_based_pipeline_feature(
    output_name: str,
    domain_name: str,
    redaction_method: str = "regex",
    chunking_method: str = "fixed_size",
    deduplication_method: str = "exact_hash",
    embedding_method: str = "mock",
) -> Feature:
    """
    Create a pipeline feature using configuration-based syntax.

    Uses nested in_features with unique discriminator keys per feature group type
    (following mloda's pattern like aggregation_type, scaler_type, etc.)

    Chain: docs -> pii_redact -> chunk -> dedupe -> embed

    Args:
        output_name: Name for the final embedded feature
        domain_name: Domain to use for all features in the chain
        redaction_method: One of: regex, simple, pattern, presidio
        chunking_method: One of: fixed_size, sentence, paragraph, semantic
        deduplication_method: One of: exact_hash, normalized, ngram
        embedding_method: One of: mock, hash, tfidf, sentence_transformer
    """
    # Build chain from inside out
    docs_feature = Feature("docs", domain=domain_name)

    pii_feature = Feature(
        f"{output_name}_pii",
        domain=domain_name,
        options=Options(context={"redaction_method": redaction_method, "in_features": docs_feature}),
    )

    chunked_feature = Feature(
        f"{output_name}_chunked",
        domain=domain_name,
        options=Options(context={"chunking_method": chunking_method, "in_features": pii_feature}),
    )

    deduped_feature = Feature(
        f"{output_name}_deduped",
        domain=domain_name,
        options=Options(context={"deduplication_method": deduplication_method, "in_features": chunked_feature}),
    )

    embedded_feature = Feature(
        output_name,
        domain=domain_name,
        options=Options(context={"embedding_method": embedding_method, "in_features": deduped_feature}),
    )

    return embedded_feature


class TestAlternativeProviders:
    """Test alternative provider implementations work correctly."""

    def test_all_provider_combinations(self) -> None:
        """
        Test 4 different provider combinations using config-based features:
        1. set1: RegexPIIRedactor, FixedSizeChunker, ExactHashDeduplicator, MockEmbedder
        2. set2: RegexPIIRedactor, FixedSizeChunker, ExactHashDeduplicator, HashEmbedder
        3. set3: RegexPIIRedactor, SentenceChunker, NormalizedDeduplicator, TfidfEmbedder
        4. set4: PresidioPIIRedactor, SemanticChunker, NormalizedDeduplicator, SentenceTransformerEmbedder

        Uses configuration-based features with unique discriminator keys:
        - redaction_method: regex, simple, pattern, presidio
        - chunking_method: fixed_size, sentence, paragraph, semantic
        - deduplication_method: exact_hash, normalized, ngram
        - embedding_method: mock, hash, tfidf, sentence_transformer
        """
        # Create domain-specific provider sets
        set1 = make_domain_providers("set1", FixedSizeChunker, ExactHashDeduplicator, MockEmbedder)
        set2 = make_domain_providers("set2", FixedSizeChunker, ExactHashDeduplicator, HashEmbedder)
        set3 = make_domain_providers("set3", SentenceChunker, NormalizedDeduplicator, TfidfEmbedder)
        set4 = make_domain_providers(
            "set4",
            SemanticChunker,
            NormalizedDeduplicator,
            SentenceTransformerEmbedder,
            pii_redactor=PresidioPIIRedactor,
        )

        all_providers = set1 | set2 | set3 | set4

        # Create config-based features with unique discriminator keys
        feature1 = make_config_based_pipeline_feature(
            "embedded_set1",
            "set1",
            chunking_method="fixed_size",
            deduplication_method="exact_hash",
            embedding_method="mock",
        )
        feature2 = make_config_based_pipeline_feature(
            "embedded_set2",
            "set2",
            chunking_method="fixed_size",
            deduplication_method="exact_hash",
            embedding_method="hash",
        )
        feature3 = make_config_based_pipeline_feature(
            "embedded_set3",
            "set3",
            chunking_method="sentence",
            deduplication_method="normalized",
            embedding_method="tfidf",
        )
        feature4 = make_config_based_pipeline_feature(
            "embedded_set4",
            "set4",
            redaction_method="presidio",
            chunking_method="semantic",
            deduplication_method="normalized",
            embedding_method="sentence_transformer",
        )

        # Single API call with 4 features, each with different domain
        raw_result = mlodaAPI.run_all(
            features=[feature1, feature2, feature3, feature4],
            compute_frameworks={PythonDictFramework},
            plugin_collector=PluginCollector.enabled_feature_groups(all_providers),
        )

        assert len(raw_result) == 4, f"Should have 4 result sets, got {len(raw_result)}"

        feature_names = ["embedded_set1", "embedded_set2", "embedded_set3", "embedded_set4"]
        for i, result_set in enumerate(raw_result):
            result = flatten_result([result_set])
            assert len(result) > 0, f"Set {i + 1}: Should produce results"

            for row in result:
                embedding = row.get(feature_names[i])
                assert isinstance(embedding, list), f"Set {i + 1}: Embedding should be a list"
                magnitude = math.sqrt(sum(x * x for x in embedding))
                assert abs(magnitude - 1.0) < 0.001, f"Set {i + 1}: Embedding should be unit length, got {magnitude}"


# =============================================================================
# Integration Test: Embedding Artifact Persistence
# =============================================================================


class TestEmbeddingArtifactIntegration:
    """Test embedding artifact save and load in a full pipeline."""

    def test_embedding_artifact_save_and_load(self) -> None:
        """
        Test that embeddings are saved on first run and loaded on second run.

        Uses SentenceTransformerEmbedder which has artifact support via EmbeddingArtifact.
        Uses string-based feature naming with options for artifact storage path.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            artifact_path = Path(tmp_dir)

            # Create domain-specific providers for artifact test
            providers = make_domain_providers(
                "artifact_test",
                FixedSizeChunker,
                ExactHashDeduplicator,
                SentenceTransformerEmbedder,
            )

            # Options with artifact storage path
            feature_options: Dict[str, Any] = {"artifact_storage_path": str(artifact_path)}

            # First run: compute and save embeddings (mloda sets artifact_to_save automatically)
            feature1 = Feature(
                "docs__pii_redacted__chunked__deduped__embedded",
                options=Options(feature_options),
                domain="artifact_test",
            )

            api1 = mloda([feature1], {PythonDictFramework}, plugin_collector=PluginCollector.enabled_feature_groups(providers))
            api1._batch_run()
            results1 = api1.get_result()
            artifacts1 = api1.get_artifacts()

            # Extract embeddings from first run
            embeddings_first_run = [
                row.get("docs__pii_redacted__chunked__deduped__embedded") for row in flatten_result(results1)
            ]

            # Verify we got results and artifacts
            assert len(results1) > 0, "Should have results"
            assert len(artifacts1) > 0, "Should have artifacts saved"

            # Verify artifact files were created
            artifact_files = list(artifact_path.glob("embedding_artifact_*.joblib"))
            assert len(artifact_files) > 0, "Artifact files should be created"

            # Second run: load embeddings from artifact (pass artifacts back in options)
            combined_options = {**feature_options, **artifacts1}
            feature2 = Feature(
                "docs__pii_redacted__chunked__deduped__embedded",
                options=Options(combined_options),
                domain="artifact_test",
            )

            api2 = mloda([feature2], {PythonDictFramework}, plugin_collector=PluginCollector.enabled_feature_groups(providers))
            api2._batch_run()
            results2 = api2.get_result()
            artifacts2 = api2.get_artifacts()

            # Extract embeddings from second run
            embeddings_second_run = [
                row.get("docs__pii_redacted__chunked__deduped__embedded") for row in flatten_result(results2)
            ]

            # Verify embeddings match between runs (loaded from artifact)
            assert len(embeddings_first_run) == len(embeddings_second_run), "Should have same number of embeddings"

            for i, (emb1, emb2) in enumerate(zip(embeddings_first_run, embeddings_second_run)):
                assert emb1 == emb2, f"Embedding {i} should match between runs"

            # Verify embeddings are valid
            for embedding in embeddings_second_run:
                assert isinstance(embedding, list), "Embedding should be a list"
                magnitude = math.sqrt(sum(x * x for x in embedding))
                assert abs(magnitude - 1.0) < 0.001, f"Embedding should be unit length, got {magnitude}"
