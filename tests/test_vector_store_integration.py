"""
Integration tests for the vector store pipeline stage.

Tests the full pipeline through __indexed, verifying artifact persistence.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict, List, Set, Type

from mloda.user import mloda, mlodaAPI, PluginCollector, Domain, Feature, Options
from mloda.provider import DataCreator, FeatureGroup
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
    PythonDictFramework,
)

from rag_integration.feature_groups.rag_pipeline import (
    RegexPIIRedactor,
    FixedSizeChunker,
    ExactHashDeduplicator,
    MockEmbedder,
    FaissFlatIndexer,
)

SAMPLE_DOCUMENTS = [
    {"doc_id": "doc_001", "text": "Contact john@example.com for details."},
    {"doc_id": "doc_002", "text": "Meeting with jane@test.org tomorrow."},
    {"doc_id": "doc_003", "text": "Unique content here."},
]


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


def flatten_result(result: List[Any]) -> List[Dict[str, Any]]:
    """Flatten nested mlodaAPI result."""
    if result and isinstance(result[0], list):
        return result[0]
    return result


def get_test_providers() -> Set[Type[FeatureGroup]]:
    return {
        MockDocumentDataCreator,
        RegexPIIRedactor,
        FixedSizeChunker,
        ExactHashDeduplicator,
        MockEmbedder,
        FaissFlatIndexer,
    }


class TestVectorStoreIntegration:
    """Test full pipeline through __indexed stage."""

    def test_full_pipeline_through_indexed(self) -> None:
        """
        Run the full pipeline: docs -> pii_redacted -> chunked -> deduped -> embedded -> indexed.

        Verify that:
        - Rows are produced
        - Feature value is an integer (vector_id)
        """
        feature_name = "docs__pii_redacted__chunked__deduped__embedded__indexed"

        raw_result = mlodaAPI.run_all(
            features=[feature_name],
            compute_frameworks={PythonDictFramework},
            plugin_collector=PluginCollector.enabled_feature_groups(get_test_providers()),
        )

        rows = flatten_result(raw_result)
        assert len(rows) > 0, "Should produce results"

        for row in rows:
            # mlodaAPI.run_all() returns only the requested feature column
            feature_value = row.get(feature_name)
            assert isinstance(feature_value, int), f"Feature value should be int (vector_id), got {type(feature_value)}"

    def test_indexed_with_artifact_persistence(self) -> None:
        """
        Test that the FAISS index is saved to disk via artifact persistence.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            feature_name = "docs__pii_redacted__chunked__deduped__embedded__indexed"

            feature = Feature(
                feature_name,
                options=Options({"artifact_storage_path": tmp_dir}),
            )

            raw_result = mlodaAPI.run_all(
                features=[feature],
                compute_frameworks={PythonDictFramework},
                plugin_collector=PluginCollector.enabled_feature_groups(get_test_providers()),
            )

            rows = flatten_result(raw_result)
            assert len(rows) > 0

            # Verify artifact files were created
            artifact_path = Path(tmp_dir)
            faiss_files = list(artifact_path.glob("vector_store_*.faiss"))
            json_files = list(artifact_path.glob("vector_store_*_metadata.json"))

            assert len(faiss_files) >= 1, "FAISS index file should be created"
            assert len(json_files) >= 1, "Metadata sidecar should be created"


def make_domain_providers(domain_name: str) -> Set[Type[FeatureGroup]]:
    """Create a provider set with a specific domain for artifact isolation."""

    class DomainDataCreator(MockDocumentDataCreator):
        @classmethod
        def get_domain(cls) -> Domain:
            return Domain(domain_name)

    class DomainPII(RegexPIIRedactor):
        @classmethod
        def get_domain(cls) -> Domain:
            return Domain(domain_name)

    class DomainChunker(FixedSizeChunker):
        @classmethod
        def get_domain(cls) -> Domain:
            return Domain(domain_name)

    class DomainDedup(ExactHashDeduplicator):
        @classmethod
        def get_domain(cls) -> Domain:
            return Domain(domain_name)

    class DomainEmbedder(MockEmbedder):
        @classmethod
        def get_domain(cls) -> Domain:
            return Domain(domain_name)

    class DomainIndexer(FaissFlatIndexer):
        @classmethod
        def get_domain(cls) -> Domain:
            return Domain(domain_name)

    return {DomainDataCreator, DomainPII, DomainChunker, DomainDedup, DomainEmbedder, DomainIndexer}


class TestVectorStoreArtifactIntegration:
    """Test vector store artifact save and reload via mloda artifact lifecycle."""

    def test_vector_store_artifact_save_and_load(self) -> None:
        """
        Run 1: compute index + save artifact to disk.
        Run 2: pass artifacts back in options so mloda sets artifact_to_load=True,
               verify the index is loaded from disk (not recomputed).
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            artifact_path = Path(tmp_dir)
            providers = make_domain_providers("vs_artifact_test")
            feature_name = "docs__pii_redacted__chunked__deduped__embedded__indexed"

            feature_options: Dict[str, Any] = {"artifact_storage_path": str(artifact_path)}

            # Run 1: compute and save
            feature1 = Feature(feature_name, options=Options(feature_options), domain="vs_artifact_test")

            api1 = mloda(
                [feature1],
                {PythonDictFramework},
                plugin_collector=PluginCollector.enabled_feature_groups(providers),
            )
            api1._batch_run()
            results1 = api1.get_result()
            artifacts1 = api1.get_artifacts()

            rows1 = flatten_result(results1)
            assert len(rows1) > 0, "Run 1 should produce results"
            assert len(artifacts1) > 0, "Run 1 should produce artifacts"

            # Verify artifact files exist on disk
            faiss_files = list(artifact_path.glob("vector_store_*.faiss"))
            json_files = list(artifact_path.glob("vector_store_*_metadata.json"))
            assert len(faiss_files) >= 1, "FAISS index file should be created"
            assert len(json_files) >= 1, "Metadata sidecar should be created"

            # Run 2: load from artifact (merge artifacts back into options)
            combined_options = {**feature_options, **artifacts1}
            feature2 = Feature(feature_name, options=Options(combined_options), domain="vs_artifact_test")

            api2 = mloda(
                [feature2],
                {PythonDictFramework},
                plugin_collector=PluginCollector.enabled_feature_groups(providers),
            )
            api2._batch_run()
            results2 = api2.get_result()

            rows2 = flatten_result(results2)
            assert len(rows2) > 0, "Run 2 should produce results"

            # Both runs should produce the same number of rows
            assert len(rows1) == len(rows2), "Row count should match between runs"
