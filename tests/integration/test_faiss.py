"""
FAISS integration tests: vector store indexing, artifact persistence, and retrieval.

Covers three scenarios:
1. Full pipeline through __indexed (vector store pipeline)
2. Artifact save/load roundtrip (vector store persistence)
3. Two-phase workflow: ingest + query with FaissRetriever
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict, Set, Type

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
    FaissRetriever,
)
from tests.integration.helpers import flatten_result

SAMPLE_DOCUMENTS = [
    {"doc_id": "doc_001", "text": "Contact john@example.com for email support."},
    {"doc_id": "doc_002", "text": "Meeting with jane@test.org at the office."},
    {"doc_id": "doc_003", "text": "Technical documentation for the API service."},
    {"doc_id": "doc_004", "text": "Customer feedback about the new product launch."},
    {"doc_id": "doc_005", "text": "Quarterly financial report summary and analysis."},
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
    def calculate_feature(cls, data: Any, features: Any) -> list[dict[str, Any]]:
        return [{"docs": doc["text"], "doc_id": doc["doc_id"]} for doc in SAMPLE_DOCUMENTS]


def get_test_providers() -> Set[Type[FeatureGroup]]:
    return {
        MockDocumentDataCreator,
        RegexPIIRedactor,
        FixedSizeChunker,
        ExactHashDeduplicator,
        MockEmbedder,
        FaissFlatIndexer,
    }


# =============================================================================
# Vector Store Pipeline
# =============================================================================


class TestVectorStorePipeline:
    """Test full pipeline through __indexed stage."""

    def test_full_pipeline_through_indexed(self) -> None:
        """Run docs -> pii_redacted -> chunked -> deduped -> embedded -> indexed."""
        feature_name = "docs__pii_redacted__chunked__deduped__embedded__indexed"

        raw_result = mlodaAPI.run_all(
            features=[feature_name],
            compute_frameworks={PythonDictFramework},
            plugin_collector=PluginCollector.enabled_feature_groups(get_test_providers()),
        )

        rows = flatten_result(raw_result)
        assert len(rows) > 0, "Should produce results"

        for row in rows:
            feature_value = row.get(feature_name)
            assert isinstance(feature_value, int), f"Feature value should be int (vector_id), got {type(feature_value)}"

    def test_indexed_with_artifact_persistence(self) -> None:
        """FAISS index should be saved to disk via artifact persistence."""
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

            artifact_path = Path(tmp_dir)
            faiss_files = list(artifact_path.glob("vector_store_*.faiss"))
            json_files = list(artifact_path.glob("vector_store_*_metadata.json"))

            assert len(faiss_files) >= 1, "FAISS index file should be created"
            assert len(json_files) >= 1, "Metadata sidecar should be created"


# =============================================================================
# Artifact Save/Load Roundtrip
# =============================================================================


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


class TestVectorStoreArtifactPersistence:
    """Test vector store artifact save and reload via mloda artifact lifecycle."""

    def test_vector_store_artifact_save_and_load(self) -> None:
        """
        Run 1: compute index + save artifact to disk.
        Run 2: pass artifacts back in options, verify index is loaded from disk.
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

            faiss_files = list(artifact_path.glob("vector_store_*.faiss"))
            json_files = list(artifact_path.glob("vector_store_*_metadata.json"))
            assert len(faiss_files) >= 1, "FAISS index file should be created"
            assert len(json_files) >= 1, "Metadata sidecar should be created"

            # Run 2: load from artifact
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
            assert len(rows1) == len(rows2), "Row count should match between runs"


# =============================================================================
# Index and Retrieve (two-phase workflow)
# =============================================================================


class TestIndexAndRetrieve:
    """End-to-end test: ingest, index, persist, then retrieve."""

    def test_full_write_then_read_workflow(self) -> None:
        """
        1. Run full ingestion pipeline to build + persist a FAISS index
        2. Discover artifact paths on disk
        3. Query with FaissRetriever and verify results
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Phase 1: Ingestion
            ingestion_feature = Feature(
                "docs__pii_redacted__chunked__deduped__embedded__indexed",
                options=Options({"artifact_storage_path": tmp_dir}),
            )

            ingestion_result = mlodaAPI.run_all(
                features=[ingestion_feature],
                compute_frameworks={PythonDictFramework},
                plugin_collector=PluginCollector.enabled_feature_groups(get_test_providers()),
            )

            ingestion_rows = flatten_result(ingestion_result)
            assert len(ingestion_rows) > 0, "Ingestion should produce rows"

            indexed_feature = "docs__pii_redacted__chunked__deduped__embedded__indexed"
            for row in ingestion_rows:
                assert isinstance(row.get(indexed_feature), int)

            # Phase 2: Discover artifact paths
            artifact_path = Path(tmp_dir)
            faiss_files = list(artifact_path.glob("vector_store_*.faiss"))
            metadata_files = list(artifact_path.glob("vector_store_*_metadata.json"))

            assert len(faiss_files) >= 1, "FAISS index file should exist"
            assert len(metadata_files) >= 1, "Metadata sidecar should exist"

            index_path = str(faiss_files[0])
            metadata_path = str(metadata_files[0])

            # Phase 3: Retrieval
            top_k = 3
            retrieval_feature = Feature(
                "retrieved",
                options=Options(
                    {
                        "index_path": index_path,
                        "metadata_path": metadata_path,
                        "query_text": "email contact information",
                        "embedding_method": "mock",
                        "top_k": top_k,
                    }
                ),
            )

            retrieval_result = mlodaAPI.run_all(
                features=[retrieval_feature],
                compute_frameworks={PythonDictFramework},
                plugin_collector=PluginCollector.enabled_feature_groups({FaissRetriever}),
            )

            retrieval_rows = flatten_result(retrieval_result)
            assert len(retrieval_rows) > 0, "Retrieval should produce results"

            row = retrieval_rows[0]
            result = row.get("retrieved", row)
            assert isinstance(result, dict), "Result should be a dict"

            # Phase 4: Assertions
            assert "indices" in result
            assert "distances" in result
            assert "texts" in result
            assert "doc_ids" in result

            assert len(result["indices"]) == top_k
            assert len(result["distances"]) == top_k
            assert len(result["texts"]) == top_k
            assert len(result["doc_ids"]) == top_k

            num_indexed = len(ingestion_rows)
            for idx in result["indices"]:
                assert 0 <= idx < num_indexed

            for dist in result["distances"]:
                assert dist >= 0

            for text in result["texts"]:
                assert isinstance(text, str)
                assert len(text) > 0
