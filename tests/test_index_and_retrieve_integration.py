"""
End-to-end integration test: build FAISS index via pipeline, then query with FaissRetriever.

Covers the complete user workflow:
1. Ingest documents through full pipeline to build + persist a FAISS index
2. Discover artifact paths on disk
3. Query the index with FaissRetriever in a second mlodaAPI call
4. Verify results contain valid indices, distances, texts, and doc_ids
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict, List, Set, Type

from mloda.user import mlodaAPI, PluginCollector, Feature, Options
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
    def calculate_feature(cls, data: Any, features: Any) -> List[Dict[str, Any]]:
        return [{"docs": doc["text"], "doc_id": doc["doc_id"]} for doc in SAMPLE_DOCUMENTS]


def flatten_result(result: List[Any]) -> List[Dict[str, Any]]:
    """Flatten nested mlodaAPI result."""
    if result and isinstance(result[0], list):
        return result[0]
    return result


class TestIndexAndRetrieveIntegration:
    """End-to-end test: ingest, index, persist, then retrieve."""

    def test_full_write_then_read_workflow(self) -> None:
        """
        Complete workflow:
        1. Run full ingestion pipeline to build + persist a FAISS index
        2. Discover artifact paths on disk
        3. Query with FaissRetriever and verify results
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            # ----------------------------------------------------------------
            # Phase 1: Ingestion - build and persist FAISS index
            # ----------------------------------------------------------------
            ingestion_providers: Set[Type[FeatureGroup]] = {
                MockDocumentDataCreator,
                RegexPIIRedactor,
                FixedSizeChunker,
                ExactHashDeduplicator,
                MockEmbedder,
                FaissFlatIndexer,
            }

            ingestion_feature = Feature(
                "docs__pii_redacted__chunked__deduped__embedded__indexed",
                options=Options({"artifact_storage_path": tmp_dir}),
            )

            ingestion_result = mlodaAPI.run_all(
                features=[ingestion_feature],
                compute_frameworks={PythonDictFramework},
                plugin_collector=PluginCollector.enabled_feature_groups(ingestion_providers),
            )

            ingestion_rows = flatten_result(ingestion_result)
            assert len(ingestion_rows) > 0, "Ingestion should produce rows"

            # mlodaAPI.run_all() returns only the requested feature column;
            # verify the feature value is an integer (vector_id)
            indexed_feature = "docs__pii_redacted__chunked__deduped__embedded__indexed"
            for row in ingestion_rows:
                assert isinstance(row.get(indexed_feature), int)

            # ----------------------------------------------------------------
            # Phase 2: Discover artifact paths
            # ----------------------------------------------------------------
            artifact_path = Path(tmp_dir)
            faiss_files = list(artifact_path.glob("vector_store_*.faiss"))
            metadata_files = list(artifact_path.glob("vector_store_*_metadata.json"))

            assert len(faiss_files) >= 1, "FAISS index file should exist"
            assert len(metadata_files) >= 1, "Metadata sidecar should exist"

            index_path = str(faiss_files[0])
            metadata_path = str(metadata_files[0])

            # ----------------------------------------------------------------
            # Phase 3: Retrieval - query the persisted index
            # ----------------------------------------------------------------
            retrieval_providers: Set[Type[FeatureGroup]] = {FaissRetriever}

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
                plugin_collector=PluginCollector.enabled_feature_groups(retrieval_providers),
            )

            retrieval_rows = flatten_result(retrieval_result)
            assert len(retrieval_rows) > 0, "Retrieval should produce results"

            row = retrieval_rows[0]
            # mlodaAPI.run_all() wraps results under the feature name key
            result = row.get("retrieved", row)
            assert isinstance(result, dict), "Result should be a dict"

            # ----------------------------------------------------------------
            # Phase 4: Assertions
            # ----------------------------------------------------------------

            # Result should contain all expected keys
            assert "indices" in result, "Result should contain indices"
            assert "distances" in result, "Result should contain distances"
            assert "texts" in result, "Result should contain texts"
            assert "doc_ids" in result, "Result should contain doc_ids"

            # top_k honored
            assert len(result["indices"]) == top_k, f"Should return {top_k} results, got {len(result['indices'])}"
            assert len(result["distances"]) == top_k
            assert len(result["texts"]) == top_k
            assert len(result["doc_ids"]) == top_k

            # All returned indices are valid (within range of indexed vectors)
            num_indexed = len(ingestion_rows)
            for idx in result["indices"]:
                assert 0 <= idx < num_indexed, f"Index {idx} out of range [0, {num_indexed})"

            # All distances are non-negative
            for dist in result["distances"]:
                assert dist >= 0, f"Distance should be non-negative, got {dist}"

            # Returned texts are non-empty strings
            for text in result["texts"]:
                assert isinstance(text, str), f"Text should be a string, got {type(text)}"
                assert len(text) > 0, "Text should be non-empty"
