"""
Focused FAISS search integration test: single document ingestion + retrieval.

Covers:
1. Ingest 1 document through full pipeline (PII -> chunk -> dedup -> embed -> index)
2. Discover artifact paths on disk
3. Query with FaissRetriever and verify result structure
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


class SingleDocDataCreator(FeatureGroup):
    """DataCreator that provides a single test document."""

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
        return [
            {
                "docs": "The capital of France is Paris. It is known for the Eiffel Tower.",
                "doc_id": "doc_france",
            }
        ]


def flatten_result(result: List[Any]) -> List[Dict[str, Any]]:
    """Flatten nested mlodaAPI result."""
    if result and isinstance(result[0], list):
        return result[0]
    return result


class TestFaissSearchIntegration:
    """Focused test: ingest single document, then search with FAISS."""

    def test_single_doc_ingest_and_search(self) -> None:
        """
        Ingest one document through the full pipeline, then query it.

        Steps:
        1. Run full ingestion pipeline into a temp directory
        2. Discover .faiss + metadata JSON artifact paths
        3. Query with FaissRetriever (query_text="capital of France", top_k=1)
        4. Assert result has indices/distances/texts/doc_ids, text is non-empty, distance >= 0
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Phase 1: Ingestion
            ingestion_providers: Set[Type[FeatureGroup]] = {
                SingleDocDataCreator,
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

            # Phase 2: Discover artifact paths
            artifact_path = Path(tmp_dir)
            faiss_files = list(artifact_path.glob("vector_store_*.faiss"))
            metadata_files = list(artifact_path.glob("vector_store_*_metadata.json"))

            assert len(faiss_files) >= 1, "FAISS index file should exist"
            assert len(metadata_files) >= 1, "Metadata sidecar should exist"

            index_path = str(faiss_files[0])
            metadata_path = str(metadata_files[0])

            # Phase 3: Retrieval
            retrieval_providers: Set[Type[FeatureGroup]] = {FaissRetriever}

            retrieval_feature = Feature(
                "retrieved",
                options=Options(
                    {
                        "index_path": index_path,
                        "metadata_path": metadata_path,
                        "query_text": "capital of France",
                        "embedding_method": "mock",
                        "top_k": 1,
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
            result = row.get("retrieved", row)
            assert isinstance(result, dict), "Result should be a dict"

            # Phase 4: Assertions
            assert "indices" in result, "Result should contain indices"
            assert "distances" in result, "Result should contain distances"
            assert "texts" in result, "Result should contain texts"
            assert "doc_ids" in result, "Result should contain doc_ids"

            assert len(result["indices"]) == 1, f"Should return 1 result, got {len(result['indices'])}"
            assert len(result["texts"]) == 1

            # Text should be non-empty
            assert isinstance(result["texts"][0], str)
            assert len(result["texts"][0]) > 0, "Returned text should be non-empty"

            # Distance should be non-negative
            assert result["distances"][0] >= 0, "Distance should be non-negative"
