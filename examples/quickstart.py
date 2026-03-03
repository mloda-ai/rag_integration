#!/usr/bin/env python3
"""
RAG pipeline quickstart using mloda plugins.

Demonstrates:
  - Phase 1: Full ingestion pipeline (source -> PII redact -> chunk -> dedup -> embed -> FAISS index)
  - Phase 2: Retrieval from the persisted FAISS index using FaissRetriever
  - Swap the indexer in one line (FaissFlatIndexer -> FaissHNSWIndexer)

Run with: python examples/quickstart.py
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

# ---------------------------------------------------------------------------
# Sample documents: 2 with PII, 1 exact duplicate pair
# ---------------------------------------------------------------------------

DOCUMENTS = [
    {"doc_id": "doc_1", "text": "Contact john@example.com or call 555-123-4567 for email support."},
    {"doc_id": "doc_2", "text": "Meeting with jane@test.org at 800-555-0199 about the project."},
    {"doc_id": "doc_3", "text": "Technical documentation for the API service endpoints."},
    {"doc_id": "doc_4", "text": "Duplicate content here."},
    {"doc_id": "doc_5", "text": "Duplicate content here."},
]


# ---------------------------------------------------------------------------
# Document source: minimal FeatureGroup that feeds DOCUMENTS into the pipeline
# ---------------------------------------------------------------------------


class DocumentSource(FeatureGroup):
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
        return [{"docs": doc["text"], "doc_id": doc["doc_id"]} for doc in DOCUMENTS]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("=== RAG Pipeline Quickstart ===\n")
    print(f"Input: {len(DOCUMENTS)} documents (2 with PII, 1 exact duplicate pair)\n")

    with tempfile.TemporaryDirectory() as tmp_dir:
        # -------------------------------------------------------------------
        # Phase 1: Ingestion — build and persist a FAISS index
        # Swap FaissFlatIndexer -> FaissHNSWIndexer here to change the index type
        # -------------------------------------------------------------------
        print("Phase 1 — Ingestion (regex PII, fixed-size chunks, exact dedup, MockEmbedder, FaissFlatIndexer)")

        ingestion_providers: Set[Type[FeatureGroup]] = {
            DocumentSource,
            RegexPIIRedactor,
            FixedSizeChunker,
            ExactHashDeduplicator,
            MockEmbedder,
            FaissFlatIndexer,  # swap to FaissHNSWIndexer here for approximate search
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

        ingestion_rows = (
            ingestion_result[0] if ingestion_result and isinstance(ingestion_result[0], list) else ingestion_result
        )
        print(f"  {len(DOCUMENTS)} docs -> {len(ingestion_rows)} indexed chunks after dedup")

        # Discover artifact paths written by FaissFlatIndexer
        artifact_path = Path(tmp_dir)
        faiss_files = list(artifact_path.glob("vector_store_*.faiss"))
        metadata_files = list(artifact_path.glob("vector_store_*_metadata.json"))
        if not faiss_files or not metadata_files:
            raise RuntimeError("Expected FAISS index and metadata sidecar on disk")

        index_path = str(faiss_files[0])
        metadata_path = str(metadata_files[0])
        print(f"  FAISS index: {Path(index_path).name}")
        print(f"  Metadata:    {Path(metadata_path).name}\n")

        # -------------------------------------------------------------------
        # Phase 2: Retrieval — query the persisted FAISS index
        # -------------------------------------------------------------------
        print("Phase 2 — Retrieval (FaissRetriever, query: 'contact email support', top_k=3)")

        retrieval_feature = Feature(
            "retrieved",
            options=Options(
                {
                    "index_path": index_path,
                    "metadata_path": metadata_path,
                    "query_text": "contact email support",
                    "embedding_method": "mock",
                    "top_k": 3,
                }
            ),
        )

        retrieval_result = mlodaAPI.run_all(
            features=[retrieval_feature],
            compute_frameworks={PythonDictFramework},
            plugin_collector=PluginCollector.enabled_feature_groups({FaissRetriever}),
        )

        retrieval_rows = (
            retrieval_result[0] if retrieval_result and isinstance(retrieval_result[0], list) else retrieval_result
        )
        row = retrieval_rows[0]
        result = row.get("retrieved", row)

        print(f"  Retrieved {len(result['indices'])} chunks:")
        for rank, (idx, dist, text) in enumerate(zip(result["indices"], result["distances"], result["texts"])):
            print(f"    [{rank + 1}] idx={idx}  dist={dist:.4f}  text={text[:60]!r}")

    print("\nDone.")


if __name__ == "__main__":
    main()
