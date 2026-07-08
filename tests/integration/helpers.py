"""Shared helpers for integration tests."""

from __future__ import annotations

from typing import Any, Dict, List, Set, Type

from mloda.provider import DataCreator, FeatureGroup
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
    PythonDictFramework,
)

from rag_integration.feature_groups.columnar import columnar_to_rows


def flatten_result(result: Any) -> List[Dict[str, Any]]:
    """Unwrap a nested mlodaAPI result to a flat list of row dicts.

    PythonDict partitions are columnar ``dict[str, list]`` (mloda 0.9.0); pivot the
    first partition back to rows. Accepts a raw result list, a single wrapped
    partition, or a bare columnar dict.
    """
    if isinstance(result, dict):
        return columnar_to_rows(result)
    if result and isinstance(result[0], (dict, list)):
        return columnar_to_rows(result[0])
    return columnar_to_rows(result)


def get_results_by_feature(raw_result: List[Any], feature_names: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    """Map mlodaAPI results to feature names.

    mlodaAPI.run_all() returns a list where each element corresponds to each requested feature.
    This helper creates a dict mapping feature_name -> result_rows.
    """
    return {name: flatten_result([raw_result[i]]) for i, name in enumerate(feature_names)}


def get_metrics(raw_result: Any, feature_name: str) -> Dict[str, Any]:
    """Unwrap the mloda result and extract the metrics dict stored under feature_name."""
    rows = flatten_result(raw_result[0])
    assert len(rows) == 1, "RetrievalEvaluator should return exactly one aggregate row"
    row = rows[0]
    assert feature_name in row, f"Expected '{feature_name}' key in result row, got keys: {list(row.keys())}"
    metrics: Dict[str, Any] = row[feature_name]
    return metrics


def make_mock_document_data_creator(documents: List[Dict[str, str]]) -> Type[FeatureGroup]:
    """Create a MockDocumentDataCreator FeatureGroup for the given documents."""

    class _MockDocumentDataCreator(FeatureGroup):
        """DataCreator that provides mock documents for testing."""

        @classmethod
        def input_data(cls) -> DataCreator:
            return DataCreator({"docs"})

        @classmethod
        def match_feature_group_criteria(
            cls, feature_name: Any, options: Any, data_access_collection: Any = None
        ) -> bool:
            return str(feature_name) == "docs"

        @classmethod
        def compute_framework_rule(cls) -> Set[Type[Any]]:
            return {PythonDictFramework}

        @classmethod
        def calculate_feature(cls, data: Any, features: Any) -> List[Dict[str, Any]]:
            return [{"docs": doc["text"], "doc_id": doc["doc_id"]} for doc in documents]

    return _MockDocumentDataCreator
