"""Base class for document source feature groups."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, ClassVar

from mloda.provider import ComputeFramework, FeatureGroup, FeatureSet
from mloda.user import FeatureName, Options
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
    PythonDictFramework,
)
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_utils import homogenize_rows

from rag_integration.feature_groups.empty_result import empty_result


class BaseDocumentSource(FeatureGroup):
    """
    Base class for document source feature groups.

    This is a ROOT feature - it has no input features and provides
    the initial documents for the pipeline.

    Data Structure (PythonDict):
        [
            {"doc_id": "1", "text": "Document content...", "metadata": {...}},
            {"doc_id": "2", "text": "Another document...", "metadata": {...}},
        ]

    Usage:
        features = ["docs"]  # Returns raw documents
    """

    EMPTY_SCHEMA_COLUMNS: ClassVar[tuple[str, ...]] = ("doc_id", "text")

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {PythonDictFramework}

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: Any = None,
    ) -> bool:
        """Match features named 'docs' exactly."""
        return feature_name == "docs"

    def input_features(self, options: Options, feature_name: FeatureName) -> None:
        """Root feature - no input features."""
        return

    @classmethod
    @abstractmethod
    def _load_documents(cls, options: Options) -> list[dict[str, Any]]:
        """
        Load documents from the source.

        Args:
            options: Options containing source configuration

        Returns:
            List of document dictionaries with 'doc_id', 'text', and optional 'metadata'
        """
        ...

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> list[dict[str, Any]] | dict[str, list[Any]]:
        """Load and return documents with a uniform key schema."""
        for feature in features.features:
            rows = homogenize_rows(cls._load_documents(feature.options))
            if not rows:
                return empty_result(None, features, cls.EMPTY_SCHEMA_COLUMNS)
            # mloda requires a column named after the requested feature.
            return [{**row, str(feature.name): row.get("text")} for row in rows]
        return []
