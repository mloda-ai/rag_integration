"""Base class for document source feature groups."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, List, Set, Type, Union

from mloda.provider import FeatureGroup, ComputeFramework, FeatureSet
from mloda.user import Options, FeatureName
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
    PythonDictFramework,
)


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

    @classmethod
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFramework]]]:
        return {PythonDictFramework}

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: Union[FeatureName, str],
        options: Options,
        data_access_collection: Any = None,
    ) -> bool:
        """Match features named 'docs' exactly."""
        if isinstance(feature_name, FeatureName):
            feature_name = feature_name.name
        return feature_name == "docs"

    def input_features(self, options: Options, feature_name: FeatureName) -> None:
        """Root feature - no input features."""
        return None

    @classmethod
    @abstractmethod
    def _load_documents(cls, options: Options) -> List[Dict[str, Any]]:
        """
        Load documents from the source.

        Args:
            options: Options containing source configuration

        Returns:
            List of document dictionaries with 'doc_id', 'text', and optional 'metadata'
        """
        ...

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> List[Dict[str, Any]]:
        """Load and return documents."""
        for feature in features.features:
            documents = cls._load_documents(feature.options)
            return documents
        return []
