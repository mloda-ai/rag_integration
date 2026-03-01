"""Base class for text evaluation dataset sources."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, List, Set, Type, Union

from mloda.provider import FeatureGroup, ComputeFramework, FeatureSet
from mloda.user import Options, FeatureName
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
    PythonDictFramework,
)


class BaseTextDatasetSource(FeatureGroup):
    """
    Base class for text evaluation dataset sources.

    ROOT feature — no input features. Loads a text retrieval benchmark dataset
    (corpus + queries + ground truth qrels) into a unified list of rows for
    the evaluation pipeline.

    Data Structure (PythonDict):
        [
            # Corpus rows
            {"doc_id": "4983", "text": "...", "row_type": "corpus"},
            # Query rows
            {"doc_id": "q_0", "text": "...", "row_type": "query",
             "relevant_doc_ids": ["4983"], "relevance_scores": {"4983": 1}},
        ]

    The ``row_type`` field distinguishes corpus documents from queries so that
    ``RetrievalEvaluator`` can split them for cosine-similarity ranking.

    Usage:
        features = ["eval_docs"]
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
        """Match features named 'eval_docs' exactly."""
        if isinstance(feature_name, FeatureName):
            feature_name = feature_name.name
        return feature_name == "eval_docs"

    def input_features(self, options: Options, feature_name: FeatureName) -> None:
        """Root feature — no input features."""
        return None

    @classmethod
    @abstractmethod
    def _load_dataset(cls, options: Options) -> List[Dict[str, Any]]:
        """
        Load corpus + queries + qrels from local storage.

        Returns:
            Unified list of rows with ``row_type`` set to "corpus" or "query".
        """
        ...

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> List[Dict[str, Any]]:
        """Load and return dataset rows."""
        for feature in features.features:
            return cls._load_dataset(feature.options)
        return []
