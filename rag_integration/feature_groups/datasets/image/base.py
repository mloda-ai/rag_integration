"""Base class for image evaluation dataset sources."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

from mloda.provider import ComputeFramework, FeatureGroup, FeatureSet
from mloda.user import FeatureName, Options
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
    PythonDictFramework,
)
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_utils import homogenize_rows


class BaseImageDatasetSource(FeatureGroup):
    """
    Base class for image evaluation dataset sources.

    ROOT feature — no input features. Loads an image retrieval benchmark dataset
    (images + captions + ground-truth relevance) into a unified list of rows for
    the evaluation pipeline.

    Data Structure (PythonDict):
        [
            # Image (corpus) rows
            {"image_id": "123456", "image_data": <bytes>, "format": "jpeg",
             "row_type": "corpus"},
            # Caption (query) rows
            {"image_id": "cap_0", "image_data": None, "caption": "A dog running...",
             "row_type": "query", "relevant_image_ids": ["123456"]},
        ]

    Usage:
        features = ["eval_images"]
    """

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
        """Match features named 'eval_images' exactly."""
        return feature_name == "eval_images"

    def input_features(self, options: Options, feature_name: FeatureName) -> None:
        """Root feature — no input features."""
        return

    @classmethod
    @abstractmethod
    def _load_dataset(cls, options: Options) -> list[dict[str, Any]]:
        """
        Load images + captions + ground truth from local storage.

        Returns:
            Unified list of rows with ``row_type`` set to "corpus" or "query".
        """
        ...

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> list[dict[str, Any]]:
        """Load and return dataset rows with a uniform key schema."""
        for feature in features.features:
            return homogenize_rows(cls._load_dataset(feature.options))
        return []
