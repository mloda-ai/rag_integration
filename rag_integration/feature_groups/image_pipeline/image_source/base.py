"""Base class for image source feature groups."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, List, Set, Type, Union

from mloda.provider import FeatureGroup, ComputeFramework, FeatureSet
from mloda.user import Options, FeatureName
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
    PythonDictFramework,
)


class BaseImageSource(FeatureGroup):
    """
    Base class for image source feature groups.

    This is a ROOT feature - it has no input features and provides
    the initial images for the pipeline.

    Data Structure (PythonDict):
        [
            {"image_id": "img_001", "image_data": <bytes>, "format": "png", "metadata": {...}},
            {"image_id": "img_002", "image_data": <bytes>, "format": "jpeg", "metadata": {...}},
        ]

    Image data is stored as raw bytes. Each row represents one image.

    Usage:
        features = ["image_docs"]  # Returns raw image documents
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
        """Match features named 'image_docs' exactly."""
        if isinstance(feature_name, FeatureName):
            feature_name = feature_name.name
        return feature_name == "image_docs"

    def input_features(self, options: Options, feature_name: FeatureName) -> None:
        """Root feature - no input features."""
        return None

    @classmethod
    @abstractmethod
    def _load_images(cls, options: Options) -> List[Dict[str, Any]]:
        """
        Load images from the source.

        Args:
            options: Options containing source configuration

        Returns:
            List of image dictionaries with 'image_id', 'image_data' (bytes),
            'format', and optional 'metadata'
        """
        ...

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> List[Dict[str, Any]]:
        """Load and return images."""
        for feature in features.features:
            images = cls._load_images(feature.options)
            return images
        return []
