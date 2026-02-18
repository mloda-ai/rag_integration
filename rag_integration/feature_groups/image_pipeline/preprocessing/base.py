"""Base class for image preprocessing feature groups."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, List, Set, Type, Union

from mloda.provider import FeatureGroup, ComputeFramework, FeatureSet
from mloda.provider import FeatureChainParserMixin
from mloda.user import Feature, FeatureName, Options
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
    PythonDictFramework,
)
from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys


class BaseImagePreprocessor(FeatureChainParserMixin, FeatureGroup):
    """
    Base class for image preprocessing feature groups.

    Transforms images for downstream processing (embedding, storage).
    Unlike text chunking which expands rows, image preprocessing
    transforms each image in place (1 image in -> 1 image out).

    Feature Naming Pattern:
        {in_feature}__preprocessed

    Examples:
        - image_docs__pii_redacted__preprocessed
        - image_docs__preprocessed

    ## Configuration-Based Creation

    Uses Options with proper group/context parameter separation:

    ```python
    feature = Feature(
        name="my_preprocessed",
        options=Options(
            context={
                "preprocessing_method": "resize",
                DefaultOptionKeys.in_features: "image_docs",
                "target_size": [224, 224],
            }
        )
    )
    ```
    """

    # Configuration keys
    TARGET_SIZE = "target_size"

    # Discriminator key for config-based feature matching
    PREPROCESSING_METHOD = "preprocessing_method"

    # Supported preprocessing methods
    PREPROCESSING_METHODS = {
        "resize": "Resize images to target dimensions",
        "normalize": "Normalize pixel values to [0, 1] range",
        "thumbnail": "Generate thumbnail preserving aspect ratio",
    }

    PREFIX_PATTERN = r".*__preprocessed$"

    MIN_IN_FEATURES = 1
    MAX_IN_FEATURES = 1

    PROPERTY_MAPPING = {
        PREPROCESSING_METHOD: {
            **PREPROCESSING_METHODS,
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: True,
        },
        TARGET_SIZE: {
            "explanation": "Target size as [width, height] in pixels",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.default: [224, 224],
        },
        DefaultOptionKeys.in_features: {
            "explanation": "Source feature containing images to preprocess",
            DefaultOptionKeys.context: True,
        },
    }

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: str | FeatureName,
        options: Options,
        data_access_collection: Any = None,
    ) -> bool:
        """Match feature using mixin logic, catching ValueError for strict validation."""
        try:
            return bool(super().match_feature_group_criteria(feature_name, options, data_access_collection))
        except ValueError:
            return False

    @classmethod
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFramework]]]:
        return {PythonDictFramework}

    @classmethod
    def _get_source_feature_name(cls, feature: Feature) -> str:
        """Extract source feature name from the feature."""
        source_features = cls._extract_source_features(feature)
        return source_features[0]

    @classmethod
    def _get_target_size(cls, feature: Feature) -> List[int]:
        """Get target size from feature options."""
        size = feature.options.get(cls.TARGET_SIZE)
        if size is None:
            return [224, 224]
        if isinstance(size, list):
            return [int(s) for s in size]
        return [224, 224]

    @classmethod
    @abstractmethod
    def _preprocess_image(
        cls,
        image_data: bytes,
        image_format: str,
        target_size: List[int],
    ) -> bytes:
        """
        Preprocess a single image.

        Args:
            image_data: Raw image bytes
            image_format: Image format (png, jpeg, etc.)
            target_size: Target dimensions [width, height]

        Returns:
            Preprocessed image as bytes
        """
        ...

    @classmethod
    def calculate_feature(cls, data: List[Dict[str, Any]], features: FeatureSet) -> List[Dict[str, Any]]:
        """Preprocess images row by row for memory efficiency."""
        for feature in features.features:
            source_feature = cls._get_source_feature_name(feature)
            target_size = cls._get_target_size(feature)
            feature_name = feature.get_name()

            for row in data:
                image_data = row.get("image_data", b"")
                image_format = row.get("format", "png")

                if not isinstance(image_data, bytes):
                    image_data = bytes(image_data) if image_data else b""

                if image_data:
                    preprocessed = cls._preprocess_image(image_data, image_format, target_size)
                else:
                    preprocessed = image_data

                row[feature_name] = preprocessed
                row["preprocessed_size"] = target_size

        return data
