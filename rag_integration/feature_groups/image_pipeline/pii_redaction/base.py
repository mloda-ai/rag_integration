"""Base class for image PII redaction feature groups."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, List, Set, Tuple, Type, Union

from mloda.provider import FeatureGroup, ComputeFramework, FeatureSet
from mloda.provider import FeatureChainParserMixin
from mloda.user import Feature, FeatureName, Options
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
    PythonDictFramework,
)
from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys


class BaseImagePIIRedactor(FeatureChainParserMixin, FeatureGroup):
    """
    Base class for image PII redaction feature groups.

    Redacts personally identifiable information from images by applying
    redaction to specified bounding box regions. Regions can be provided
    via options or detected by providers.

    Feature Naming Pattern:
        {in_feature}__pii_redacted

    Examples:
        - image_docs__pii_redacted

    PII Region Format:
        Each region is a dict with:
        - "bbox": [x1, y1, x2, y2] pixel coordinates
        - "type": PII type label (e.g., "FACE", "TEXT", "LICENSE_PLATE")

    ## Configuration-Based Creation

    Uses Options with proper group/context parameter separation:

    ```python
    feature = Feature(
        name="my_redacted",
        options=Options(
            context={
                "image_redaction_method": "blur",
                DefaultOptionKeys.in_features: "image_docs",
                "pii_regions": [{"bbox": [10, 10, 100, 100], "type": "FACE"}],
            }
        )
    )
    ```
    """

    # Configuration keys
    PII_REGIONS = "pii_regions"

    # Discriminator key for config-based feature matching
    IMAGE_REDACTION_METHOD = "image_redaction_method"

    # Supported redaction methods
    REDACTION_METHODS = {
        "blur": "Gaussian blur over PII regions",
        "pixel": "Pixelate PII regions",
        "solid": "Solid color fill over PII regions",
    }

    # Supported PII region types
    SUPPORTED_PII_TYPES = {
        "FACE": "Human faces",
        "TEXT": "Text regions containing PII",
        "LICENSE_PLATE": "Vehicle license plates",
        "ALL": "All detected PII regions",
    }

    PREFIX_PATTERN = r".*__pii_redacted$"

    MIN_IN_FEATURES = 1
    MAX_IN_FEATURES = 1

    PROPERTY_MAPPING = {
        IMAGE_REDACTION_METHOD: {
            **REDACTION_METHODS,
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: True,
        },
        PII_REGIONS: {
            "explanation": "List of PII region dicts with 'bbox' [x1,y1,x2,y2] and 'type'",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.default: [],
        },
        DefaultOptionKeys.in_features: {
            "explanation": "Source feature containing images to redact",
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
    def _get_pii_regions(cls, feature: Feature) -> List[Dict[str, Any]]:
        """Get PII regions from feature options."""
        regions = feature.options.get(cls.PII_REGIONS)
        if regions is None:
            return []
        if not isinstance(regions, list):
            return [regions]
        return regions

    @classmethod
    @abstractmethod
    def _redact_region(
        cls,
        image_data: bytes,
        image_format: str,
        regions: List[Dict[str, Any]],
    ) -> bytes:
        """
        Redact PII regions in an image.

        Args:
            image_data: Raw image bytes
            image_format: Image format (png, jpeg, etc.)
            regions: List of region dicts with 'bbox' and 'type'

        Returns:
            Redacted image as bytes
        """
        ...

    @classmethod
    def calculate_feature(cls, data: List[Dict[str, Any]], features: FeatureSet) -> List[Dict[str, Any]]:
        """Perform PII redaction on images, processing row by row for memory efficiency."""
        for feature in features.features:
            source_feature = cls._get_source_feature_name(feature)
            regions = cls._get_pii_regions(feature)
            feature_name = feature.get_name()

            for row in data:
                image_data = row.get("image_data", b"")
                image_format = row.get("format", "png")

                if not isinstance(image_data, bytes):
                    image_data = bytes(image_data) if image_data else b""

                if regions and image_data:
                    redacted = cls._redact_region(image_data, image_format, regions)
                else:
                    # No regions to redact — pass through
                    redacted = image_data

                row[feature_name] = redacted
                row["pii_redacted"] = len(regions) > 0
                row["pii_regions_count"] = len(regions)

        return data
