"""Base class for image embedding feature groups."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, List, Optional, Set, Type, Union

from mloda.provider import BaseArtifact, FeatureGroup, ComputeFramework, FeatureSet
from mloda.provider import FeatureChainParserMixin
from mloda.user import Feature, FeatureName, Options
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
    PythonDictFramework,
)
from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys


class BaseImageEmbedder(FeatureChainParserMixin, FeatureGroup):
    """
    Base class for image embedding feature groups.

    Converts images into dense vector representations for similarity search.
    Processes images row by row for memory efficiency.

    Feature Naming Pattern:
        {in_feature}__embedded

    Examples:
        - image_docs__pii_redacted__preprocessed__deduped__embedded

    The embedding is stored as a list of floats in the output feature.

    ## Configuration-Based Creation

    Uses Options with proper group/context parameter separation:

    ```python
    feature = Feature(
        name="my_embedded",
        options=Options(
            context={
                "image_embedding_method": "mock",
                DefaultOptionKeys.in_features: "image_docs",
            }
        )
    )
    ```
    """

    # Configuration keys
    EMBEDDING_DIM = "embedding_dim"
    MODEL_NAME = "model_name"

    # Discriminator key for config-based feature matching
    IMAGE_EMBEDDING_METHOD = "image_embedding_method"

    # Supported embedding methods
    EMBEDDING_METHODS = {
        "mock": "Deterministic mock image embeddings for testing",
        "hash": "Feature hashing based image embeddings",
        "clip": "CLIP model image embeddings",
    }

    PREFIX_PATTERN = r".*__embedded$"

    MIN_IN_FEATURES = 1
    MAX_IN_FEATURES = 1

    PROPERTY_MAPPING = {
        IMAGE_EMBEDDING_METHOD: {
            **EMBEDDING_METHODS,
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: True,
        },
        EMBEDDING_DIM: {
            "explanation": "Dimension of the embedding vectors",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.default: 512,
        },
        MODEL_NAME: {
            "explanation": "Name of the embedding model",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.default: "default",
        },
        DefaultOptionKeys.in_features: {
            "explanation": "Source feature containing images to embed",
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
    def _get_embedding_dim(cls, feature: Feature) -> int:
        """Get embedding dimension from feature options."""
        dim = feature.options.get(cls.EMBEDDING_DIM)
        return int(dim) if dim is not None else 512

    @classmethod
    def _get_model_name(cls, feature: Feature) -> str:
        """Get model name from feature options, falling back to PROPERTY_MAPPING default."""
        name = feature.options.get(cls.MODEL_NAME)
        if name is not None:
            return str(name)
        # Check subclass PROPERTY_MAPPING for a provider-specific default
        mapping = cls.PROPERTY_MAPPING.get(cls.MODEL_NAME, {})
        default = mapping.get(DefaultOptionKeys.default, "default")
        return str(default)

    @staticmethod
    def artifact() -> Optional[Type[BaseArtifact]]:
        """Return the artifact class for this embedder. None by default."""
        return None

    @classmethod
    @abstractmethod
    def _embed_image(
        cls,
        image_data: bytes,
        embedding_dim: int,
        model_name: str,
    ) -> List[float]:
        """
        Generate embedding for a single image.

        Args:
            image_data: Raw image bytes
            embedding_dim: Dimension of output vector
            model_name: Model identifier

        Returns:
            Embedding vector (list of floats)
        """
        ...

    @classmethod
    def calculate_feature(cls, data: List[Dict[str, Any]], features: FeatureSet) -> List[Dict[str, Any]]:
        """Generate embeddings for images, processing row by row for memory efficiency."""
        for feature in features.features:
            cls._get_source_feature_name(feature)
            embedding_dim = cls._get_embedding_dim(feature)
            model_name = cls._get_model_name(feature)
            feature_name = feature.get_name()

            for row in data:
                image_data = row.get("image_data", b"")
                if not isinstance(image_data, bytes):
                    image_data = bytes(image_data) if image_data else b""

                embedding = cls._embed_image(image_data, embedding_dim, model_name)

                row[feature_name] = embedding
                row["embedding_dim"] = len(embedding)
                row["embedding_model"] = model_name

        return data
