"""Base class for embedding feature groups."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, List, Optional, Set, Type

from mloda.provider import BaseArtifact, ComputeFramework, FeatureGroup, FeatureSet, property_spec
from mloda.provider import FeatureChainParserMixin
from mloda.user import Feature
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
    PythonDictFramework,
)
from mloda.provider import DefaultOptionKeys

from rag_integration.feature_groups.columnar import columnar_to_rows


class BaseEmbedder(FeatureChainParserMixin, FeatureGroup):
    """
    Base class for text embedding feature groups.

    Converts text into dense vector representations for similarity search.

    Feature Naming Pattern:
        {in_feature}__embedded

    Examples:
        - docs__pii_redacted__chunked__deduped__embedded

    The embedding is stored as a list of floats in the output feature.

    ## Configuration-Based Creation

    Uses Options with proper group/context parameter separation:

    ```python
    feature = Feature(
        name="my_embedded",
        options=Options(
            context={
                "embedding_method": "mock",
                DefaultOptionKeys.in_features: "docs",
            }
        )
    )
    ```
    """

    # Configuration keys
    EMBEDDING_DIM = "embedding_dim"
    MODEL_NAME = "model_name"

    # Discriminator key for config-based feature matching
    EMBEDDING_METHOD = "embedding_method"

    # Supported embedding methods (implementations must define which they handle)
    EMBEDDING_METHODS = {
        "mock": "Deterministic mock embeddings for testing",
        "hash": "Feature hashing based embeddings",
        "tfidf": "TF-IDF based embeddings",
    }

    PREFIX_PATTERN = r".*__embedded$"

    MIN_IN_FEATURES = 1
    MAX_IN_FEATURES = 1

    PROPERTY_MAPPING = {
        EMBEDDING_METHOD: property_spec(
            "Algorithm used to embed text into vectors", strict=True, allowed_values=EMBEDDING_METHODS
        ),
        EMBEDDING_DIM: property_spec("Dimension of the embedding vectors", default=384),
        MODEL_NAME: property_spec("Name of the embedding model", default="default"),
        DefaultOptionKeys.in_features: property_spec("Source feature containing text to embed"),
    }

    @classmethod
    def compute_framework_rule(cls) -> Optional[Set[Type[ComputeFramework]]]:
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
        return int(dim) if dim is not None else 384

    @classmethod
    def _get_model_name(cls, feature: Feature) -> str:
        """Get model name from feature options."""
        name = feature.options.get(cls.MODEL_NAME)
        return str(name) if name is not None else "default"

    @staticmethod
    def artifact() -> Optional[Type[BaseArtifact]]:
        """
        Return the artifact class for this embedder.

        Subclasses can override this to enable artifact persistence.
        Returns None by default (no artifact support).
        """
        return None

    @classmethod
    @abstractmethod
    def _embed_texts(
        cls,
        texts: List[str],
        embedding_dim: int,
        model_name: str,
    ) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed
            embedding_dim: Dimension of output vectors
            model_name: Model identifier

        Returns:
            List of embedding vectors (each is a list of floats)
        """
        ...

    @classmethod
    def calculate_feature(cls, data: List[Dict[str, Any]], features: FeatureSet) -> List[Dict[str, Any]]:
        """Generate embeddings for the source feature, with optional artifact support."""
        artifact_cls = cls.artifact()

        # mloda 0.9.0 delivers columnar data; read it row-wise.
        rows = columnar_to_rows(data)

        for feature in features.features:
            source_feature = cls._get_source_feature_name(feature)
            embedding_dim = cls._get_embedding_dim(feature)
            model_name = cls._get_model_name(feature)
            feature_name = feature.name

            # Artifact key must be the feature name for mloda's artifact detection to work
            artifact_key = feature_name

            # Try to load from artifact if available
            embeddings: Optional[List[List[float]]] = None
            if artifact_cls is not None:
                try:
                    loaded = artifact_cls.load_embedding_artifact(features, artifact_key)  # type: ignore[attr-defined]
                    if loaded is not None:
                        embeddings = loaded.get("embeddings")
                        model_name = loaded.get("model_name", model_name)
                except ValueError:
                    # Artifact not found, will compute embeddings
                    pass

            if embeddings is None:
                # Extract texts from source feature
                texts = []
                for row in rows:
                    if source_feature in row:
                        texts.append(str(row[source_feature]))
                    elif "text" in row:
                        texts.append(str(row["text"]))
                    else:
                        texts.append("")

                # Generate embeddings
                embeddings = cls._embed_texts(texts, embedding_dim, model_name)

                # Save artifact only if we computed new embeddings (not loaded from artifact)
                if artifact_cls is not None:
                    artifact_data = {
                        "embeddings": embeddings,
                        "model_name": model_name,
                        "embedding_dim": len(embeddings[0]) if embeddings else embedding_dim,
                        "num_texts": len(texts),
                    }
                    artifact_cls.save_embedding_artifact(features, artifact_key, artifact_data)  # type: ignore[attr-defined]

            # Add results to data
            for i, row in enumerate(rows):
                row[feature_name] = embeddings[i]
                row["embedding_dim"] = len(embeddings[i])
                row["embedding_model"] = model_name

        return rows
