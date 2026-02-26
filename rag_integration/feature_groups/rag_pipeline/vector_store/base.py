"""Base class for vector store feature groups."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, List, Optional, Set, Type, Union

from mloda.provider import BaseArtifact, FeatureGroup, ComputeFramework, FeatureSet
from mloda.provider import FeatureChainParserMixin
from mloda.user import Feature
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
    PythonDictFramework,
)
from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys

from rag_integration.feature_groups.rag_pipeline.vector_store.vector_store_artifact import VectorStoreArtifact


class BaseVectorStore(FeatureChainParserMixin, FeatureGroup):
    """
    Base class for vector store (indexing) feature groups.

    Builds a FAISS index from embedding vectors and persists it via VectorStoreArtifact.

    Feature Naming Pattern:
        {in_feature}__indexed

    Examples:
        - docs__pii_redacted__chunked__deduped__embedded__indexed

    Each row gets metadata: vector_id, index_type, index_size.
    The index itself is stored in the artifact.

    Configuration:
        index_method: Which indexer to use (e.g. "flat", "ivf", "hnsw")
    """

    INDEX_METHOD = "index_method"

    INDEX_METHODS = {
        "flat": "Exact search using IndexFlatL2",
    }

    PREFIX_PATTERN = r".*__indexed$"

    MIN_IN_FEATURES = 1
    MAX_IN_FEATURES = 1

    PROPERTY_MAPPING = {
        INDEX_METHOD: {
            **INDEX_METHODS,
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: True,
        },
        DefaultOptionKeys.in_features: {
            "explanation": "Source feature containing embedding vectors to index",
            DefaultOptionKeys.context: True,
        },
    }

    @classmethod
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFramework]]]:
        return {PythonDictFramework}

    @staticmethod
    def artifact() -> Optional[Type[BaseArtifact]]:
        """Return VectorStoreArtifact for persisting FAISS indices."""
        return VectorStoreArtifact

    @classmethod
    def _get_source_feature_name(cls, feature: Feature) -> str:
        """Extract source feature name from the feature."""
        source_features = cls._extract_source_features(feature)
        return source_features[0]

    @classmethod
    @abstractmethod
    def _build_index(cls, embeddings: List[List[float]], dimension: int) -> Any:
        """
        Build a FAISS index from embedding vectors.

        Args:
            embeddings: List of embedding vectors
            dimension: Dimension of the vectors

        Returns:
            A FAISS index object
        """
        ...

    @classmethod
    @abstractmethod
    def _index_type_name(cls) -> str:
        """Return a string identifying the index type (e.g. 'flat_l2', 'ivf_flat')."""
        ...

    @classmethod
    def calculate_feature(cls, data: List[Dict[str, Any]], features: FeatureSet) -> List[Dict[str, Any]]:
        """Build FAISS index from embeddings, save via artifact, attach row metadata."""
        artifact_cls = cls.artifact()

        for feature in features.features:
            source_feature = cls._get_source_feature_name(feature)
            feature_name = feature.get_name()
            artifact_key = feature_name

            # Try to load from artifact
            loaded_index = None
            if artifact_cls is not None:
                try:
                    loaded = artifact_cls.load_vector_store_artifact(features, artifact_key)  # type: ignore[attr-defined]
                    if loaded is not None:
                        loaded_index = loaded.get("index")
                except ValueError:
                    pass

            if loaded_index is not None:
                index = loaded_index
                index_size = index.ntotal
            else:
                # Extract embeddings from rows
                embeddings: List[List[float]] = []
                texts: List[str] = []
                doc_ids: List[str] = []

                for row in data:
                    embedding = row.get(source_feature)
                    if embedding is not None and isinstance(embedding, list):
                        embeddings.append(embedding)
                    else:
                        embeddings.append([])

                    # Collect text and doc_id for metadata sidecar
                    text = row.get("text", row.get(source_feature.rsplit("__", 1)[0], ""))
                    if isinstance(text, list):
                        text = ""
                    texts.append(str(text))
                    doc_ids.append(str(row.get("doc_id", f"row_{len(doc_ids)}")))

                if not embeddings or not embeddings[0]:
                    for i, row in enumerate(data):
                        row[feature_name] = None
                        row["vector_id"] = i
                        row["index_type"] = cls._index_type_name()
                        row["index_size"] = 0
                    return data

                dimension = len(embeddings[0])
                index = cls._build_index(embeddings, dimension)
                index_size = index.ntotal

                # Save artifact
                if artifact_cls is not None:
                    artifact_cls.save_vector_store_artifact(  # type: ignore[attr-defined]
                        features, artifact_key, index, texts, doc_ids
                    )

            # Attach row metadata
            for i, row in enumerate(data):
                row[feature_name] = i  # vector_id as the feature value
                row["vector_id"] = i
                row["index_type"] = cls._index_type_name()
                row["index_size"] = index_size

        return data
