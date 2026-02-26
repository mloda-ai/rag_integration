"""Base class for retrieval feature groups."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, List, Optional, Set, Type, Union

from rag_integration.feature_groups.rag_pipeline.embedding.base import BaseEmbedder

from mloda.provider import DataCreator, FeatureGroup, ComputeFramework, FeatureSet
from mloda.user import Options, FeatureName
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
    PythonDictFramework,
)
from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys


class BaseRetriever(FeatureGroup):
    """
    Base class for retrieval feature groups.

    This is a ROOT feature (like BaseDocumentSource): it has no input features
    and produces search results from a pre-built vector index.

    Accepts EITHER:
        - query_embedding: Pre-embedded query vector (List[float])
        - query_text + embedding_method: Raw text to embed at query time

    Configuration via Options:
        - index_path: Path to the .faiss index file
        - metadata_path: Path to the _metadata.json sidecar
        - top_k: Number of results to return (default: 5)
        - query_embedding: Pre-embedded query vector
        - query_text: Raw text query (requires embedding_method)
        - embedding_method: Which embedder to use for query_text

    Output rows contain: indices, distances, texts, doc_ids
    """

    TOP_K = "top_k"
    QUERY_EMBEDDING = "query_embedding"
    QUERY_TEXT = "query_text"
    EMBEDDING_METHOD = "embedding_method"
    INDEX_PATH = "index_path"
    METADATA_PATH = "metadata_path"
    RETRIEVAL_METHOD = "retrieval_method"

    RETRIEVAL_METHODS: Dict[str, str] = {}

    PROPERTY_MAPPING = {
        RETRIEVAL_METHOD: {
            "explanation": "Which retriever implementation to use",
            DefaultOptionKeys.context: True,
        },
        TOP_K: {
            "explanation": "Number of results to return",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.default: 5,
        },
        QUERY_TEXT: {
            "explanation": "Raw text query to embed and search",
            DefaultOptionKeys.context: True,
        },
        INDEX_PATH: {
            "explanation": "Path to the FAISS index file",
            DefaultOptionKeys.context: True,
        },
        METADATA_PATH: {
            "explanation": "Path to the metadata JSON sidecar",
            DefaultOptionKeys.context: True,
        },
    }

    @classmethod
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFramework]]]:
        return {PythonDictFramework}

    @classmethod
    def input_data(cls) -> DataCreator:
        return DataCreator({"retrieved"})

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: Union[FeatureName, str],
        options: Options,
        data_access_collection: Any = None,
    ) -> bool:
        """Match features named 'retrieved' exactly."""
        if isinstance(feature_name, FeatureName):
            feature_name = feature_name.name
        return feature_name == "retrieved"

    def input_features(self, options: Options, feature_name: FeatureName) -> None:
        """Root feature: no input features."""
        return None

    @classmethod
    def _get_top_k(cls, options: Options) -> int:
        """Get top_k from options, default 5."""
        val = options.get(cls.TOP_K)
        return int(val) if val is not None else 5

    @classmethod
    def _embed_query(cls, query_text: str, embedding_method: str) -> List[float]:
        """
        Embed a query text string using the specified embedding method.

        Delegates to the existing embedder implementations.
        """
        from rag_integration.feature_groups.rag_pipeline.embedding.mock import MockEmbedder
        from rag_integration.feature_groups.rag_pipeline.embedding.hash_embed import HashEmbedder
        from rag_integration.feature_groups.rag_pipeline.embedding.tfidf import TfidfEmbedder

        embedders: Dict[str, Type[BaseEmbedder]] = {
            "mock": MockEmbedder,
            "hash": HashEmbedder,
            "tfidf": TfidfEmbedder,
        }

        embedder_cls = embedders.get(embedding_method)
        if embedder_cls is None:
            raise ValueError(
                f"Unknown embedding_method '{embedding_method}'. Supported methods: {list(embedders.keys())}"
            )

        results = embedder_cls._embed_texts([query_text], 384, "default")
        return results[0]

    @classmethod
    @abstractmethod
    def _search(
        cls,
        query_vector: List[float],
        top_k: int,
        options: Options,
    ) -> Dict[str, Any]:
        """
        Perform similarity search.

        Args:
            query_vector: The query embedding vector
            top_k: Number of results to return
            options: Options containing index_path, metadata_path, etc.

        Returns:
            Dict with keys: indices, distances, texts, doc_ids
        """
        ...

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> List[Dict[str, Any]]:
        """Run retrieval: embed query if needed, search index, return results."""
        for feature in features.features:
            options = feature.options

            # Get query vector (pre-embedded or embed from text)
            query_embedding_raw = options.get(cls.QUERY_EMBEDDING)
            query_vector: Optional[List[float]] = None

            if query_embedding_raw is not None:
                query_vector = list(query_embedding_raw)
            else:
                query_text = options.get(cls.QUERY_TEXT)
                embedding_method = options.get(cls.EMBEDDING_METHOD)
                if query_text is not None and embedding_method is not None:
                    query_vector = cls._embed_query(str(query_text), str(embedding_method))

            if query_vector is None:
                raise ValueError(
                    "FaissRetriever requires either 'query_embedding' or 'query_text' + 'embedding_method' in options."
                )

            top_k = cls._get_top_k(options)
            results = cls._search(query_vector, top_k, options)

            return [{"retrieved": results, **results}]

        return []
