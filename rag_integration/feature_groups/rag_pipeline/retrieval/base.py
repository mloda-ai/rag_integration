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
from mloda.provider import DefaultOptionKeys


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

    Output rows contain: indices, distances, texts, doc_ids, and the canonical
    ranked-passage list under PASSAGES_KEY (same shape as the retrieve
    connector family).
    """

    # Mirrors BaseRetrieveConnector.ROOT_FEATURE_NAME as a literal so the stage
    # layer does not import the connectors layer; pinned by the parity test.
    PASSAGES_KEY = "retrieved_passages"

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
    def compute_framework_rule(cls) -> Optional[Set[Type[ComputeFramework]]]:
        return {PythonDictFramework}

    @classmethod
    def input_data(cls) -> DataCreator:
        return DataCreator({"retrieved", cls.PASSAGES_KEY})

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: Union[FeatureName, str],
        options: Options,
        data_access_collection: Any = None,
    ) -> bool:
        """Match 'retrieved', or PASSAGES_KEY for index-backed options.

        Serving PASSAGES_KEY makes migration a pure option swap. The gate on
        index_path, plus yielding when an explicit retrieve-connector selector
        is present, keeps the stage and the connector family from both
        claiming one request.
        """
        name = str(feature_name)
        if name == "retrieved":
            return True
        if name != cls.PASSAGES_KEY:
            return False
        if options.get("retrieve_backend") is not None:
            return False
        return options.get(cls.INDEX_PATH) is not None

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

    # Cosine scores this close to zero are float32 noise around orthogonality.
    _SCORE_EPSILON = 1e-6

    @classmethod
    def _to_passages(cls, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert a :meth:`_search` result into the retrieve family's contract.

        ``[{doc_id, text, score, rank}]``, best first, only positive scores.
        The repo's embedders L2-normalize, so ``score = 1 - distance / 2`` is
        the cosine, the same scale the dense connector emits; raw distances
        stay unfiltered in the row. Blank or missing ``doc_id`` falls back to
        the index position, missing ``text`` to ``""``.
        """
        indices = results.get("indices", [])
        distances = results.get("distances", [])
        texts = results.get("texts", [])
        doc_ids = results.get("doc_ids", [])

        passages: List[Dict[str, Any]] = []
        for i, distance in enumerate(distances):
            score = 1.0 - float(distance) / 2.0
            if score <= cls._SCORE_EPSILON:
                continue
            raw_doc_id = doc_ids[i] if i < len(doc_ids) else None
            if raw_doc_id is not None and str(raw_doc_id) != "":
                doc_id = str(raw_doc_id)
            else:
                doc_id = str(indices[i]) if i < len(indices) else str(i)
            text = str(texts[i]) if i < len(texts) else ""
            passages.append(
                {
                    "doc_id": doc_id,
                    "text": text,
                    "score": score,
                    "rank": len(passages),
                }
            )
        return passages

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

            return [{"retrieved": results, **results, cls.PASSAGES_KEY: cls._to_passages(results)}]

        return []
