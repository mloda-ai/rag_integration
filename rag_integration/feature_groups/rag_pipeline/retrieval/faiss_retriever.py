"""FAISS-based retriever for similarity search."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import numpy as np
from mloda.user import Options
from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys

from rag_integration.feature_groups.rag_pipeline.retrieval.base import BaseRetriever


class FaissRetriever(BaseRetriever):
    """
    FAISS-based retriever for nearest-neighbor search on a pre-built index.

    Loads a FAISS index from disk, performs search, returns top-k results
    with indices, distances, texts, and doc_ids.

    Supports index caching: once loaded, the index is cached at class level.

    Config-based matching:
        retrieval_method="faiss"

    Note: Caches the index at class level for performance. Not thread-safe.
    """

    RETRIEVAL_METHODS = {
        "faiss": "FAISS-based similarity search",
    }

    PROPERTY_MAPPING = {
        BaseRetriever.RETRIEVAL_METHOD: {
            "faiss": "FAISS-based similarity search",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: True,
        },
        BaseRetriever.TOP_K: {
            "explanation": "Number of results to return",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.default: 5,
        },
        BaseRetriever.QUERY_TEXT: {
            "explanation": "Raw text query to embed and search",
            DefaultOptionKeys.context: True,
        },
        BaseRetriever.INDEX_PATH: {
            "explanation": "Path to the FAISS index file",
            DefaultOptionKeys.context: True,
        },
        BaseRetriever.METADATA_PATH: {
            "explanation": "Path to the metadata JSON sidecar",
            DefaultOptionKeys.context: True,
        },
    }

    # Class-level cache for loaded index
    _cached_index: Optional[Any] = None
    _cached_index_path: Optional[str] = None
    _cached_metadata: Optional[Dict[str, Any]] = None
    _cached_metadata_path: Optional[str] = None

    @classmethod
    def _load_index(cls, index_path: str) -> Any:
        """Load a FAISS index from file, with caching."""
        import faiss

        if cls._cached_index is not None and cls._cached_index_path == index_path:
            return cls._cached_index

        index = faiss.read_index(index_path)
        cls._cached_index = index
        cls._cached_index_path = index_path
        return index

    @classmethod
    def _load_metadata(cls, metadata_path: str) -> Dict[str, Any]:
        """Load metadata from JSON sidecar, with caching."""
        if cls._cached_metadata is not None and cls._cached_metadata_path == metadata_path:
            return cls._cached_metadata

        with open(metadata_path) as f:
            raw = json.load(f)

        metadata: Dict[str, Any] = raw
        cls._cached_metadata = metadata
        cls._cached_metadata_path = metadata_path
        return metadata

    @classmethod
    def _search(
        cls,
        query_vector: List[float],
        top_k: int,
        options: Options,
    ) -> Dict[str, Any]:
        """
        Search the FAISS index for nearest neighbors.

        Returns:
            Dict with keys: indices, distances, texts, doc_ids
        """
        index_path = options.get(cls.INDEX_PATH)
        metadata_path = options.get(cls.METADATA_PATH)

        if index_path is None:
            raise ValueError("FaissRetriever requires 'index_path' in options.")

        index = cls._load_index(str(index_path))

        # Clamp top_k to index size
        effective_k = min(top_k, index.ntotal)

        # Search
        query_array = np.array([query_vector], dtype=np.float32)
        distances, indices = index.search(query_array, effective_k)

        result_indices = indices[0].tolist()
        result_distances = distances[0].tolist()

        # Load metadata if available
        result_texts: List[str] = []
        result_doc_ids: List[str] = []

        if metadata_path is not None:
            metadata = cls._load_metadata(str(metadata_path))
            texts = metadata.get("texts", [])
            doc_ids = metadata.get("doc_ids", [])

            for idx in result_indices:
                if 0 <= idx < len(texts):
                    result_texts.append(texts[idx])
                else:
                    result_texts.append("")
                if 0 <= idx < len(doc_ids):
                    result_doc_ids.append(doc_ids[idx])
                else:
                    result_doc_ids.append("")

        return {
            "indices": result_indices,
            "distances": result_distances,
            "texts": result_texts,
            "doc_ids": result_doc_ids,
        }
