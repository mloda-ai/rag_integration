"""FAISS-based retriever for similarity search."""

from __future__ import annotations

import json
import threading
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from mloda.user import Options
from mloda.provider import DefaultOptionKeys

from rag_integration.feature_groups.rag_pipeline.retrieval.base import BaseRetriever


class FaissRetriever(BaseRetriever):
    """
    FAISS-based retriever for nearest-neighbor search on a pre-built index.

    Loads a FAISS index from disk, performs search, returns top-k results
    with indices, distances, texts, and doc_ids.

    Supports index caching: once loaded, the index is cached at class level.

    Config-based matching:
        retrieval_method="faiss"

    Note: Caches the index at class level for performance. Loading is guarded by
    a lock so concurrent callers do not read the same index/metadata twice.
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

    # Class-level caches, each stored as a single (path, value) tuple so the
    # lock-free fast path reads it atomically (one attribute load) instead of two
    # fields that could be observed mid-update.
    _index_cache: Optional[Tuple[str, Any]] = None
    _metadata_cache: Optional[Tuple[str, Dict[str, Any]]] = None
    _cache_lock = threading.Lock()

    @classmethod
    def _load_index(cls, index_path: str) -> Any:
        """Load a FAISS index from file, with caching (thread-safe)."""
        import faiss

        # Fast path: single atomic read of the (path, index) cache.
        cache = cls._index_cache
        if cache is not None and cache[0] == index_path:
            return cache[1]

        with cls._cache_lock:
            # Re-check inside the lock: another thread may have loaded it.
            cache = cls._index_cache
            if cache is not None and cache[0] == index_path:
                return cache[1]

            index = faiss.read_index(index_path)
            cls._index_cache = (index_path, index)
            return index

    @classmethod
    def _load_metadata(cls, metadata_path: str) -> Dict[str, Any]:
        """Load metadata from JSON sidecar, with caching (thread-safe)."""
        # Fast path: single atomic read of the (path, metadata) cache.
        cache = cls._metadata_cache
        if cache is not None and cache[0] == metadata_path:
            return cache[1]

        with cls._cache_lock:
            # Re-check inside the lock: another thread may have loaded it.
            cache = cls._metadata_cache
            if cache is not None and cache[0] == metadata_path:
                return cache[1]

            with open(metadata_path, encoding="utf-8") as f:
                metadata: Dict[str, Any] = json.load(f)
            cls._metadata_cache = (metadata_path, metadata)
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
