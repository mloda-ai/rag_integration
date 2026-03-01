"""FAISS HNSW (hierarchical navigable small world) vector indexer."""

from __future__ import annotations

from typing import Any, List

import numpy as np
from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys

from rag_integration.feature_groups.rag_pipeline.vector_store.base import BaseVectorStore


class FaissHNSWIndexer(BaseVectorStore):
    """
    FAISS IndexHNSWFlat indexer for graph-based approximate nearest-neighbor search.

    Uses Hierarchical Navigable Small World graph. No training required.
    Configurable M (number of connections) and efConstruction (construction quality).

    Config-based matching:
        index_method="hnsw"
    """

    INDEX_METHODS = {
        "hnsw": "Graph-based ANN using IndexHNSWFlat",
    }

    HNSW_M = "hnsw_m"
    HNSW_EF_CONSTRUCTION = "hnsw_ef_construction"

    PROPERTY_MAPPING = {
        BaseVectorStore.INDEX_METHOD: {
            "hnsw": "Graph-based ANN using IndexHNSWFlat",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: True,
        },
        HNSW_M: {
            "explanation": "Number of connections per node in HNSW graph",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.default: 32,
        },
        HNSW_EF_CONSTRUCTION: {
            "explanation": "Construction quality parameter for HNSW",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.default: 40,
        },
        DefaultOptionKeys.in_features: {
            "explanation": "Source feature containing embedding vectors to index",
            DefaultOptionKeys.context: True,
        },
    }

    @classmethod
    def _index_type_name(cls) -> str:
        return "hnsw_flat"

    @classmethod
    def _build_index(cls, embeddings: List[List[float]], dimension: int) -> Any:
        """Build a FAISS IndexHNSWFlat from embeddings."""
        import faiss

        m = 32
        index = faiss.IndexHNSWFlat(dimension, m)
        index.hnsw.efConstruction = 40

        vectors = np.array(embeddings, dtype=np.float32)
        index.add(vectors)
        return index
