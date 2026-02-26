"""FAISS Flat (exact search) vector indexer."""

from __future__ import annotations

from typing import Any, List

import numpy as np
from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys

from rag_integration.feature_groups.rag_pipeline.vector_store.base import BaseVectorStore


class FaissFlatIndexer(BaseVectorStore):
    """
    FAISS IndexFlatL2 indexer for exact nearest-neighbor search.

    Uses brute-force L2 distance. Best for small datasets (<100k vectors).
    No training required.

    Config-based matching:
        index_method="flat"
    """

    PROPERTY_MAPPING = {
        BaseVectorStore.INDEX_METHOD: {
            "flat": "Exact search using IndexFlatL2",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: True,
        },
        DefaultOptionKeys.in_features: {
            "explanation": "Source feature containing embedding vectors to index",
            DefaultOptionKeys.context: True,
        },
    }

    @classmethod
    def _index_type_name(cls) -> str:
        return "flat_l2"

    @classmethod
    def _build_index(cls, embeddings: List[List[float]], dimension: int) -> Any:
        """Build a FAISS IndexFlatL2 from embeddings."""
        import faiss

        index = faiss.IndexFlatL2(dimension)
        vectors = np.array(embeddings, dtype=np.float32)
        index.add(vectors)
        return index
