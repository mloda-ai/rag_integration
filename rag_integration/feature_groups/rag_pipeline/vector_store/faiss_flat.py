"""FAISS Flat (exact search) vector indexer."""

from __future__ import annotations

from typing import Any, List

import numpy as np
from mloda.provider import DefaultOptionKeys, property_spec

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
        BaseVectorStore.INDEX_METHOD: property_spec(
            "FAISS index type backing the vector store",
            strict=True,
            allowed_values={"flat": "Exact search using IndexFlatL2"},
        ),
        DefaultOptionKeys.in_features: property_spec("Source feature containing embedding vectors to index"),
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
