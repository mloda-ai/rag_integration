"""FAISS IVF (inverted file) vector indexer."""

from __future__ import annotations

from typing import Any, List

import numpy as np
from mloda.provider import DefaultOptionKeys, property_spec

from rag_integration.feature_groups.rag_pipeline.vector_store.base import BaseVectorStore


class FaissIVFIndexer(BaseVectorStore):
    """
    FAISS IndexIVFFlat indexer for approximate nearest-neighbor search.

    Uses inverted file index with flat quantizer. Requires training.
    Configurable nlist (number of clusters) and nprobe (clusters to search).
    Clamps nlist to not exceed vector count.

    Config-based matching:
        index_method="ivf"
    """

    INDEX_METHODS = {
        "ivf": "Approximate search using IndexIVFFlat",
    }

    NLIST = "nlist"
    NPROBE = "nprobe"

    PROPERTY_MAPPING = {
        BaseVectorStore.INDEX_METHOD: property_spec(
            "FAISS index type backing the vector store",
            strict=True,
            allowed_values={"ivf": "Approximate search using IndexIVFFlat"},
        ),
        NLIST: property_spec("Number of clusters for IVF index", default=10),
        NPROBE: property_spec("Number of clusters to search at query time", default=3),
        DefaultOptionKeys.in_features: property_spec("Source feature containing embedding vectors to index"),
    }

    @classmethod
    def _index_type_name(cls) -> str:
        return "ivf_flat"

    @classmethod
    def _build_index(cls, embeddings: List[List[float]], dimension: int) -> Any:
        """Build a FAISS IndexIVFFlat from embeddings."""
        import faiss

        vectors = np.array(embeddings, dtype=np.float32)
        n_vectors = vectors.shape[0]

        # Clamp nlist to not exceed vector count
        nlist = min(10, n_vectors)

        quantizer = faiss.IndexFlatL2(dimension)
        index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
        index.train(vectors)
        index.add(vectors)

        # Set nprobe for search
        index.nprobe = min(3, nlist)

        return index
