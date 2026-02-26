"""Retrieval feature groups."""

from rag_integration.feature_groups.rag_pipeline.retrieval.base import BaseRetriever
from rag_integration.feature_groups.rag_pipeline.retrieval.faiss_retriever import FaissRetriever

__all__ = [
    "BaseRetriever",
    "FaissRetriever",
]
