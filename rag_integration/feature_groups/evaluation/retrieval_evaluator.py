"""RetrievalEvaluator feature group.

Computes Recall@1, @5, @10 over a mixed corpus+query embedding set using
brute-force cosine similarity. No external vector index required.
"""

from __future__ import annotations

from typing import Any, Dict, List, Set, Type, Union

from mloda.provider import FeatureGroup, ComputeFramework, FeatureSet
from mloda.provider import FeatureChainParserMixin
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
    PythonDictFramework,
)
from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys

from rag_integration.feature_groups.evaluation.metrics import mean_recall_at_k


class RetrievalEvaluator(FeatureChainParserMixin, FeatureGroup):
    """
    Evaluates retrieval quality using brute-force cosine similarity.

    Consumes rows produced by an embedding step where each row has a
    ``row_type`` field (``"corpus"`` or ``"query"``) and a vector stored
    under the source feature name key.

    For text (SciFact):  query rows carry ``relevant_doc_ids``
    For image (Flickr30K): query rows carry ``relevant_image_ids``

    Feature Naming Pattern::

        {in_feature}__evaluated

    Example::

        eval_docs__embedded__evaluated

    Algorithm:
        1. Split rows into corpus set and query set by ``row_type``.
        2. Build embedding matrices (unit-normalised → dot product = cosine sim).
        3. sims = query_matrix @ corpus_matrix.T  (shape: Q × N)
        4. ranked = argsort(-sims, axis=1)         (each row is a query's ranking)
        5. Compute Recall@1, @5, @10 per query; return mean over all queries.

    Returns a single aggregate row::

        [{"recall@1": float, "recall@5": float, "recall@10": float,
          "num_queries": int, "num_corpus": int}]
    """

    PREFIX_PATTERN = r".*__evaluated$"
    MIN_IN_FEATURES = 1
    MAX_IN_FEATURES = 1

    PROPERTY_MAPPING = {
        DefaultOptionKeys.in_features: {
            "explanation": "Source feature containing embedded corpus + query rows",
            DefaultOptionKeys.context: True,
        },
    }

    @classmethod
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFramework]]]:
        return {PythonDictFramework}

    @classmethod
    def calculate_feature(cls, data: List[Dict[str, Any]], features: FeatureSet) -> List[Dict[str, Any]]:
        """Compute Recall@K over the embedded corpus and query rows."""
        try:
            import numpy as np
        except ImportError as e:
            raise ImportError("numpy is required for RetrievalEvaluator. Install with: pip install numpy") from e

        for feature in features.features:
            source_feature = cls._extract_source_features(feature)[0]
            feature_name = feature.get_name()

            corpus_rows = [r for r in data if r.get("row_type") == "corpus"]
            query_rows = [r for r in data if r.get("row_type") == "query"]

            if not corpus_rows or not query_rows:
                metrics: Dict[str, Any] = {
                    "recall@1": 0.0,
                    "recall@5": 0.0,
                    "recall@10": 0.0,
                    "num_queries": len(query_rows),
                    "num_corpus": len(corpus_rows),
                }
                return [{feature_name: metrics, **metrics}]

            # Build embedding matrices
            corpus_embeddings = [r[source_feature] for r in corpus_rows]
            query_embeddings = [r[source_feature] for r in query_rows]

            corpus_matrix = np.array(corpus_embeddings, dtype=np.float32)  # N × D
            query_matrix = np.array(query_embeddings, dtype=np.float32)  # Q × D

            # Cosine similarity (vectors are unit-normalised → dot product suffices)
            sims = query_matrix @ corpus_matrix.T  # Q × N

            # Map corpus index → doc/image id
            corpus_ids = [cls._get_id(r) for r in corpus_rows]

            # Ground-truth: query_id → set of relevant corpus ids
            query_relevant: Dict[str, Set[str]] = {}
            query_ranked: Dict[str, List[str]] = {}

            for q_idx, q_row in enumerate(query_rows):
                q_id = cls._get_id(q_row)
                relevant_ids = set(q_row.get("relevant_doc_ids", []) + q_row.get("relevant_image_ids", []))
                query_relevant[q_id] = relevant_ids

                ranked_indices = np.argsort(-sims[q_idx]).tolist()
                query_ranked[q_id] = [corpus_ids[i] for i in ranked_indices]

            r1 = mean_recall_at_k(query_relevant, query_ranked, k=1)
            r5 = mean_recall_at_k(query_relevant, query_ranked, k=5)
            r10 = mean_recall_at_k(query_relevant, query_ranked, k=10)

            metrics = {
                "recall@1": round(r1, 4),
                "recall@5": round(r5, 4),
                "recall@10": round(r10, 4),
                "num_queries": len(query_rows),
                "num_corpus": len(corpus_rows),
            }
            # Store metrics under the feature name key so mloda's column selector
            # can find this result row, and also as flat keys for direct access.
            return [{feature_name: metrics, **metrics}]

        return []

    @staticmethod
    def _get_id(row: Dict[str, Any]) -> str:
        """Return the identifier field for a row (doc_id or image_id)."""
        return str(row.get("doc_id") or row.get("image_id") or "")
