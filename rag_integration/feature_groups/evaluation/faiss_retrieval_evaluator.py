"""FaissRetrievalEvaluator feature group.

Computes Recall@1, @5, @10 through the full ingestion pipeline
(chunking → deduplication → embedding → FAISS indexing) using an
in-memory FAISS IndexFlatIP for batch evaluation.

Sits at the end of the chain::

    {source}__indexed__evaluated

Example::

    eval_docs__chunked__deduped__embedded__indexed__evaluated

Unlike RetrievalEvaluator (brute-force numpy), this evaluator builds a
real FAISS index from the corpus embeddings and measures retrieval quality
through the same FAISS search path that production queries will use.

Chunked corpora are handled at doc level: Recall@K is satisfied when *any*
chunk from the relevant document appears in the top-K results.
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


class FaissRetrievalEvaluator(FeatureChainParserMixin, FeatureGroup):
    """
    Evaluates retrieval quality via FAISS through the full ingestion pipeline.

    Consumes rows produced by a FAISS indexing step (``__indexed``).  Each row
    must have a ``row_type`` field (``"corpus"`` or ``"query"``), a ``doc_id``
    field, and an embedding vector stored under the *embedding* feature name
    (one level above the indexed feature).

    For chunked corpora the original ``doc_id`` is preserved on every chunk row
    (the chunker does ``new_row = row.copy()``), so Recall@K is measured at the
    document level: a query is a hit at K if *any* chunk whose ``doc_id`` matches
    a relevant document ID appears in the top-K FAISS results.

    Feature Naming Pattern::

        {in_feature}__indexed__evaluated

    Example::

        eval_docs__chunked__deduped__embedded__indexed__evaluated

    Returns a single aggregate row::

        [{"recall@1": float, "recall@5": float, "recall@10": float,
          "num_queries": int, "num_corpus": int}]
    """

    PREFIX_PATTERN = r".*__indexed__evaluated$"
    MIN_IN_FEATURES = 1
    MAX_IN_FEATURES = 1

    PROPERTY_MAPPING = {
        DefaultOptionKeys.in_features: {
            "explanation": "Source feature containing FAISS-indexed corpus + query rows",
            DefaultOptionKeys.context: True,
        },
    }

    @classmethod
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFramework]]]:
        return {PythonDictFramework}

    @classmethod
    def calculate_feature(cls, data: List[Dict[str, Any]], features: FeatureSet) -> List[Dict[str, Any]]:
        """Build FAISS index from corpus embeddings, search with query embeddings, compute Recall@K."""
        try:
            import faiss
            import numpy as np
        except ImportError as e:
            raise ImportError(
                "faiss-cpu and numpy are required for FaissRetrievalEvaluator. "
                "Install with: pip install faiss-cpu numpy"
            ) from e

        for feature in features.features:
            # source_feature = ...__indexed  (the FAISS indexing step output)
            source_feature = cls._extract_source_features(feature)[0]
            feature_name = feature.get_name()

            # embedding_feature = ...__embedded  (one level above __indexed)
            # The chunker/embedder store vectors under this key; the indexer adds
            # vector_id under source_feature but does NOT overwrite the embedding.
            embedding_feature = source_feature.rsplit("__indexed", 1)[0]

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

            # Build corpus embedding matrix
            corpus_embeddings = [r[embedding_feature] for r in corpus_rows]
            query_embeddings = [r[embedding_feature] for r in query_rows]

            corpus_matrix = np.array(corpus_embeddings, dtype=np.float32)  # N × D
            query_matrix = np.array(query_embeddings, dtype=np.float32)  # Q × D
            dimension = corpus_matrix.shape[1]

            # Build FAISS IndexFlatIP (inner product = cosine sim for unit-normalised vecs)
            index = faiss.IndexFlatIP(dimension)
            index.add(corpus_matrix)

            # Batch search: top-10 (or fewer if corpus is smaller) for all queries at once
            top_k = min(10, len(corpus_rows))
            _, faiss_indices = index.search(query_matrix, top_k)  # Q × top_k

            # doc_id for each corpus row (original doc_id preserved through chunking)
            corpus_doc_ids = [str(r.get("doc_id", "")) for r in corpus_rows]

            # Build ground-truth and ranked lists for mean_recall_at_k
            query_relevant: Dict[str, Set[str]] = {}
            query_ranked: Dict[str, List[str]] = {}

            for q_idx, q_row in enumerate(query_rows):
                q_id = str(q_row.get("doc_id", q_idx))
                relevant_ids = set(str(rid) for rid in q_row.get("relevant_doc_ids", []))
                query_relevant[q_id] = relevant_ids

                # Map FAISS result indices → doc_ids (dedup while preserving order)
                ranked_doc_ids: List[str] = []
                seen: Set[str] = set()
                for idx in faiss_indices[q_idx]:
                    if idx < 0:
                        continue
                    doc_id = corpus_doc_ids[idx]
                    if doc_id not in seen:
                        ranked_doc_ids.append(doc_id)
                        seen.add(doc_id)
                query_ranked[q_id] = ranked_doc_ids

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
            return [{feature_name: metrics, **metrics}]

        return []
