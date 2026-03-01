"""SciFact dataset source for text retrieval evaluation."""

from __future__ import annotations

from typing import Any, Dict, List

from mloda.user import Options

from rag_integration.feature_groups.datasets.text.base import BaseTextDatasetSource


class ScifactDatasetSource(BaseTextDatasetSource):
    """
    Loads the BeIR/SciFact dataset from a local directory for retrieval evaluation.

    SciFact (Wadden et al., 2020) is a scientific fact-checking dataset from the BEIR
    benchmark. It contains 5,183 corpus documents and 300 test queries with ground-truth
    relevance labels (qrels).

    Download the dataset first using the provided Jupyter notebook:
        /Volumes/ExtraStorage/mlodadatasetevaluation/download_datasets.ipynb

    Configuration:
        data_dir (str, required): Path to the local scifact folder containing
            corpus.jsonl, queries.jsonl, and qrels/test.tsv.
            E.g. "/Volumes/ExtraStorage/mlodadatasetevaluation/datasets/scifact"

    Output rows:
        Corpus rows:  {"doc_id": str, "text": str, "row_type": "corpus"}
        Query rows:   {"doc_id": str, "text": str, "row_type": "query",
                       "relevant_doc_ids": list[str],
                       "relevance_scores": dict[str, int]}

    Example::

        from mloda.user import Feature, Options
        feature = Feature("eval_docs", options=Options(context={
            "data_dir": "/Volumes/ExtraStorage/mlodadatasetevaluation/datasets/scifact"
        }))
    """

    DATA_DIR = "data_dir"

    @classmethod
    def _load_dataset(cls, options: Options) -> List[Dict[str, Any]]:
        """Load scifact corpus + queries + qrels using the beir GenericDataLoader."""
        data_dir = options.get(cls.DATA_DIR)
        if not data_dir:
            raise ValueError(
                f"'{cls.DATA_DIR}' option is required. Set it to the local path of the scifact dataset folder."
            )

        try:
            from beir.datasets.data_loader import GenericDataLoader
        except ImportError as e:
            raise ImportError(
                "The 'beir' package is required to load SciFact. Install it with: pip install beir"
            ) from e

        corpus, queries, qrels = GenericDataLoader(data_folder=str(data_dir)).load(split="test")

        rows: List[Dict[str, Any]] = []

        # Add corpus rows
        for doc_id, doc in corpus.items():
            text = f"{doc.get('title', '')} {doc.get('text', '')}".strip()
            rows.append({"doc_id": str(doc_id), "text": text, "row_type": "corpus"})

        # Add query rows with their relevant doc ids
        for query_id, query_text in queries.items():
            relevant = qrels.get(query_id, {})
            rows.append(
                {
                    "doc_id": str(query_id),
                    "text": query_text,
                    "row_type": "query",
                    "relevant_doc_ids": list(relevant.keys()),
                    "relevance_scores": {str(k): int(v) for k, v in relevant.items()},
                }
            )

        return rows
