"""Retrieval evaluation demo.

Two pipeline modes:

``--pipeline direct`` (default):
    Calls embedders directly and uses brute-force numpy cosine similarity.
    Fast, no mloda wiring needed.

``--pipeline faiss``:
    Runs the *full* mloda ingestion chain end-to-end::

        eval_docs → __chunked → __deduped → __embedded → __indexed → __evaluated

    This exercises every stage of the production pipeline (chunking, dedup,
    embedding, FAISS indexing) and reports Recall@K through real FAISS search.

Usage::

    # SciFact — direct (brute-force cosine sim)
    python cli/eval_demo.py \\
        --dataset scifact \\
        --data-dir /Volumes/ExtraStorage/mlodadatasetevaluation/datasets/scifact \\
        --embedder sentence-transformer

    # SciFact — full pipeline through FAISS
    python cli/eval_demo.py \\
        --dataset scifact \\
        --data-dir /Volumes/ExtraStorage/mlodadatasetevaluation/datasets/scifact \\
        --embedder sentence-transformer \\
        --pipeline faiss

    # Flickr30K with CLIP
    python cli/eval_demo.py \\
        --dataset flickr30k \\
        --data-dir /Volumes/ExtraStorage/mlodadatasetevaluation/datasets/flickr30k_raw \\
        --embedder clip \\
        --max-samples 100
"""

from __future__ import annotations

import argparse
from typing import Dict, List, Set, Tuple

from mloda.user import Options

from rag_integration.feature_groups.datasets.text.scifact import ScifactDatasetSource
from rag_integration.feature_groups.datasets.image.flickr30k import Flickr30kDatasetSource
from rag_integration.feature_groups.evaluation.metrics import mean_recall_at_k


def _print_results(dataset: str, embedder: str, results: Dict[str, object]) -> None:
    print(f"\nDataset: {dataset}  |  Corpus: {results['num_corpus']}  |  Queries: {results['num_queries']}")
    print(f"Embedder: {embedder}")
    print("─" * 40)
    print(f"Recall@1:   {results['recall@1']:.4f}")
    print(f"Recall@5:   {results['recall@5']:.4f}")
    print(f"Recall@10:  {results['recall@10']:.4f}")
    print("─" * 40)


def _cosine_sim_rankings(
    corpus_rows: List[Dict],  # type: ignore[type-arg]
    query_rows: List[Dict],  # type: ignore[type-arg]
    embedding_key: str,
) -> Tuple[Dict[str, Set[str]], Dict[str, List[str]]]:
    """Compute cosine-similarity rankings. Returns (query_relevant, query_ranked)."""
    import numpy as np

    corpus_ids = [str(r.get("doc_id") or r.get("image_id")) for r in corpus_rows]
    query_ids = [str(r.get("doc_id") or r.get("image_id")) for r in query_rows]

    corpus_matrix = np.array([r[embedding_key] for r in corpus_rows], dtype=np.float32)
    query_matrix = np.array([r[embedding_key] for r in query_rows], dtype=np.float32)

    sims = query_matrix @ corpus_matrix.T  # Q × N

    query_relevant: Dict[str, Set[str]] = {}
    query_ranked: Dict[str, List[str]] = {}

    for q_idx, q_row in enumerate(query_rows):
        q_id = query_ids[q_idx]
        query_relevant[q_id] = set(q_row.get("relevant_doc_ids", []) + q_row.get("relevant_image_ids", []))
        ranked_indices = (-sims[q_idx]).argsort().tolist()
        query_ranked[q_id] = [corpus_ids[i] for i in ranked_indices]

    return query_relevant, query_ranked


def run_text_eval(data_dir: str, embedder_name: str) -> None:
    """Run SciFact text retrieval evaluation."""
    print("Loading SciFact dataset...")
    options = Options(context={ScifactDatasetSource.DATA_DIR: data_dir})
    rows = ScifactDatasetSource._load_dataset(options)

    corpus_rows = [r for r in rows if r["row_type"] == "corpus"]
    query_rows = [r for r in rows if r["row_type"] == "query"]
    print(f"  Corpus: {len(corpus_rows)} docs, Queries: {len(query_rows)}")

    print(f"Embedding with '{embedder_name}'...")
    embedding_key = "embedding"

    if embedder_name == "sentence-transformer":
        from rag_integration.feature_groups.rag_pipeline.embedding.sentence_transformer import (
            SentenceTransformerEmbedder,
        )

        texts = [r["text"] for r in corpus_rows + query_rows]
        embeddings = SentenceTransformerEmbedder._embed_texts(texts, embedding_dim=384, model_name="all-MiniLM-L6-v2")
    else:
        from rag_integration.feature_groups.rag_pipeline.embedding.mock import MockEmbedder

        texts = [r["text"] for r in corpus_rows + query_rows]
        embeddings = MockEmbedder._embed_texts(texts, embedding_dim=384, model_name="default")

    all_rows = corpus_rows + query_rows
    for row, emb in zip(all_rows, embeddings):
        row[embedding_key] = emb

    print("Computing Recall@K...")
    query_relevant, query_ranked = _cosine_sim_rankings(corpus_rows, query_rows, embedding_key)

    results: Dict[str, object] = {
        "recall@1": mean_recall_at_k(query_relevant, query_ranked, k=1),
        "recall@5": mean_recall_at_k(query_relevant, query_ranked, k=5),
        "recall@10": mean_recall_at_k(query_relevant, query_ranked, k=10),
        "num_corpus": len(corpus_rows),
        "num_queries": len(query_rows),
    }
    _print_results("BeIR/SciFact", embedder_name, results)


def run_image_eval(data_dir: str, embedder_name: str, max_samples: int) -> None:
    """Run Flickr30K image retrieval evaluation."""
    print(f"Loading Flickr30K dataset (max {max_samples} images)...")
    options = Options(
        context={
            Flickr30kDatasetSource.DATA_DIR: data_dir,
            Flickr30kDatasetSource.MAX_SAMPLES: max_samples,
        }
    )
    rows = Flickr30kDatasetSource._load_dataset(options)

    corpus_rows = [r for r in rows if r["row_type"] == "corpus"]
    query_rows = [r for r in rows if r["row_type"] == "query"]
    print(f"  Images: {len(corpus_rows)}, Captions (queries): {len(query_rows)}")

    embedding_key = "embedding"

    if embedder_name == "clip":
        from rag_integration.feature_groups.image_pipeline.embedding.clip import CLIPImageEmbedder

        model_path = CLIPImageEmbedder._resolve_model_path("openai/clip-vit-base-patch32")
        print(f"Embedding images with CLIP (vision encoder, model={model_path})...")
        for row in corpus_rows:
            image_data = row.get("image_data") or b""
            row[embedding_key] = CLIPImageEmbedder._embed_image(image_data, embedding_dim=512, model_name=model_path)

        print("Embedding captions with CLIP (text encoder)...")
        for row in query_rows:
            caption = row.get("caption") or ""
            row[embedding_key] = CLIPImageEmbedder._embed_text(caption, embedding_dim=512, model_name=model_path)
    else:
        from rag_integration.feature_groups.image_pipeline.embedding.mock import MockImageEmbedder

        print("Embedding images with mock embedder...")
        for row in corpus_rows:
            image_data = row.get("image_data") or b""
            row[embedding_key] = MockImageEmbedder._embed_image(image_data, embedding_dim=512, model_name="default")

        for row in query_rows:
            caption_bytes = (row.get("caption") or "").encode()
            row[embedding_key] = MockImageEmbedder._embed_image(caption_bytes, embedding_dim=512, model_name="default")

    print("Computing Recall@K...")
    query_relevant, query_ranked = _cosine_sim_rankings(corpus_rows, query_rows, embedding_key)

    results: Dict[str, object] = {
        "recall@1": mean_recall_at_k(query_relevant, query_ranked, k=1),
        "recall@5": mean_recall_at_k(query_relevant, query_ranked, k=5),
        "recall@10": mean_recall_at_k(query_relevant, query_ranked, k=10),
        "num_corpus": len(corpus_rows),
        "num_queries": len(query_rows),
    }
    _print_results(f"Flickr30K (n={max_samples})", embedder_name, results)


def run_faiss_eval(data_dir: str, embedder_name: str) -> None:
    """Run SciFact through the full mloda pipeline with FAISS indexing."""
    from mloda.user import mlodaAPI, PluginCollector, Feature, Options as MlodaOptions
    from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
        PythonDictFramework,
    )
    from rag_integration.feature_groups.datasets.text.scifact import ScifactDatasetSource
    from rag_integration.feature_groups.evaluation.faiss_retrieval_evaluator import FaissRetrievalEvaluator
    from rag_integration.feature_groups.rag_pipeline.chunking.fixed_size import FixedSizeChunker
    from rag_integration.feature_groups.rag_pipeline.deduplication.exact_hash import ExactHashDeduplicator
    from rag_integration.feature_groups.rag_pipeline.embedding.mock import MockEmbedder
    from rag_integration.feature_groups.rag_pipeline.embedding.sentence_transformer import SentenceTransformerEmbedder
    from rag_integration.feature_groups.rag_pipeline.vector_store.faiss_flat import FaissFlatIndexer

    embedder_cls = SentenceTransformerEmbedder if embedder_name == "sentence-transformer" else MockEmbedder
    embedding_method = "sentence_transformer" if embedder_name == "sentence-transformer" else "mock"
    model_name = "all-MiniLM-L6-v2" if embedder_name == "sentence-transformer" else "default"

    feature_name = "eval_docs__chunked__deduped__embedded__indexed__evaluated"
    print(f"Running full pipeline: {feature_name}")
    print(f"  Dataset: {data_dir}")
    print(f"  Embedder: {embedder_name}")

    feature = Feature(
        feature_name,
        options=MlodaOptions(
            {ScifactDatasetSource.DATA_DIR: data_dir},  # group → propagates down chain to eval_docs
            context={
                "chunking_method": "fixed_size",
                "deduplication_method": "exact_hash",
                "keep_strategy": "all_unique",
                "embedding_method": embedding_method,
                "model_name": model_name,
                "index_method": "flat",
            },
        ),
    )

    raw_result = mlodaAPI.run_all(
        features=[feature],
        compute_frameworks={PythonDictFramework},
        plugin_collector=PluginCollector.enabled_feature_groups(
            {
                ScifactDatasetSource,
                FixedSizeChunker,
                ExactHashDeduplicator,
                embedder_cls,
                FaissFlatIndexer,
                FaissRetrievalEvaluator,
            }
        ),
    )

    rows = raw_result[0] if raw_result else []
    if rows and isinstance(rows[0], list):
        rows = rows[0]

    row = rows[0] if rows else {}
    metrics = row.get(feature_name, row)

    results: Dict[str, object] = {
        "recall@1": metrics.get("recall@1", 0.0),
        "recall@5": metrics.get("recall@5", 0.0),
        "recall@10": metrics.get("recall@10", 0.0),
        "num_corpus": metrics.get("num_corpus", 0),
        "num_queries": metrics.get("num_queries", 0),
    }
    _print_results("BeIR/SciFact (full pipeline + FAISS)", embedder_name, results)


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG retrieval evaluation demo")
    parser.add_argument("--dataset", choices=["scifact", "flickr30k"], required=True)
    parser.add_argument("--data-dir", required=True, help="Local path to dataset folder")
    parser.add_argument("--embedder", default="mock", choices=["mock", "sentence-transformer", "clip"])
    parser.add_argument("--max-samples", type=int, default=100, help="Max images for Flickr30K (default: 100)")
    parser.add_argument(
        "--pipeline",
        default="direct",
        choices=["direct", "faiss"],
        help="'direct' = brute-force cosine sim (default), 'faiss' = full mloda chain with FAISS indexing",
    )
    args = parser.parse_args()

    if args.pipeline == "faiss":
        if args.dataset != "scifact":
            print("--pipeline faiss currently only supports --dataset scifact")
            return
        run_faiss_eval(args.data_dir, args.embedder)
    elif args.dataset == "scifact":
        run_text_eval(args.data_dir, args.embedder)
    else:
        run_image_eval(args.data_dir, args.embedder, args.max_samples)


if __name__ == "__main__":
    main()
