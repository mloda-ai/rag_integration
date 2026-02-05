#!/usr/bin/env python3
"""
RAG Pipeline Demo CLI

Run with: python -m cli.rag_demo --help
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Type

from mloda.user import mlodaAPI, PluginCollector, Feature, Options
from mloda.provider import FeatureGroup
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
    PythonDictFramework,
)

from rag_integration.feature_groups.rag_pipeline import (
    DictDocumentSource,
    FixedSizeChunker,
    SentenceChunker,
    ParagraphChunker,
    SemanticChunker,
    MockEmbedder,
    HashEmbedder,
    TfidfEmbedder,
    SentenceTransformerEmbedder,
    ExactHashDeduplicator,
    NormalizedDeduplicator,
    NGramDeduplicator,
    RegexPIIRedactor,
    SimplePIIRedactor,
    PatternPIIRedactor,
    PresidioPIIRedactor,
)

# ANSI color codes
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
MAGENTA = "\033[35m"
BLUE = "\033[34m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"

# Component registry with descriptions
COMPONENTS = {
    "chunking": {
        "fixed_size": ("FixedSizeChunker", "Fixed character count chunks"),
        "sentence": ("SentenceChunker", "Sentence boundary splits"),
        "paragraph": ("ParagraphChunker", "Paragraph boundary splits"),
        "semantic": ("SemanticChunker", "Semantic similarity grouping (requires transformers)"),
    },
    "embedding": {
        "mock": ("MockEmbedder", "Deterministic random vectors (for testing)"),
        "hash": ("HashEmbedder", "Feature hashing"),
        "tfidf": ("TfidfEmbedder", "TF-IDF vectors"),
        "sentence_transformer": ("SentenceTransformerEmbedder", "Neural embeddings (requires transformers)"),
    },
    "dedup": {
        "exact_hash": ("ExactHashDeduplicator", "MD5 hash exact matching"),
        "normalized": ("NormalizedDeduplicator", "Whitespace-normalized matching"),
        "ngram": ("NGramDeduplicator", "N-gram Jaccard similarity"),
    },
    "pii": {
        "regex": ("RegexPIIRedactor", "Regex patterns (email, phone, SSN)"),
        "simple": ("SimplePIIRedactor", "Word-list based (common names)"),
        "pattern": ("PatternPIIRedactor", "User-configurable patterns"),
        "presidio": ("PresidioPIIRedactor", "Microsoft Presidio (requires presidio)"),
    },
}


# Mapping from CLI method names to provider classes
PROVIDER_CLASSES = {
    "chunking": {
        "fixed_size": FixedSizeChunker,
        "sentence": SentenceChunker,
        "paragraph": ParagraphChunker,
        "semantic": SemanticChunker,
    },
    "embedding": {
        "mock": MockEmbedder,
        "hash": HashEmbedder,
        "tfidf": TfidfEmbedder,
        "sentence_transformer": SentenceTransformerEmbedder,
    },
    "dedup": {
        "exact_hash": ExactHashDeduplicator,
        "normalized": NormalizedDeduplicator,
        "ngram": NGramDeduplicator,
    },
    "pii": {
        "regex": RegexPIIRedactor,
        "simple": SimplePIIRedactor,
        "pattern": PatternPIIRedactor,
        "presidio": PresidioPIIRedactor,
    },
}


def get_providers(args: argparse.Namespace) -> Set[Type[FeatureGroup]]:
    """Get the set of FeatureGroup providers needed for the pipeline based on CLI args."""
    providers: Set[Type[FeatureGroup]] = {
        DictDocumentSource,
        PROVIDER_CLASSES["chunking"][args.chunking],
        PROVIDER_CLASSES["dedup"][args.dedup],
        PROVIDER_CLASSES["embedding"][args.embedding],
    }
    if args.pii:
        providers.add(PROVIDER_CLASSES["pii"][args.pii])
    return providers


def build_pipeline_feature(docs: List[Dict[str, Any]], args: argparse.Namespace) -> Feature:
    """
    Build the mloda Feature chain for the RAG pipeline.

    Constructs nested features from inside out:
    docs -> [pii] -> chunked -> deduped -> embedded
    """
    # Start with document source
    current_feature: Feature = Feature(
        "docs",
        options=Options(context={"documents": docs}),
    )

    # PII redaction (optional)
    if args.pii:
        current_feature = Feature(
            "result_pii",
            options=Options(context={
                "redaction_method": args.pii,
                "in_features": current_feature,
            }),
        )

    # Chunking
    current_feature = Feature(
        "result_chunked",
        options=Options(context={
            "chunking_method": args.chunking,
            "chunk_size": args.chunk_size,
            "chunk_overlap": args.chunk_overlap,
            "in_features": current_feature,
        }),
    )

    # Deduplication
    current_feature = Feature(
        "result_deduped",
        options=Options(context={
            "deduplication_method": args.dedup,
            "in_features": current_feature,
        }),
    )

    # Embedding
    current_feature = Feature(
        "result_embedded",
        options=Options(context={
            "embedding_method": args.embedding,
            "embedding_dim": args.embedding_dim,
            "in_features": current_feature,
        }),
    )

    return current_feature


def print_box(title: str, lines: list[tuple[str, str, str]]) -> None:
    """Print a fancy box with pipeline results."""
    width = 44
    print(f"\n{CYAN}╭─ {BOLD}{title} {RESET}{CYAN}{'─' * (width - len(title) - 4)}╮{RESET}")
    for icon, label, value in lines:
        padding = width - len(label) - len(value) - 6
        print(f"{CYAN}│{RESET} {icon} {BOLD}{label}:{RESET}{' ' * padding}{value} {CYAN}│{RESET}")
    print(f"{CYAN}╰{'─' * width}╯{RESET}")


def cmd_list(args: argparse.Namespace) -> None:
    """Show available components."""
    print(f"\n{BOLD}Available RAG Pipeline Components{RESET}\n")

    categories = [
        ("CHUNKING METHODS", "chunking", "🔪"),
        ("EMBEDDING METHODS", "embedding", "📊"),
        ("DEDUPLICATION METHODS", "dedup", "🔍"),
        ("PII REDACTION METHODS", "pii", "🔒"),
    ]

    for title, category, icon in categories:
        print(f"{CYAN}{icon} {BOLD}{title}{RESET}")
        for method, (class_name, description) in COMPONENTS[category].items():
            print(f"  {GREEN}{method:20}{RESET} {DIM}{description}{RESET}")
        print()


def cmd_run(args: argparse.Namespace) -> None:
    """Run the RAG pipeline using mlodaAPI."""
    # Load documents
    docs: List[Dict[str, Any]] = []
    if args.docs:
        docs = json.loads(args.docs)
    elif args.input:
        input_path = Path(args.input)
        if input_path.is_file():
            with open(input_path) as f:
                content = f.read()
                if input_path.suffix == ".json":
                    docs = json.loads(content)
                else:
                    docs = [{"doc_id": input_path.name, "text": content}]
        elif input_path.is_dir():
            for file_path in input_path.glob("*"):
                if file_path.is_file() and file_path.suffix in (".txt", ".md", ".json"):
                    with open(file_path) as f:
                        content = f.read()
                    if file_path.suffix == ".json":
                        docs.extend(json.loads(content))
                    else:
                        docs.append({"doc_id": file_path.name, "text": content})

    if not docs:
        print(f"{YELLOW}No documents provided. Use --input or --docs{RESET}")
        sys.exit(1)

    doc_count = len(docs)

    if args.verbose:
        print(f"\n{BOLD}Running RAG Pipeline via mlodaAPI{RESET}")
        print(f"{DIM}{'─' * 40}{RESET}")

    # Build the feature chain and get providers
    final_feature = build_pipeline_feature(docs, args)
    providers = get_providers(args)

    if args.verbose:
        print(f"  {MAGENTA}▶{RESET} Building pipeline: {' -> '.join(p.__name__ for p in providers)}")

    # Run the pipeline via mlodaAPI
    raw_result = mlodaAPI.run_all(
        features=[final_feature],
        compute_frameworks={PythonDictFramework},
        plugin_collector=PluginCollector.enabled_feature_groups(providers),
    )

    # Extract results - mlodaAPI returns list of results per feature
    result_rows = raw_result[0] if raw_result else []
    if result_rows and isinstance(result_rows[0], list):
        result_rows = result_rows[0]

    # Extract embeddings and chunks from the result
    embeddings = []
    unique_chunks = []
    for i, row in enumerate(result_rows):
        embedding = row.get("result_embedded")
        if embedding is not None:
            embeddings.append(embedding)
            # Extract chunk text from the row if available
            chunk_text = row.get("result_deduped") or row.get("result_chunked") or ""
            unique_chunks.append({
                "chunk_idx": i,
                "text": chunk_text if isinstance(chunk_text, str) else str(chunk_text),
            })

    if args.verbose:
        print(f"  {MAGENTA}▶{RESET} Pipeline complete: {len(embeddings)} embeddings generated")

    # Show embedding preview
    if embeddings:
        first_emb = embeddings[0]
        preview_count = min(5, len(first_emb))
        preview = ", ".join(f"{v:.4f}" for v in first_emb[:preview_count])
        print(f"\n{BOLD}Embedding Preview (first vector, first {preview_count} dims):{RESET}")
        print(f"  [{preview}, ...]")

    # Print summary box
    lines = [
        ("📄", "Documents", f"{doc_count} loaded"),
    ]
    if args.pii:
        lines.append(("🔒", "PII", f"{args.pii} redaction applied"))
    lines.extend([
        ("✂️ ", "Chunking", f"{args.chunking} (size={args.chunk_size})"),
        ("🔍", "Dedup", f"{args.dedup}"),
        ("📊", "Embeddings", f"{len(embeddings)} vectors (dim={args.embedding_dim})"),
    ])
    print_box("RAG Pipeline", lines)

    # Prepare output
    result = {
        "documents": doc_count,
        "chunks": unique_chunks,
        "embeddings": embeddings,
        "config": {
            "chunking": args.chunking,
            "embedding": args.embedding,
            "dedup": args.dedup,
            "pii": args.pii,
            "chunk_size": args.chunk_size,
            "chunk_overlap": args.chunk_overlap,
            "embedding_dim": args.embedding_dim,
        },
    }

    # Output
    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n{GREEN}Output written to {args.output}{RESET}")

    if args.verbose:
        print(f"\n{DIM}Pipeline complete!{RESET}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="RAG Pipeline Demo CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m cli.rag_demo list
  python -m cli.rag_demo run --docs '[{"doc_id":"1","text":"Hello John at john@test.com"}]'
  python -m cli.rag_demo run --input ./docs/ --chunking sentence --embedding tfidf -v
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # List command
    subparsers.add_parser("list", help="Show available components")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run the pipeline")

    # Input options
    input_group = run_parser.add_argument_group("Input")
    input_group.add_argument("--input", "-i", help="Input file or directory")
    input_group.add_argument("--docs", help="Inline JSON documents")

    # Method options
    method_group = run_parser.add_argument_group("Methods")
    method_group.add_argument("--chunking", default="fixed_size",
                              choices=list(COMPONENTS["chunking"].keys()),
                              help="Chunking method (default: fixed_size)")
    method_group.add_argument("--embedding", default="mock",
                              choices=list(COMPONENTS["embedding"].keys()),
                              help="Embedding method (default: mock)")
    method_group.add_argument("--dedup", default="exact_hash",
                              choices=list(COMPONENTS["dedup"].keys()),
                              help="Deduplication method (default: exact_hash)")
    method_group.add_argument("--pii", default=None,
                              choices=list(COMPONENTS["pii"].keys()),
                              help="PII redaction method (optional)")

    # Parameter options
    param_group = run_parser.add_argument_group("Parameters")
    param_group.add_argument("--chunk-size", type=int, default=512,
                             help="Chunk size in characters (default: 512)")
    param_group.add_argument("--chunk-overlap", type=int, default=50,
                             help="Overlap between chunks (default: 50)")
    param_group.add_argument("--embedding-dim", type=int, default=384,
                             help="Embedding dimension (default: 384)")

    # Output options
    output_group = run_parser.add_argument_group("Output")
    output_group.add_argument("--output", "-o", help="Output file (JSON)")
    output_group.add_argument("--verbose", "-v", action="store_true",
                              help="Verbose output")

    args = parser.parse_args()

    if args.command == "list":
        cmd_list(args)
    elif args.command == "run":
        cmd_run(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
