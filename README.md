[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![mloda](https://img.shields.io/badge/built%20with-mloda-blue.svg)](https://github.com/mloda-ai/mloda)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

# rag_integration

> End-to-end RAG as composable mloda plugins: PII redaction, chunking, deduplication, embedding, vector indexing, retrieval, and LLM response with pluggable backends.

## What is this?

`rag_integration` is a family of [mloda](https://github.com/mloda-ai/mloda) plugins that implement every stage of a RAG pipeline. Each stage (document loading, PII redaction, chunking, deduplication, embedding, vector indexing, retrieval, LLM response) is an independent, swappable `FeatureGroup`. mloda resolves the dependency graph automatically: declare the feature you want and the framework assembles the pipeline.

The MVP covers the full end-to-end pipeline: text and image ingestion, FAISS vector indexing with artifact persistence, semantic retrieval, and LLM response generation via Claude CLI.

## Why mloda for RAG?

### 1. Swap any stage with one line

Change `FixedSizeChunker` to `SentenceChunker`, or `FaissFlatIndexer` to `FaissHNSWIndexer`, by swapping one class in the providers set. The feature name (`docs__pii_redacted__chunked__deduped__embedded__indexed`) is identical across all configurations.

```python
from rag_integration.feature_groups.rag_pipeline import (
    RegexPIIRedactor, PresidioPIIRedactor,
    FixedSizeChunker, SentenceChunker,
    ExactHashDeduplicator, NormalizedDeduplicator,
    MockEmbedder, SentenceTransformerEmbedder,
    FaissFlatIndexer, FaissHNSWIndexer,
)

# Config A: regex PII, fixed-size chunks, flat FAISS index
providers_a = {MyDocSource, RegexPIIRedactor, FixedSizeChunker, ExactHashDeduplicator, MockEmbedder, FaissFlatIndexer}

# Config B: neural PII, sentence chunks, HNSW index — swap one class per stage
providers_b = {MyDocSource, PresidioPIIRedactor, SentenceChunker, NormalizedDeduplicator, SentenceTransformerEmbedder, FaissHNSWIndexer}

result = mlodaAPI.run_all(
    features=["docs__pii_redacted__chunked__deduped__embedded__indexed"],
    compute_frameworks={PythonDictFramework},
    plugin_collector=PluginCollector.enabled_feature_groups(providers_a),  # swap to providers_b here
)
```

### 2. Compare multiple configs in a single API call

Run 4 different pipeline configurations in one `mlodaAPI.run_all()` call using mloda's domain system. The framework routes each domain to its own provider set and executes everything in one pass.

```python
from mloda.user import Feature

features = [
    Feature("docs__pii_redacted__chunked__deduped__embedded", domain="regex_fixed_tfidf"),
    Feature("docs__pii_redacted__chunked__deduped__embedded", domain="regex_sentence_hash"),
    Feature("docs__pii_redacted__chunked__deduped__embedded", domain="presidio_semantic_st"),
    Feature("docs__pii_redacted__chunked__deduped__embedded", domain="regex_paragraph_ngram"),
]

raw_results = mlodaAPI.run_all(
    features=features,
    compute_frameworks={PythonDictFramework},
    plugin_collector=PluginCollector.enabled_feature_groups(all_providers),
)
# raw_results[0] = TF-IDF config, raw_results[1] = hash config, etc.
```

See `TestAlternativeProviders.test_all_provider_combinations` in `tests/test_rag_pipeline_integration.py` for the working 4-config comparison.

### 3. Request any pipeline stage independently

mloda resolves the dependency graph automatically. Request `docs__pii_redacted` alone to inspect redacted text without running the rest of the pipeline. Each stage is also independently unit-tested.

```python
# PII redaction only (no chunking, no embedding)
mlodaAPI.run_all(features=["docs__pii_redacted"], ...)

# Up to deduplication
mlodaAPI.run_all(features=["docs__pii_redacted__chunked__deduped"], ...)

# Full ingestion pipeline including vector indexing
mlodaAPI.run_all(features=["docs__pii_redacted__chunked__deduped__embedded__indexed"], ...)
```

PII redaction is a first-class stage, not an afterthought: 4 text implementations (regex, Presidio NLP, pattern-based, word-list) plus image redaction (blur, pixelate, solid fill). Most RAG tutorials skip PII entirely.

## Pipeline Overview

| Stage | Implementations | Feature suffix |
|-------|----------------|----------------|
| Document Source | `DictDocumentSource`, `FileDocumentSource` | `docs` |
| PII Redaction | `RegexPIIRedactor`, `PresidioPIIRedactor`, `PatternPIIRedactor`, `SimplePIIRedactor` | `__pii_redacted` |
| Chunking | `FixedSizeChunker`, `SentenceChunker`, `ParagraphChunker`, `SemanticChunker` | `__chunked` |
| Deduplication | `ExactHashDeduplicator`, `NormalizedDeduplicator`, `NGramDeduplicator` | `__deduped` |
| Embedding | `MockEmbedder`, `HashEmbedder`, `TfidfEmbedder`, `SentenceTransformerEmbedder` | `__embedded` |
| Vector Store | `FaissFlatIndexer`, `FaissIVFIndexer`, `FaissHNSWIndexer` | `__indexed` |
| Retrieval | `FaissRetriever` | `retrieved` (separate query feature) |
| LLM Response | `ClaudeCliResponse` | `llm_response` (separate query feature) |

An image pipeline mirrors the ingestion structure with image-specific providers (CLIP embeddings, perceptual hash dedup, blur/pixelate PII redaction).

## Quick Start

### Install

```bash
uv venv
source .venv/bin/activate
uv sync --all-extras
```

### Run the example

```bash
python examples/quickstart.py
```

[`examples/quickstart.py`](examples/quickstart.py) is a self-contained script that runs the full ingestion pipeline (PII redact, chunk, dedup, embed, FAISS index) and then queries the index with `FaissRetriever`. No external services required.

### Minimal code example

```python
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Set, Type

from mloda.user import mlodaAPI, PluginCollector, Feature, Options
from mloda.provider import DataCreator, FeatureGroup
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
    PythonDictFramework,
)
from rag_integration.feature_groups.rag_pipeline import (
    RegexPIIRedactor, FixedSizeChunker, ExactHashDeduplicator, MockEmbedder,
    FaissFlatIndexer, FaissRetriever,
)

DOCUMENTS = [
    {"doc_id": "doc_1", "text": "Contact john@example.com or call 555-123-4567."},
    {"doc_id": "doc_2", "text": "Meeting with jane@test.org at 800-555-0199."},
]

class DocumentSource(FeatureGroup):
    @classmethod
    def input_data(cls) -> DataCreator:
        return DataCreator({"docs"})

    @classmethod
    def match_feature_group_criteria(cls, feature_name: Any, options: Any, data_access_collection: Any = None) -> bool:
        return str(feature_name) == "docs"

    @classmethod
    def compute_framework_rule(cls) -> Set[Type[Any]]:
        return {PythonDictFramework}

    @classmethod
    def calculate_feature(cls, data: Any, features: Any) -> List[Dict[str, Any]]:
        return [{"docs": doc["text"], "doc_id": doc["doc_id"]} for doc in DOCUMENTS]


with tempfile.TemporaryDirectory() as tmp_dir:
    # Phase 1: ingest and build FAISS index
    ingestion_providers = {DocumentSource, RegexPIIRedactor, FixedSizeChunker, ExactHashDeduplicator, MockEmbedder, FaissFlatIndexer}
    mlodaAPI.run_all(
        features=[Feature("docs__pii_redacted__chunked__deduped__embedded__indexed", options=Options({"artifact_storage_path": tmp_dir}))],
        compute_frameworks={PythonDictFramework},
        plugin_collector=PluginCollector.enabled_feature_groups(ingestion_providers),
    )

    # Phase 2: discover artifacts and query
    index_path = str(next(Path(tmp_dir).glob("vector_store_*.faiss")))
    metadata_path = str(next(Path(tmp_dir).glob("vector_store_*_metadata.json")))

    raw = mlodaAPI.run_all(
        features=[Feature("retrieved", options=Options({"index_path": index_path, "metadata_path": metadata_path, "query_text": "email contact", "embedding_method": "mock", "top_k": 2}))],
        compute_frameworks={PythonDictFramework},
        plugin_collector=PluginCollector.enabled_feature_groups({FaissRetriever}),
    )
    result = raw[0] if raw and isinstance(raw[0], list) else raw
    print(result[0]["retrieved"])
```

## What's in the MVP

**In scope (done):**

- Text pipeline: 8 stages (source, PII, chunk, dedup, embed, vector store, retrieval, LLM), multiple provider implementations each, all independently unit-tested
- Image pipeline: image loading, PII redaction (blur/pixelate/solid fill), preprocessing (resize, normalize, thumbnail), deduplication (exact hash, pHash, dHash), CLIP embedding
- Vector store: `FaissFlatIndexer`, `FaissIVFIndexer`, `FaissHNSWIndexer` with FAISS artifact persistence (index + metadata sidecar)
- Retrieval: `FaissRetriever` with configurable `top_k`, returns indices, distances, texts, and doc_ids
- LLM response: `ClaudeCliResponse` via `claude -p` for zero-dependency local inference
- Artifact persistence: embeddings and FAISS indexes persist to disk, so repeated pipeline runs skip recomputation
- CLI demo: interactive pipeline explorer with configurable providers (`cli/rag_demo.py`)

**What's next:**

- Additional vector store backends (Chroma, Qdrant, pgvector)
- Cloud LLM providers (OpenAI, Bedrock) alongside Claude CLI
- Evaluation and benchmarking stage
- Streaming retrieval for large corpora

## CLI Demo

Interactive pipeline explorer: pick your chunker, embedder, PII redactor, and input documents from a menu.

```bash
python -m cli.rag_demo
```

Or with arguments directly:

```bash
python -m cli.rag_demo --chunking sentence --embedding tfidf --pii regex
```

## Development

```bash
uv venv
source .venv/bin/activate
uv sync --all-extras
tox
```

Individual checks:

```bash
pytest
ruff format --check --line-length 120 .
ruff check .
mypy --strict --ignore-missing-imports .
bandit -c pyproject.toml -r -q .
```

## Related

- [mloda](https://github.com/mloda-ai/mloda): the core library
- [mloda-registry](https://github.com/mloda-ai/mloda-registry): community plugins and 40+ development guides
- [mloda.ai](https://mloda.ai): overview and business context
