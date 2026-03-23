[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/mloda-ai/rag_integration/blob/main/LICENSE)
[![mloda](https://img.shields.io/badge/built%20with-mloda-blue.svg)](https://github.com/mloda-ai/mloda)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

# rag-integration

RAG integration plugin for [mloda](https://github.com/mloda-ai/mloda). Composes modular FeatureGroups into text and image processing pipelines with PII redaction, chunking, deduplication, embedding, FAISS vector search, retrieval, evaluation, and LLM response generation.

See the [demo notebook](demo.ipynb) for an interactive walkthrough or the [CLI](cli/README.md) for command-line usage.

## Project Structure

```
rag_integration/
  feature_groups/
    rag_pipeline/       # Text: source, PII, chunk, dedup, embed, index, retrieve, LLM
    image_pipeline/     # Image: source, PII, preprocess, dedup, embed
    datasets/           # BEIR (SciFact) and image (Flickr30k) loaders
    evaluation/         # Retrieval metrics (precision, recall, NDCG, MAP)
cli/                    # Command-line demo tools (see cli/README.md)
tests/                  # Unit and integration tests
docs/                   # Detailed guides
```

## Quick Start

### 1. Define your documents

Documents are loaded through a `DataCreator`-based FeatureGroup. The simplest option is `DictDocumentSource`, which loads from a Python list:

```python
documents = [
    {"doc_id": "1", "text": "Contact support@example.com for help."},
    {"doc_id": "2", "text": "Our office is at 123 Main St."},
]
```

### 2. Build the pipeline

Each pipeline stage is expressed as a feature name. Stages chain with `__`:

```
docs                                    # raw documents
docs__pii_redacted                      # PII removed
docs__pii_redacted__chunked             # text split into chunks
docs__pii_redacted__chunked__deduped    # duplicates removed
docs__pii_redacted__chunked__deduped__embedded  # vector embeddings
```

### 3. Run with mlodaAPI

```python
from mloda.user import mlodaAPI, PluginCollector, Feature, Options
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
    PythonDictFramework,
)

from rag_integration.feature_groups.rag_pipeline import (
    DictDocumentSource,
    RegexPIIRedactor,
    FixedSizeChunker,
    ExactHashDeduplicator,
    MockEmbedder,
)

providers = {
    DictDocumentSource,
    RegexPIIRedactor,
    FixedSizeChunker,
    ExactHashDeduplicator,
    MockEmbedder,
}

results = mlodaAPI.run_all(
    features=["docs__pii_redacted__chunked__deduped__embedded"],
    compute_frameworks={PythonDictFramework},
    plugin_collector=PluginCollector.enabled_feature_groups(providers),
)
```

### 4. Configure pipeline stages

Use `Options` to select specific implementations and tune parameters:

```python
feature = Feature(
    "docs__pii_redacted__chunked__deduped__embedded",
    options=Options(context={
        "redaction_method": "regex",        # or "simple", "pattern", "presidio"
        "chunking_method": "sentence",      # or "fixed_size", "paragraph", "semantic"
        "deduplication_method": "exact_hash",  # or "normalized", "ngram"
        "embedding_method": "sentence-transformer",  # or "hash", "tfidf", "mock"
        "chunk_size": 512,
        "chunk_overlap": 50,
    }),
)
```

## Available Components

### Text Pipeline

| Stage           | Implementations                                                                           |
|-----------------|-------------------------------------------------------------------------------------------|
| Document Source | `DictDocumentSource`, `FileDocumentSource`                                                |
| PII Redaction   | `RegexPIIRedactor`, `SimplePIIRedactor`, `PatternPIIRedactor`, `PresidioPIIRedactor`      |
| Chunking        | `FixedSizeChunker`, `SentenceChunker`, `ParagraphChunker`, `SemanticChunker`              |
| Deduplication   | `ExactHashDeduplicator`, `NormalizedDeduplicator`, `NGramDeduplicator`                    |
| Embedding       | `MockEmbedder`, `HashEmbedder`, `TfidfEmbedder`, `SentenceTransformerEmbedder`            |
| Vector Store    | `FaissFlatIndexer`, `FaissIVFIndexer`, `FaissHNSWIndexer`                                 |
| Retrieval       | `FaissRetriever`                                                                          |
| LLM Response    | `ClaudeCliResponse`                                                                       |

### Image Pipeline

| Stage           | Implementations                                                                                       |
|-----------------|-------------------------------------------------------------------------------------------------------|
| Image Source    | `DictImageSource`, `FileImageSource`                                                                  |
| PII Redaction   | `BlurPIIRedactor`, `PixelPIIRedactor`, `SolidFillPIIRedactor`                                        |
| Preprocessing   | `ResizePreprocessor`, `NormalizePreprocessor`, `ThumbnailPreprocessor`                                |
| Deduplication   | `ExactHashImageDeduplicator`, `PerceptualHashImageDeduplicator`, `DifferenceHashImageDeduplicator`   |
| Embedding       | `MockImageEmbedder`, `HashImageEmbedder`, `CLIPImageEmbedder`                                        |

## Installation

Clone the repository and install with uv:

```bash
git clone https://github.com/mloda-ai/rag_integration.git
cd rag_integration
uv venv
source .venv/bin/activate
uv sync --all-extras
```

To install only specific extras, use `uv sync --extra <name>`:

| Extra      | What it adds                                         |
|------------|------------------------------------------------------|
| `faiss`    | FAISS vector indexing (`faiss-cpu`)                   |
| `advanced` | Presidio, sentence-transformers, joblib, Pillow, FAISS|
| `eval`     | BEIR benchmark datasets, pandas, numpy               |
| `dev`      | tox, pytest, ruff, mypy, bandit                      |

## CLI

A command-line interface is available for running pipelines interactively. See [cli/README.md](cli/README.md) for full usage.

```bash
python3 -m cli.rag_demo run --input cli/docs/ --pii regex --chunking sentence --embedding tfidf -v
```

## Development Setup

```bash
uv venv
source .venv/bin/activate
uv sync --all-extras
```

Run all checks (pytest, ruff, mypy, bandit):

```bash
tox
```

### Run individual checks

```bash
pytest
ruff format --check --line-length 120 .
ruff check .
mypy --strict --ignore-missing-imports .
bandit -c pyproject.toml -r -q .
```

## Related

- [Getting Started Guide](docs/getting-started.md) for a detailed walkthrough
- [GitHub Workflows](docs/github-workflows.md) for CI/CD setup and required secrets
- [mloda](https://github.com/mloda-ai/mloda) core library
- [mloda-registry](https://github.com/mloda-ai/mloda-registry) for plugin guides and community plugins
