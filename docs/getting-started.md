# Getting Started

## Installation

Install with pip:

```bash
pip install rag-integration
```

For advanced features (PII redaction with Presidio, sentence-transformer embeddings, FAISS indexing):

```bash
pip install rag-integration[advanced]
```

Other extras:

| Extra      | What it adds                                      |
|------------|---------------------------------------------------|
| `faiss`    | FAISS vector indexing (`faiss-cpu`)                |
| `advanced` | Presidio, sentence-transformers, joblib, Pillow, FAISS |
| `eval`     | BEIR benchmark datasets, pandas, numpy            |
| `dev`      | tox, pytest, ruff, mypy, bandit                   |

## Quick Start

rag-integration is a set of mloda FeatureGroups that compose into a RAG pipeline. Each stage is a feature that chains onto the previous one using the `__` naming convention.

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
```

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

| Stage         | Implementations                                          |
|---------------|----------------------------------------------------------|
| Document Source | `DictDocumentSource`, `FileDocumentSource`             |
| PII Redaction | `RegexPIIRedactor`, `SimplePIIRedactor`, `PatternPIIRedactor`, `PresidioPIIRedactor` |
| Chunking      | `FixedSizeChunker`, `SentenceChunker`, `ParagraphChunker`, `SemanticChunker` |
| Deduplication | `ExactHashDeduplicator`, `NormalizedDeduplicator`, `NGramDeduplicator` |
| Embedding     | `MockEmbedder`, `HashEmbedder`, `TfidfEmbedder`, `SentenceTransformerEmbedder` |
| Vector Store  | `FaissFlatIndexer`, `FaissIVFIndexer`, `FaissHNSWIndexer` |
| Retrieval     | `FaissRetriever`                                         |
| LLM Response  | `ClaudeCliResponse`                                      |

## Development Setup

```bash
# Clone and set up
git clone <repo-url>
cd rag_integration

# Create virtual environment and install all deps
uv venv
source .venv/bin/activate
uv sync --all-extras

# Run all checks (pytest, ruff, mypy, bandit)
tox
```
