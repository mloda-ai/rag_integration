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
| `dev`      | tox, pytest, ruff, mypy, bandit, mloda-testing    |

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

# Group options forward to every upstream stage, down to `docs`; context options do not.
feature = Feature(
    "docs__pii_redacted__chunked__deduped__embedded",
    options=Options(group={"documents": documents}),
)

results = mlodaAPI.run_all(
    features=[feature],
    compute_frameworks=[PythonDictFramework],
    plugin_collector=PluginCollector.enabled_feature_groups(providers),
)
```

### 4. Configure pipeline stages

With several implementations per stage enabled, group options select one each and tune parameters. To pick another value, add its class to `providers`:

```python
from rag_integration.feature_groups.rag_pipeline import (
    HashEmbedder,
    NormalizedDeduplicator,
    SentenceChunker,
    SimplePIIRedactor,
)

providers |= {SimplePIIRedactor, SentenceChunker, NormalizedDeduplicator, HashEmbedder}

feature = Feature(
    "docs__pii_redacted__chunked__deduped__embedded",
    options=Options(
        group={
            "documents": documents,
            "redaction_method": "regex",  # or "simple", "pattern", "presidio"
            "chunking_method": "sentence",  # or "fixed_size", "paragraph", "semantic"
            "deduplication_method": "exact_hash",  # or "normalized", "ngram"
            "embedding_method": "hash",  # or "mock", "tfidf", "sentence_transformer"
            "chunk_size": 512,
            "chunk_overlap": 128,
        }
    ),
)

results = mlodaAPI.run_all(
    features=[feature],
    compute_frameworks=[PythonDictFramework],
    plugin_collector=PluginCollector.enabled_feature_groups(providers),
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
