# RAG Pipeline Design Document

## Overview

This document describes the design of the `docs__pii_redacted__chunked__deduped__embedded` RAG (Retrieval-Augmented Generation) pipeline built on mloda's plugin architecture.

The pipeline demonstrates the **Provider Inheritance Pattern** where each processing step has a base class with 3 alternative provider implementations, allowing users to swap implementations without changing the pipeline definition.

## Architecture

### Pattern: Provider Inheritance (Proposal 9)

Each pipeline step follows this structure:

```
BaseStep (FeatureChainParserMixin + FeatureGroup)
├── PREFIX_PATTERN = r".*__step_suffix$"
├── calculate_feature() - shared orchestration
└── _abstract_method() - provider-specific logic
        │
        ├── ProviderA_Step
        ├── ProviderB_Step
        └── ProviderC_Step
```

### Compute Framework

All implementations use **PythonDictFramework** (from mloda):
- Data structure: `List[Dict[str, Any]]`
- Zero external dependencies
- Each row is a dictionary with feature values

## Pipeline Stages

```
docs → pii_redacted → chunked → deduped → embedded
```

### Stage 1: Document Source (Root Feature)

**Feature name:** `docs`

Loads documents into the pipeline. This is a root feature with no inputs.

| Provider | Description | Use Case |
|----------|-------------|----------|
| FileDocumentSource | Loads from JSON/CSV files | File-based ingestion |
| DictDocumentSource | Accepts documents via Options | API/programmatic input |

For testing, use the mloda `DataCreator` pattern with a test-only FeatureGroup (see integration tests for examples).

**Output schema:**
```python
[
    {"doc_id": "1", "text": "Document content...", "metadata": {...}},
    {"doc_id": "2", "text": "Another document...", "metadata": {...}},
]
```

### Stage 2: PII Redaction

**Feature pattern:** `{input}__pii_redacted`

Detects and redacts Personally Identifiable Information from text.

| Provider | Description | Use Case |
|----------|-------------|----------|
| RegexPIIRedactor | Regex patterns for email, phone, SSN | Fast, no dependencies |
| SimplePIIRedactor | Word-list based (common names) | Simple name redaction |
| PatternPIIRedactor | User-configurable patterns | Custom PII types |

**Configuration options:**
- `pii_types`: List of PII types to detect (EMAIL, PHONE, SSN, NAME, ALL)
- `replacement_strategy`: How to replace detected PII (mask, hash, type_label)

### Stage 3: Chunking

**Feature pattern:** `{input}__chunked`

Splits documents into smaller chunks for embedding and retrieval.

| Provider | Description | Use Case |
|----------|-------------|----------|
| FixedSizeChunker | Fixed character count with overlap | Uniform chunks, fastest |
| SentenceChunker | Splits on sentence boundaries | Readable chunks |
| ParagraphChunker | Splits on paragraph boundaries | Document-aware chunks |

**Configuration options:**
- `chunk_size`: Maximum chunk size (characters)
- `chunk_overlap`: Overlap between chunks

**Note:** Chunking transforms 1 document into N chunks, expanding the row count.

### Stage 4: Deduplication

**Feature pattern:** `{input}__deduped`

Removes duplicate or near-duplicate chunks.

| Provider | Description | Use Case |
|----------|-------------|----------|
| ExactHashDeduplicator | MD5 hash comparison | Exact duplicates only |
| NormalizedDeduplicator | Whitespace-normalized comparison | Near-exact duplicates |
| NGramDeduplicator | N-gram Jaccard similarity | Fuzzy duplicates |

**Configuration options:**
- `similarity_threshold`: Threshold for considering texts duplicates (0.0-1.0)
- `keep_strategy`: Which duplicate to keep (first, longest, all_unique)

### Stage 5: Embedding

**Feature pattern:** `{input}__embedded`

Converts text chunks into dense vector representations.

| Provider | Description | Use Case |
|----------|-------------|----------|
| MockEmbedder | Random vectors | Testing, development |
| HashEmbedder | Deterministic hash-based vectors | Reproducible, fast |
| TfidfEmbedder | TF-IDF vectors (dense) | Simple semantic |

**Configuration options:**
- `embedding_dim`: Dimension of output vectors
- `model_name`: Model identifier (provider-specific)

## Data Flow

```
Input Document:
┌─────────────────────────────────────────────────────┐
│ {"doc_id": "1", "text": "Contact john@example.com"} │
└─────────────────────────────────────────────────────┘
                          │
                          ▼ PII Redaction
┌─────────────────────────────────────────────────────┐
│ {"doc_id": "1",                                     │
│  "text": "Contact john@example.com",                │
│  "docs__pii_redacted": "Contact [EMAIL]"}           │
└─────────────────────────────────────────────────────┘
                          │
                          ▼ Chunking (1 → N rows)
┌─────────────────────────────────────────────────────┐
│ {"doc_id": "1", "chunk_id": "1_0", "chunk_index": 0,│
│  "docs__pii_redacted__chunked": "Contact [EMAIL]"}  │
└─────────────────────────────────────────────────────┘
                          │
                          ▼ Deduplication
┌─────────────────────────────────────────────────────┐
│ {"...", "is_duplicate": false,                      │
│  "docs__pii_redacted__chunked__deduped": "..."}     │
└─────────────────────────────────────────────────────┘
                          │
                          ▼ Embedding
┌─────────────────────────────────────────────────────┐
│ {"...", "embedding_dim": 384,                       │
│  "docs__pii_redacted__chunked__deduped__embedded":  │
│   [0.1, -0.2, 0.3, ...]}                            │
└─────────────────────────────────────────────────────┘
```

## Usage Examples

### Basic Usage

```python
from mloda.user import mloda
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
    PythonDictFramework,
)

# Run the full pipeline
result = mloda.run_all(
    features=["docs__pii_redacted__chunked__deduped__embedded"],
    compute_frameworks=[PythonDictFramework],
)

print(f"Generated {len(result)} embedded chunks")
```

### With Configuration

```python
from mloda.user import mloda, Feature, Options

result = mloda.run_all(
    features=[
        Feature(
            name="docs__pii_redacted__chunked__deduped__embedded",
            options=Options(
                context={
                    # PII Redaction options
                    "pii_types": ["EMAIL", "PHONE"],
                    "replacement_strategy": "type_label",

                    # Chunking options
                    "chunk_size": 256,
                    "chunk_overlap": 25,

                    # Deduplication options
                    "similarity_threshold": 0.95,
                    "keep_strategy": "first",

                    # Embedding options
                    "embedding_dim": 384,
                }
            )
        )
    ],
    compute_frameworks=[PythonDictFramework],
)
```

### Partial Pipeline

```python
# Only run PII redaction and chunking
result = mloda.run_all(
    features=["docs__pii_redacted__chunked"],
    compute_frameworks=[PythonDictFramework],
)
```

### With Custom Documents

```python
from mloda.user import mloda, Feature, Options

# Pass documents via Options (using DictDocumentSource)
result = mloda.run_all(
    features=[
        Feature(
            name="docs__pii_redacted",
            options=Options(
                context={
                    "documents": [
                        {"doc_id": "custom1", "text": "My email is user@test.com"},
                        {"doc_id": "custom2", "text": "Call me at 555-1234"},
                    ]
                }
            )
        )
    ],
    compute_frameworks=[PythonDictFramework],
)
```

## Provider Selection

mloda automatically selects the appropriate provider based on:

1. **Compute Framework**: All providers in this pipeline use `PythonDictFramework`
2. **Feature Matching**: Providers match on `PREFIX_PATTERN` (e.g., `*__pii_redacted`)
3. **First Match**: When multiple providers match, the first registered one is used

To use a specific provider, you can:
- Import only the desired provider (recommended)
- Use domain disambiguation (advanced)

## File Structure

```
rag_integration/feature_groups/rag_pipeline/
├── __init__.py
├── document_source/
│   ├── __init__.py
│   ├── base.py              # BaseDocumentSource
│   ├── file_loader.py       # FileDocumentSource
│   └── dict_loader.py       # DictDocumentSource
├── pii_redaction/
│   ├── __init__.py
│   ├── base.py              # BasePIIRedactor
│   ├── regex.py             # RegexPIIRedactor
│   ├── simple.py            # SimplePIIRedactor
│   └── pattern.py           # PatternPIIRedactor
├── chunking/
│   ├── __init__.py
│   ├── base.py              # BaseChunker
│   ├── fixed_size.py        # FixedSizeChunker
│   ├── sentence.py          # SentenceChunker
│   └── paragraph.py         # ParagraphChunker
├── deduplication/
│   ├── __init__.py
│   ├── base.py              # BaseDeduplicator
│   ├── exact_hash.py        # ExactHashDeduplicator
│   ├── normalized.py        # NormalizedDeduplicator
│   └── ngram.py             # NGramDeduplicator
├── embedding/
│   ├── __init__.py
│   ├── base.py              # BaseEmbedder
│   ├── mock.py              # MockEmbedder
│   ├── hash_embed.py        # HashEmbedder
│   └── tfidf.py             # TfidfEmbedder
└── tests/
    └── ...
```

## Extension Points

### Adding a New Provider

1. Create a new file in the appropriate subdirectory
2. Inherit from the base class
3. Implement the abstract method(s)
4. The provider is automatically registered via mloda's plugin system

Example:
```python
from rag_integration.feature_groups.rag_pipeline.pii_redaction.base import BasePIIRedactor

class MyCustomPIIRedactor(BasePIIRedactor):
    @classmethod
    def _redact_pii(cls, texts, pii_types, replacement_strategy):
        # Custom implementation
        return [custom_redact(t) for t in texts]
```

### Adding a New Pipeline Stage

1. Create a new subdirectory with `base.py` and provider files
2. Define `PREFIX_PATTERN` for the new stage
3. Implement the base class with abstract methods
4. Create provider implementations

## Dependencies

- **mloda**: Core framework (FeatureGroup, FeatureChainParserMixin)
- **mloda_plugins**: PythonDictFramework
- **Python standard library**: hashlib, re, statistics (no external deps)

## References

- [Proposal 9: Provider Inheritance Pattern](../discussion/architecture_proposal/proposal_9_provider_inheritance.md)
- [Build Specification](../discussion/meeting/01_feb_25/build-spec.md)
- [mloda Documentation](https://mloda-ai.github.io/mloda/)
