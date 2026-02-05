# Proposal 1: Flat FeatureGroup Structure

Each RAG operation is a standalone FeatureGroup using `FeatureChainParserMixin`. Users chain transformations via `__` syntax.

## Feature Chain Example

```python
# mloda resolves: documents → ocr → pii_redacted → chunked → embedded
Feature(name="documents__ocr__pii_redacted__chunked__embedded")
```

## Structure

```
mloda_rag/feature_groups/
├── ingestion/      # documents.py, ocr.py
├── governance/     # pii_redact.py, phi_anonymize.py
├── chunking/       # fixed_chunk.py, semantic_chunk.py
├── embedding/      # openai_embed.py, sentence_tx_embed.py
└── storage/        # pgvector_store.py, duckdb_store.py
```

## Example FeatureGroup

```python
class PIIRedact(FeatureChainParserMixin, FeatureGroup):
    PREFIX_PATTERN = r".*__pii_redacted$"
    MIN_IN_FEATURES = 1
    MAX_IN_FEATURES = 1

    PROPERTY_MAPPING = {
        "redact_types": {
            DefaultOptionKeys.mloda_context: True,
            DefaultOptionKeys.default: ["email", "phone", "ssn"],
        },
        DefaultOptionKeys.in_features: {DefaultOptionKeys.mloda_context: True},
    }

    @classmethod
    def calculate_feature(cls, data, features):
        # Redaction implementation
        pass
```

## Usage

```python
# Basic pipeline
mloda.run_all(features=["documents__chunked_semantic__embedded_openai__stored_pgvector"])

# Healthcare pipeline
mloda.run_all(features=["pdfs__ocr__phi_anonymized__chunked__embedded"])
```

## Pros/Cons

| Pros | Cons |
|------|------|
| Simple, direct mapping | Many small files |
| Full mloda chaining | No higher-level abstraction |
| Each FG is testable | Provider selection only via name suffix |

## Discussion Questions

1. Naming: `embedded_openai` vs `openai_embedded`?
2. Strategy in name or options?
3. How to handle provider-specific options?
