# Proposal 4: Composite FeatureGroups (Pipeline Templates)

High-level "pipeline template" FeatureGroups that internally chain multiple transformations. Users request end results; composite handles orchestration.

## Architecture

```
Feature("documents__healthcare_ready")
                │
                ▼
┌─────────────────────────────────────┐
│     HealthcareRAG Composite         │
│  Internally chains:                 │
│  docs → ocr → phi → ner → chunk →   │
│  embed → audit                      │
└─────────────────────────────────────┘
                │
    ┌───────────┼───────────┐
    ▼           ▼           ▼
  Atomic     Atomic      Atomic
    FGs        FGs         FGs
```

## Structure

```
mloda_rag/feature_groups/
├── atomic/              # Single-responsibility FGs
│   ├── ocr.py, pii_redact.py, chunking/, embedding/, storage/
└── composite/           # Pipeline templates
    ├── basic_rag.py         # docs → chunk → embed → store
    ├── healthcare_rag.py    # + PHI + medical NER + audit
    ├── legal_rag.py         # + citations + precedents
    └── enterprise_rag.py    # + DLP + access control
```

## Example Composite

```python
class HealthcareRAG(FeatureChainParserMixin, FeatureGroup):
    PREFIX_PATTERN = r".*__healthcare_ready$"

    PROPERTY_MAPPING = {
        "embedding_provider": {DefaultOptionKeys.default: "openai"},
        "store_provider": {DefaultOptionKeys.default: "pgvector"},
    }

    @classmethod
    def calculate_feature(cls, data, features):
        embed = features.features[0].options.get("embedding_provider", "openai")
        chain = f"docs__ocr__phi_anonymized__chunked__embedded_{embed}"
        return mloda.run_all(features=[chain], api_data={"source": data})
```

## Reference Pipelines

| Template | Internal Chain |
|----------|----------------|
| `basic_ready` | chunk → embed → store |
| `healthcare_ready` | ocr → phi → ner → chunk → embed → audit |
| `legal_ready` | citations → precedents → chunk → graphrag |

## Usage

```python
mloda.run_all(features=["documents__healthcare_ready"])
mloda.run_all(features=[Feature("documents__legal_ready", Options(context={"store": "duckdb"}))])
```

## Pros/Cons

| Pros | Cons |
|------|------|
| Very simple user API | Less flexibility |
| Encapsulates complexity | Composite depends on many FGs |
| Easy for LLMs | Debugging chains of chains |

## Discussion Questions

1. Should composites allow step overrides?
2. How to handle partial execution?
3. Auto-generate composites from config?
