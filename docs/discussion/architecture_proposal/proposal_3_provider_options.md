# Proposal 3: Generic FeatureGroups with Provider Options

Small set of generic FeatureGroups that accept provider configuration via Options. Provider selection at runtime, not via feature name.

## Architecture

```
Feature("docs__chunked__embedded__stored", Options(context={
    "chunker": "semantic",
    "embedder": "openai",
    "store": "pgvector"
}))
        │
        ▼
┌──────────┐  ┌──────────┐  ┌──────────┐
│ Chunker  │→ │ Embedder │→ │  Store   │
│ (generic)│  │ (generic)│  │ (generic)│
└────┬─────┘  └────┬─────┘  └────┬─────┘
     ▼             ▼             ▼
  Provider      Provider      Provider
  Registry      Registry      Registry
```

## Structure

```
mloda_rag/
├── feature_groups/
│   ├── chunker.py      # Generic with provider selection
│   ├── embedder.py
│   └── store.py
└── providers/
    ├── chunking/       # fixed.py, semantic.py
    ├── embedding/      # openai.py, sentence_tx.py
    └── storage/        # pgvector.py, duckdb.py
```

## Example FeatureGroup

```python
EMBEDDING_PROVIDERS = {"openai": OpenAIProvider, "sentence_tx": SentenceTransformersProvider}

class Embedder(FeatureChainParserMixin, FeatureGroup):
    PREFIX_PATTERN = r".*__embedded$"

    PROPERTY_MAPPING = {
        "provider": {
            "openai": "OpenAI embeddings",
            "sentence_tx": "Sentence Transformers",
            DefaultOptionKeys.mloda_context: True,
            DefaultOptionKeys.default: "openai",
        },
    }

    @classmethod
    def calculate_feature(cls, data, features):
        provider_name = features.features[0].options.get("provider", "openai")
        provider = EMBEDDING_PROVIDERS[provider_name]()
        return provider.embed(data)
```

## Usage

```python
mloda.run_all(
    features=[Feature("documents__chunked__embedded", Options(
        context={"embedding_provider": "openai", "embedding_model": "text-embedding-3-small"}
    ))]
)
```

## Pros/Cons

| Pros | Cons |
|------|------|
| Fewer FGs to maintain | Less discoverable |
| Provider explicit in options | Providers not mloda plugins |
| Easy to add new providers | Config can get verbose |

## Discussion Questions

1. Should providers also be mloda plugins?
2. How to validate provider-specific options?
3. Default providers per environment?
