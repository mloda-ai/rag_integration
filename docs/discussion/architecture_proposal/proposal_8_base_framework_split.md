# Proposal 8: Base + Framework-Specific Pattern

*Based on mloda_plugins aggregation pattern*

Each RAG operation has a **base class** with shared logic and **framework-specific subclasses** (Pandas, Polars, Spark) implementing data operations.

## Pattern

```
BaseChunker (ABC)
├── PREFIX_PATTERN, PROPERTY_MAPPING
├── calculate_feature() orchestration
└── Abstract: _chunk_text(), _get_column()
        │
        ├── PandasChunker     (compute_framework = PandasDataFrame)
        ├── PolarsChunker     (compute_framework = PolarsLazyDataFrame)
        └── SparkChunker      (compute_framework = SparkDataFrame)
```

## Structure

```
mloda_rag/feature_groups/
├── chunking/
│   ├── base.py       # BaseChunker (abstract)
│   ├── pandas.py     # PandasChunker
│   ├── polars.py     # PolarsChunker
│   └── spark.py      # SparkChunker
├── embedding/
│   ├── base.py, pandas.py, polars.py, spark.py
```

## Base Class

```python
class BaseChunker(FeatureChainParserMixin, FeatureGroup):
    PREFIX_PATTERN = r".*__chunked$"

    @classmethod
    def calculate_feature(cls, data, features):
        texts = cls._get_text_column(data, source_col)  # Abstract
        chunks = cls._chunk_texts(texts, strategy)       # Abstract
        return cls._add_result(data, feature.name, chunks)  # Abstract

    @classmethod
    @abstractmethod
    def _get_text_column(cls, data, col): ...

    @classmethod
    @abstractmethod
    def _chunk_texts(cls, texts, strategy): ...
```

## Framework Implementation

```python
class PandasChunker(BaseChunker):
    @classmethod
    def compute_framework_rule(cls):
        return {PandasDataFrame}

    @classmethod
    def _get_text_column(cls, data, col):
        return data[col]  # Pandas Series

    @classmethod
    def _chunk_texts(cls, texts, strategy):
        return texts.apply(lambda t: chunk_fn(t))
```

## Usage

```python
# mloda picks implementation based on compute_framework
mloda.run_all(features=["docs__chunked"], compute_frameworks=["PandasDataFrame"])   # → PandasChunker
mloda.run_all(features=["docs__chunked"], compute_frameworks=["SparkDataFrame"])    # → SparkChunker
```

## Pros/Cons

| Pros | Cons |
|------|------|
| Scale-transparent (Pandas→Spark) | More files (base + impls) |
| Framework-optimized operations | Boilerplate per implementation |
| DRY shared logic | Testing burden |

## Discussion Questions

1. Which operations need framework variants?
2. Chunking logic in base or implementations?
3. Minimum supported frameworks?
