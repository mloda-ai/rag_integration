# Proposal 9: Provider Inheritance Pattern

Base classes define interface, **provider-specific subclasses** implement actual integrations (Presidio, OpenAI, LangChain). Orthogonal to framework pattern (Proposal 8).

## Pattern

```
BasePIIRedactor
├── PresidioPIIRedactor    # Microsoft Presidio
├── SpacyPIIRedactor       # SpaCy NER
├── AWSComprehendPIIRedactor
└── RegexPIIRedactor

BaseEmbedder
├── OpenAIEmbedder
├── SentenceTransformersEmbedder
├── CohereEmbedder
└── OllamaEmbedder
```

## Structure

```
mloda_rag/feature_groups/
├── pii_redaction/
│   ├── base.py           # BasePIIRedactor
│   ├── presidio.py       # PresidioPIIRedactor
│   ├── spacy.py          # SpacyPIIRedactor
│   └── regex.py
├── embedding/
│   ├── base.py, openai.py, sentence_transformers.py, cohere.py
```

## Base Class

```python
class BasePIIRedactor(FeatureChainParserMixin, FeatureGroup):
    PREFIX_PATTERN = r".*__pii_redacted$"

    @classmethod
    def calculate_feature(cls, data, features):
        texts = cls._extract_texts(data, source_col)
        redacted = cls._redact_pii(texts, pii_types)  # Abstract
        return cls._add_result(data, feature.name, redacted)

    @classmethod
    @abstractmethod
    def _redact_pii(cls, texts, pii_types): ...
```

## Provider Implementation

```python
class PresidioPIIRedactor(BasePIIRedactor):
    @classmethod
    def _redact_pii(cls, texts, pii_types):
        from presidio_analyzer import AnalyzerEngine
        from presidio_anonymizer import AnonymizerEngine
        analyzer, anonymizer = AnalyzerEngine(), AnonymizerEngine()
        return [anonymizer.anonymize(t, analyzer.analyze(t)).text for t in texts]
```

## Combined with Framework (Proposal 8)

```python
# Provider → Framework hierarchy
PresidioPIIRedactor
├── PresidioPandasPIIRedactor   # compute_framework = PandasDataFrame
└── PresidioPolarsPIIRedactor   # compute_framework = PolarsLazyDataFrame
```

## Usage

```python
# Provider in feature name
"documents__pii_redacted_presidio__embedded_openai"

# Or provider in options
Feature("documents__pii_redacted", Options(context={"provider": "presidio"}))
```

## Pros/Cons

| Pros | Cons |
|------|------|
| Swappable providers | Many classes (Provider × Framework) |
| Provider-specific options | Import complexity |
| Clear dependencies | Config overhead |

## Discussion Questions

1. Provider in name vs options?
2. Which operations need provider variants?
3. Separate packages (`mloda-rag-presidio`)?
