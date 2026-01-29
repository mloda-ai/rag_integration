# Proposal 6: CWA Layer-Based Architecture

*Inspired by [mvp-cwa](~/project/mvp/mvp-cwa)*

Map RAG pipeline stages to **Context Window Architecture layers**. Each layer is a FeatureGroup. Assemblers compose layers into pipelines.

## RAG Layers

```
L1: Source Documents     │ Raw input
L2: Extracted Content    │ OCR, parsing
L3: Governance Applied   │ PII redaction, access tags
L4: Domain Enriched      │ NER, citations
L5: Chunked Content      │ Semantic/hierarchical
L6: Embedded Vectors     │ Vector representations
L7: Stored Index         │ Persisted to vector store
L8: Retrieved Context    │ Query-time retrieval
L9: Re-ranked Results    │ Relevance scoring
L10: Assembled Context   │ Final context for LLM
```

## Structure

```
mloda_rag/
├── layers/              # One FG per layer
│   ├── l1_source.py ... l10_assembly.py
├── assemblers/          # Pipeline compositions
│   ├── basic_rag.py     # L1→L5→L6→L7
│   ├── healthcare_rag.py # + L3 (PHI) + L4 (medical NER)
│   └── query_pipeline.py # L8→L9→L10
```

## Assembler Example

```python
class HealthcareRAGAssembler(FeatureGroup):
    LAYERS = ["source", "extracted", "governance", "enriched", "chunked", "embedded", "stored"]
    LAYER_CONFIG = {
        "governance": {"rules": ["phi_anonymize", "access_tag"]},
        "enriched": {"type": "medical_ner"},
    }

    @classmethod
    def input_features(cls, options, feature_name):
        return {Feature(layer) for layer in cls.LAYERS}
```

## Usage

```python
# Index with healthcare assembler
mloda.run_all(features=["documents__healthcare_rag_indexed"])

# Dynamic layer composition
mloda.run_all(features=[Feature("assembler", Options(context={
    "_layers": ["source", "extraction", "governance", "chunking", "embedding"]
}))])
```

## Pros/Cons

| Pros | Cons |
|------|------|
| Structured thinking | More abstraction |
| Composable layers | Rigid ordering |
| Clear boundaries, auditable | Learning curve |

## Discussion Questions

1. Are 10 layers the right granularity?
2. Should layers be skippable?
3. Layer-specific providers?
