# Architecture Proposals for mloda-RAG

Nine architectural approaches using mloda's plugin architecture.

## Quick Comparison

| # | Proposal | Core Idea | Best For |
|---|----------|-----------|----------|
| 1 | [Flat FGs](proposal_1_flat_featuregroups.md) | 1 FG per operation | Max flexibility |
| 2 | [Domain-Based](proposal_2_domain_based.md) | mloda Domains | Vertical markets |
| 3 | [Provider Options](proposal_3_provider_options.md) | Generic FGs + options | Provider switching |
| 4 | [Composite FGs](proposal_4_composite_featuregroups.md) | Pipeline templates | Simple UX |
| 5 | [Extender-Driven](proposal_5_extender_based.md) | Cross-cutting via Extenders | Compliance/audit |
| 6 | [CWA Layers](proposal_6_cwa_layers.md) | 10 structured layers | Structured design |
| 7 | [Property-Driven](proposal_7_property_driven.md) | 20 orthogonal properties | Enterprise |
| 8 | [Base+Framework](proposal_8_base_framework_split.md) | Pandas/Polars/Spark | Multi-framework |
| 9 | [Provider Inheritance](proposal_9_provider_inheritance.md) | Presidio/OpenAI/etc | Multi-provider |

## Two Inheritance Axes (8 & 9)

```
Provider (WHAT library)          Framework (HOW to process)
─────────────────────►          │
                                ▼
BasePIIRedactor                 BaseChunker
├── Presidio                    ├── PandasChunker
├── SpaCy                       ├── PolarsChunker
└── AWS Comprehend              └── SparkChunker

Can combine: PresidioPandasPIIRedactor, OpenAISparkEmbedder
```

## mloda Concepts Used

| Concept | Proposals |
|---------|-----------|
| `FeatureChainParserMixin` | All |
| `PROPERTY_MAPPING` | All |
| `compute_framework_rule()` | 8, 9 |
| `Domain` | 2, 7 |
| `Extender` | 5, 7 |

## Recommendation Matrix

| Requirement | Proposals |
|-------------|-----------|
| Quick prototype | 1 or 4 |
| Multi-framework | **8** |
| Multi-provider | **9** or 3 |
| Healthcare/HIPAA | 2 + 5 + 9 |
| Enterprise | 5 + 7 |

## Hybrid Approach

```python
# Base + Provider (9) + Framework (8)
class PresidioPandasPIIRedactor(PresidioPIIRedactor):
    def compute_framework_rule(cls): return {PandasDataFrame}

# Usage with chaining (1) + Extenders (5)
mloda.run_all(
    features=["docs__pii_redacted_presidio__embedded_openai"],
    function_extender={AuditExtender(), LineageExtender()}
)
```

## Next Steps

1. Decide inheritance strategy (8, 9, or both)
2. Define which operations need variants
3. Build minimal prototype
