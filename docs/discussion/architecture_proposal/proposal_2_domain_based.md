# Proposal 2: Domain-Based FeatureGroups

Use mloda's **Domain** concept to organize by use case. Same feature names resolve to different implementations based on domain context.

## Architecture

```
┌─────────────────────────────────────────────────┐
│  Feature("docs__pii_redacted__chunked")         │
│  Domain: "healthcare" | "legal" | "enterprise"  │
└─────────────────────────────────────────────────┘
                    │
    ┌───────────────┼───────────────┐
    ▼               ▼               ▼
Healthcare       Legal          Enterprise
(PHI rules)    (Citations)    (DLP + Audit)
```

## Structure

```
mloda_rag/feature_groups/
├── core/                    # Domain-agnostic
│   ├── documents.py
│   └── chunking.py
└── domains/
    ├── healthcare/          # PHI redaction, medical NER
    ├── legal/               # Citation extraction, precedents
    └── enterprise/          # DLP, access control, audit
```

## Domain Definition

```python
class HealthcareDomain(Domain):
    @classmethod
    def domain_name(cls) -> str:
        return "healthcare"

class PHIRedact(FeatureChainParserMixin, FeatureGroup):
    PREFIX_PATTERN = r".*__pii_redacted$"

    @classmethod
    def domain(cls):
        return HealthcareDomain  # Only matches in healthcare context
```

## Usage

```python
# Healthcare context - uses PHI redaction
mloda.run_all(
    features=["documents__pii_redacted__chunked"],
    domain=HealthcareDomain,
)

# Legal context - uses legal-specific redaction
mloda.run_all(
    features=["documents__pii_redacted__chunked"],
    domain=LegalDomain,
)
```

## Pros/Cons

| Pros | Cons |
|------|------|
| Same syntax, domain-specific behavior | More complex mental model |
| Clean compliance separation | Confusion if domain not specified |
| Aligns with build-spec verticals | Need to understand domain resolution |

## Discussion Questions

1. Should there be a "default" domain?
2. Can domains inherit from each other?
3. How to handle features that span domains?
