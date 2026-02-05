# Proposal 7: Property-Driven Architecture

*Inspired by [mvp-cwa property examples](~/project/mvp/mvp-cwa/docs/proposals/06_property_examples/)*

Design around **20 key properties** that production RAG systems need. Each property is orthogonal, implemented via FeatureGroups or Extenders.

## Property Matrix

| # | Property | Implementation |
|---|----------|----------------|
| 01 | Lineage | Extender: tracks data origin |
| 02 | Privacy | FG: PII/PHI redaction |
| 03 | Auditability | Extender: audit trails |
| 04 | Reproducibility | Artifacts |
| 05 | Governance | Extender: access control |
| 06 | Versioning | FG versioning |
| 07 | Caching | Extender: result caching |
| 08 | Composability | Feature chaining |
| 09 | Testability | Test fixtures |
| 10 | Observability | Extender: metrics/logging |

## Structure

```
mloda_rag/
├── core/                # Core RAG operations
├── properties/
│   ├── governance/      # lineage, audit, trace extenders
│   ├── security/        # privacy FG, access extender
│   ├── quality/         # validators, type safety
│   └── performance/     # cache extender, spark framework
└── profiles/
    ├── minimal.py, production.py, hipaa.py
```

## Property Profiles

```python
HIPAA_PROFILE = {
    "extenders": {
        LineageExtender(retention_days=2555),
        AuditExtender(detailed=True),
        PIIValidationExtender(strict=True, types="PHI"),
    },
    "domain": "HIPAADomain",
}
```

## Usage

```python
# Select specific properties
mloda.run_all(
    features=["docs__chunked__embedded"],
    function_extender={LineageExtender(), CacheExtender()},
)

# Apply full profile
mloda.run_all(features=["docs__chunked"], function_extender=HIPAA_PROFILE["extenders"])
```

## Property Compatibility

| Profile | Lineage | Privacy | Audit | Cache | Scale |
|---------|---------|---------|-------|-------|-------|
| Minimal | - | - | - | - | - |
| Dev | - | - | - | ✓ | - |
| Prod | ✓ | ✓ | ✓ | ✓ | ✓ |
| HIPAA | ✓ | ✓✓ | ✓✓ | ✓ | ✓ |

## Pros/Cons

| Pros | Cons |
|------|------|
| Explicit requirements | Many moving parts |
| Mix and match | Configuration complexity |
| Compliance-ready profiles | Property interactions subtle |

## Discussion Questions

1. Which properties are must-have?
2. Opt-in or opt-out?
3. How to test property combinations?
