# Proposal 5: Extender-Driven Cross-Cutting Concerns

Use mloda **Extenders** for cross-cutting concerns (audit, validation, PII detection). FeatureGroups stay focused on transformations.

## Architecture

```
Feature Request → Extender Stack → Core FeatureGroups
                       │
    ┌──────────────────┼──────────────────┐
    ▼                  ▼                  ▼
AuditTrail       PIIValidator      AccessControl
Extender          Extender          Extender
                       │
                       ▼
              Ingest → Chunk → Embed → Store
```

## Structure

```
mloda_rag/
├── feature_groups/      # Pure transformations
│   ├── ingest.py, chunk.py, embed.py, store.py
├── extenders/           # Cross-cutting concerns
│   ├── audit_trail.py, pii_validator.py, access_control.py, cache.py
└── profiles/            # Extender combinations
    ├── development.py   # Minimal logging
    ├── production.py    # Full audit + metrics
    └── hipaa.py         # All + strict PII + encryption
```

## Example Extenders

```python
class AuditTrailExtender(Extender):
    def wraps(self):
        return {ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE}

    def __call__(self, func, feature_group, data, features, *args, **kwargs):
        result = func(feature_group, data, features, *args, **kwargs)
        self.log_audit({"fg": feature_group.__class__.__name__, "ts": time.time()})
        return result

class PIIValidationExtender(Extender):
    def __call__(self, func, feature_group, data, features, *args, **kwargs):
        result = func(feature_group, data, features, *args, **kwargs)
        if self._contains_pii(result):
            raise PIILeakageError("PII detected in output")
        return result
```

## Profiles

```python
# profiles/hipaa.py
HIPAA_EXTENDERS = {
    AuditTrailExtender(retention_days=2555),
    PIIValidationExtender(strict=True),
    AccessControlExtender(require_mfa=True),
}
```

## Usage

```python
# Development
mloda.run_all(features=["docs__chunked__embedded"])

# Production with HIPAA compliance
mloda.run_all(features=["docs__chunked__embedded"], function_extender=HIPAA_EXTENDERS)
```

## Pros/Cons

| Pros | Cons |
|------|------|
| Clean separation | Extender stacks complex to debug |
| Same pipeline, different profiles | Ordering matters |
| Composable, reusable | Some concerns better as FGs |

## Discussion Questions

1. Which concerns: Extenders vs FeatureGroups?
2. How to manage extender ordering?
3. Profiles in code or config?
