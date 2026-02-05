# AGENTS.md

Must read [README.md](README.md) first.

This project uses the mloda framework. Assume any given task is related to mloda.

## Environment

```bash
source .venv/bin/activate
```

## Dependencies

Use `uv` to install dependencies:
```bash
uv sync --all-extras
```

## Running checks

Use `tox` to run all checks:
```bash
tox
```

### Run individual checks

```bash
pytest
ruff format --check --line-length 120 .
ruff check .
mypy --strict --ignore-missing-imports .
bandit -c pyproject.toml -r -q .
```

## Claude Code Skills

This repository includes skills for mloda plugin development:

- `/mloda-plugins` - Plugin development guides with decision trees for FeatureGroups, ComputeFrameworks, and Extenders
- `/mloda-core` - Core library reference for understanding mloda internals and API

When helping with FeatureGroups, ComputeFrameworks, or Extenders, use these skills for pattern guidance and best practices.
