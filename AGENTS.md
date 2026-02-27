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

Run checks from the project virtualenv:
```bash
source .venv/bin/activate && tox
```

`tox` is the required final verification step after code or dependency changes.
Running only `pytest` is not sufficient for completion.

### Run individual checks

```bash
pytest
ruff format --check --line-length 120 .
ruff check .
mypy --strict --ignore-missing-imports .
bandit -c pyproject.toml -r -q .
```

## Commit messages

Use Conventional Commit format for all commits so semantic versioning/release tooling can parse intent.

Examples:
- `fix: handle empty feature set`
- `chore(deps): bump mloda to 0.4.6`

## Claude Code Skills

This repository includes skills for mloda plugin development:

- `/mloda-plugins` - Plugin development guides with decision trees for FeatureGroups, ComputeFrameworks, and Extenders
- `/mloda-core` - Core library reference for understanding mloda internals and API

When helping with FeatureGroups, ComputeFrameworks, or Extenders, use these skills for pattern guidance and best practices.
