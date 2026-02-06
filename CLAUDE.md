# CLAUDE.md

Must read [AGENTS.md](AGENTS.md) first.

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
