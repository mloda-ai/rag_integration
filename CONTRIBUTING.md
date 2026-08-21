# Contributing to rag-integration

Thanks for your interest! rag-integration is a RAG integration plugin for
[mloda](https://github.com/mloda-ai/mloda).

## Development setup

```bash
uv venv && source .venv/bin/activate
uv sync --all-extras
```

## Before you open a PR

`tox` is the merge gate. It must pass:

```bash
tox
```

It runs pytest, `ruff format --check`, `ruff check`, `mypy --strict`, and bandit.
CI runs the same gate on all supported Python versions.

## Ground rules

- **Tests required.** Every feature or fix ships with tests; follow the
  patterns in the existing `tests/` tree.
- **Conventional Commits.** Use `feat:` / `fix:` / `chore:` / `docs:` / etc.;
  release tooling parses them. No `Co-Authored-By` / AI-agent trailers.

## License

By contributing, you agree that your contributions will be licensed under the
[Apache License, Version 2.0](LICENSE).
