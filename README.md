[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/mloda-ai/mloda-plugin-template/blob/main/LICENSE)
[![mloda](https://img.shields.io/badge/built%20with-mloda-blue.svg)](https://github.com/mloda-ai/mloda)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

# mloda-plugin-template

> **A GitHub template for creating standalone mloda plugins.** Part of the [mloda](https://github.com/mloda-ai/mloda) ecosystem for open data access. Visit [mloda.ai](https://mloda.ai) for an overview and business context, the [GitHub repository](https://github.com/mloda-ai/mloda) for technical context, or the [documentation](https://mloda-ai.github.io/mloda/) for detailed guides.

Create your own FeatureGroups, ComputeFrameworks, and Extenders as standalone packages. See the [Getting Started guide](docs/getting-started.md) to create your repository, then follow the setup steps below.

## Related Repositories

- **[mloda](https://github.com/mloda-ai/mloda)**: The core library for open data access. Declaratively define what data you need, not how to get it. mloda handles feature resolution, dependency management, and compute framework abstraction automatically.

- **[mloda-registry](https://github.com/mloda-ai/mloda-registry)**: The central hub for discovering and sharing mloda plugins. Browse community-contributed FeatureGroups, find integration guides, and publish your own plugins for others to use.

## Structure

```
placeholder/
├── feature_groups/
│   └── my_plugin/
│       ├── __init__.py           # Package exports
│       ├── my_feature_group.py   # Example FeatureGroup implementation
│       └── tests/
│           └── test_my_feature_group.py
├── compute_frameworks/
│   └── my_framework/
│       ├── __init__.py
│       └── my_compute_framework.py
└── extenders/
    └── my_extender/
        ├── __init__.py
        └── my_extender.py
```

## Key Files

- `placeholder/` - Root namespace (users rename to company name)
- `pyproject.toml` - Package config (users edit directly, not auto-generated)
- `.github/workflows/test.yml` - CI workflow running pytest

## Common Tasks

### Setup Your Plugin

Follow these steps to customize the template for your organization:

#### 1. Rename the directory

```bash
mv placeholder acme
```

#### 2. Update pyproject.toml

Edit the following fields in `pyproject.toml`:

- `name`: Change `"placeholder-my-plugin"` to `"acme-my-plugin"`
- `authors`: Update name and email
- `description`: Update to describe your plugin
- `tool.setuptools.packages.find.include`: Change `["placeholder*"]` to `["acme*"]`
- `tool.pytest.ini_options.testpaths`: Change `["placeholder", "tests"]` to `["acme", "tests"]`

#### 3. Update .releaserc.yaml

Edit the following fields in `.releaserc.yaml`:

- `message`: Change `mloda-plugin-template` to your package name (e.g., `"chore(release acme-my-plugin): ${nextRelease.version}"`)
- `repositoryUrl`: Change to your repository URL

#### 4. Update Python imports

Update imports in these files (change `from placeholder.` to `from acme.`):

- `acme/feature_groups/my_plugin/__init__.py`
- `acme/feature_groups/my_plugin/tests/test_my_feature_group.py`
- `acme/compute_frameworks/my_plugin/__init__.py`
- `acme/compute_frameworks/my_plugin/tests/test_my_compute_framework.py`
- `acme/extenders/my_plugin/__init__.py`
- `acme/extenders/my_plugin/tests/test_my_extender.py`

#### 5. Verify setup

```bash
uv venv && source .venv/bin/activate && uv sync --all-extras && tox
```

### Development Setup with uv

**Install uv** (if not already installed):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Create virtual environment and install dependencies:**
```bash
uv venv
source .venv/bin/activate
uv sync --all-extras
```

**Run all checks with tox:**
```bash
# Install tox with uv backend
uv tool install tox --with tox-uv

# Run all checks (pytest, ruff, mypy, bandit)
tox
```

### Run individual checks

```bash
# Tests only
pytest

# Format check
ruff format --check --line-length 120 .

# Lint check
ruff check .

# Type check
mypy --strict --ignore-missing-imports .

# Security check
bandit -c pyproject.toml -r -q .
```

## Related Documentation

Guides for plugin development can be found in mloda-registry:

- https://github.com/mloda-ai/mloda-registry/tree/main/docs/guides/

This template includes pre-configured GitHub Actions workflows for testing, security scanning, and automated releases. See the [GitHub Workflows documentation](docs/github-workflows.md) for setup instructions and required secrets.
