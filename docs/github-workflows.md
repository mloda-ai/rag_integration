# GitHub Workflows

This project uses three GitHub Actions workflows to automate testing and releases.

Dependency CVE scanning is left to GitHub's native Dependabot security alerts (enable them under **Settings > Code security**), with `tox -e security` (`pip-audit`) available for on-demand local scans. Pulling later template changes is a deliberate, manual step (fetch the template remote and merge), not an automated job.

## Test Workflow

**File:** `.github/workflows/test.yml`

**Triggers:**
- Push to any branch
- Pull request to `main`

**Purpose:** Runs the full test suite using tox across multiple Python versions (3.11, 3.12, 3.13, 3.14). This includes pytest, ruff linting, mypy type checking, and bandit security analysis.

**Requirements:** None. This workflow uses only public GitHub Actions and requires no secrets.

## Release Workflow

**File:** `.github/workflows/release.yaml`

**Triggers:**
- Manual dispatch only

**Purpose:** Automates semantic versioning and publishing to TestPyPI. The workflow:
1. Analyzes commits using [semantic-release](https://semantic-release.gitbook.io/) to determine the next version
2. Creates a GitHub release with changelog
3. Updates version in `pyproject.toml`
4. Builds the package once from the release tag, checks the wheel contents (no `conftest.py` or `tests/` path, package data present), and publishes it to TestPyPI
5. Installs the wheel TestPyPI serves and imports it (smoke test)
6. Attaches the smoke-tested files to the GitHub release

A release does not reach PyPI on its own: run the Publish to PyPI workflow afterwards.

**Requirements:**
- `SEMANTIC_RELEASE_TOKEN` - GitHub Personal Access Token with `repo` write permissions
- TestPyPI Trusted Publisher (no secret), see below

**Prerequisites:** Commits must follow the [Conventional Commits](https://www.conventionalcommits.org/) format for semantic-release to determine version bumps:
- `fix:` - Patch release (0.0.x)
- `feat:` - Minor release (0.x.0)
- `feat!:` or `BREAKING CHANGE:` - Major release (x.0.0)

## Publish to PyPI Workflow

**File:** `.github/workflows/publish-pypi.yaml`

**Triggers:**
- Manual dispatch from `main`, with the released `version` as input

**Purpose:** Downloads the files the Release workflow attached to the GitHub release, checks they match what TestPyPI serves, and uploads them to PyPI. Nothing reaches PyPI without passing the TestPyPI smoke test first.

**Requirements:**
- PyPI Trusted Publisher (no secret), see below

## Setting Up Secrets

To configure the required secrets for the release workflow:

### SEMANTIC_RELEASE_TOKEN

1. Go to GitHub **Settings** > **Developer settings** > **Personal access tokens** > **Tokens (classic)**
2. Click **Generate new token (classic)**
3. Give it a descriptive name (e.g., "semantic-release")
4. Select the `repo` scope (full control of private repositories)
5. Click **Generate token** and copy the token
6. In your repository, go to **Settings** > **Secrets and variables** > **Actions**
7. Click **New repository secret**
8. Name: `SEMANTIC_RELEASE_TOKEN`
9. Value: Paste the token
10. Click **Add secret**

### Trusted Publishing (PyPI and TestPyPI)

Uploads use [Trusted Publishing](https://docs.pypi.org/trusted-publishers/) (OIDC), so no API token is stored. Set up the TestPyPI publisher before the next Release run (otherwise it stops at the upload; re-run the failed jobs once fixed) and the PyPI publisher before the first Publish to PyPI run.

1. In the repository, go to **Settings** > **Environments** and create `testpypi` and `pypi`, each with a deployment branch rule allowing only `main`, plus required reviewers on `pypi`. The publishers do not check the branch, and the workflows' own `main` check can be edited away on another branch, so this rule is what guards publishing
2. On TestPyPI, where the project does not exist yet, add a pending publisher (it does not reserve the name, so add it right before the first Release run) under [Your account > Publishing](https://test.pypi.org/manage/account/publishing/): project `rag-integration`, owner `mloda-ai`, repository `rag_integration`, workflow `release.yaml`, environment `testpypi`
3. On [PyPI](https://pypi.org/manage/project/rag-integration/settings/publishing/), add a GitHub publisher: owner `mloda-ai`, repository `rag_integration`, workflow `publish-pypi.yaml`, environment `pypi`
4. After the first successful PyPI publish, delete the old `PYPI_API_TOKEN` repository secret and revoke the token on PyPI
