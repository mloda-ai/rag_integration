# Getting Started

There are two ways to create your own mloda plugin from this template.

## Option 1: Use as Template (Recommended)

GitHub's template feature creates a new repository with all files but a clean commit history.

1. Click the green **"Use this template"** button on the [repository page](https://github.com/mloda-ai/mloda-plugin-template)
2. Choose **"Create a new repository"**
3. Name your repository (e.g., `acme-features`, `acme-data-plugins`, or `mycompany-feature-groups`)

   > **Note:** Please don't use "mloda" in your package or repository name. However, you can use formulations like "for mloda" or "mloda-compatible" in descriptions.

4. Choose public or private visibility
5. Click **"Create repository"**

**Advantages:**
- Clean commit history (starts fresh)
- No fork relationship to maintain
- Your repo is completely independent
- One-click setup in GitHub UI

## Option 2: Fork

Forking creates a copy that maintains a connection to the original repository.

1. Click the **"Fork"** button on the [repository page](https://github.com/mloda-ai/mloda-plugin-template)
2. Choose your account or organization
3. Name your forked repository
4. Click **"Create fork"**

**Advantages:**
- Can pull updates from the original template
- Familiar workflow for open source contributors

**Disadvantages:**
- Maintains fork relationship (shows "forked from" on your repo)
- Copies entire commit history
- GitHub may suggest contributing back to the original

## After Creating Your Repository

Regardless of which option you chose, follow the setup steps in the [README](../README.md#setup-your-plugin) to customize the template:

1. Clone your new repository locally
2. Rename the `placeholder/` directory
3. Update `pyproject.toml`
4. Update Python imports
5. Verify with `tox`
