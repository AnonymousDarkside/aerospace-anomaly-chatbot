# Automatic Semantic Versioning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Automate semantic versioning for the project using `python-semantic-release` integrated into the existing CI pipeline.

**Architecture:** Update `pyproject.toml` and `app/__init__.py` to support `python-semantic-release`, and integrate a release job into `.github/workflows/ci.yml`.

**Tech Stack:** `python-semantic-release`, GitHub Actions.

---

### Task 1: Update `pyproject.toml`

**Files:**
- Modify: `/Personal/pyproject.toml`

- [ ] **Step 1: Update `pyproject.toml` with `[project]` and `[tool.semantic_release]`**

```toml
[project]
name = "aerospace"
version = "1.0.0"
description = "Aerospace project with automatic semantic versioning"
authors = [{ name = "User", email = "user@example.com" }]

[tool.semantic_release]
version_variables = ["app/__init__.py:__version__"]
version_toml = ["pyproject.toml:project.version"]
commit_parser = "angular"
upload_to_pypi = false
upload_to_release = true
build_command = "false" # We don't need to build a package for now

[tool.semantic_release.branches.main]
match = "main"
```

- [ ] **Step 2: Commit**

```bash
git add pyproject.toml
git commit -m "chore: update pyproject.toml for semantic-release"
```

### Task 2: Update `app/__init__.py`

**Files:**
- Modify: `/Personal/app/__init__.py`

- [ ] **Step 1: Add `__version__` to `app/__init__.py`**

```python
__version__ = "1.0.0"
```

- [ ] **Step 2: Commit**

```bash
git add app/__init__.py
git commit -m "chore: add __version__ to app/__init__.py"
```

### Task 3: Integrate with GitHub Actions

**Files:**
- Modify: `/Personal/.github/workflows/ci.yml`

- [ ] **Step 1: Add `release` job to `ci.yml`**

```yaml
  release:
    name: Semantic Release
    needs: [docker-build]
    runs-on: ubuntu-latest
    concurrency: release
    permissions:
      contents: write
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Python Semantic Release
        uses: python-semantic-release/python-semantic-release@v9.15.2
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
```

- [ ] **Step 2: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: add semantic-release job to ci.yml"
```

### Task 4: Verification (Local)

- [ ] **Step 1: Dry-run `python-semantic-release` locally**

Run: `pip install python-semantic-release && semantic-release --noop version`
Expected: Output showing the next version bump based on commit history.

- [ ] **Step 2: Finalize**

```bash
git push origin demo
```
