# Repository Guidelines

## Project Structure & Module Organization
Primary code lives in `django_lightrag/`. Key modules include `views.py`, `urls.py`, `schemas.py`, `core.py`, `entity_extraction.py`, `graph_builder.py`, `query_engine.py`, and `storage.py`. Django management commands live in `django_lightrag/management/commands/`, and schema changes belong in `django_lightrag/migrations/`. Put tests under `django_lightrag/tests/`; this directory exists but needs broader coverage. Treat `build/`, `*.egg-info/`, `.pytest_cache/`, and `.ruff_cache/` as generated output, not hand-edited source.

## Build, Test, and Development Commands
Use `uv` for environment and command execution.

```bash
uv sync --extra dev          # install runtime and dev dependencies
uv run pytest                # run the test suite
uv run ruff check .          # lint Python code
uv run ruff format .         # format Python code
uvx pre-commit run --all-files
uv build                     # build the package
```

Repo-local agent rules also expect backend work to use Django/Ninja conventions and to run the server on port `8001` via `uv run manage.py runserver 8001` when working from an integration project with a `manage.py`.

## Coding Style & Naming Conventions
Target Python 3.13+ with 4-space indentation, type hints, and an 88-character line length. Follow Ruff and Black-compatible formatting from `pyproject.toml`. Use `snake_case` for functions, variables, modules, and management commands; use `PascalCase` for classes and Django models. Keep HTTP APIs in `django-ninja` schemas/views, and prefer explicit functions over signals for business logic.

## Testing Guidelines
Use `pytest` with `pytest-django`. Name files `test_*.py` and keep endpoint tests functional by exercising Django Ninja `TestClient` against the HTTP API. Reuse request/response schemas in tests, prefer fixtures, and use `model_bakery` for ORM-backed fixtures when needed. Do not rely on mocks or monkeypatching for API and vector-db behavior.

## Commit & Pull Request Guidelines
Recent history uses short imperative subjects, often with an issue prefix, for example `#23: simplify core.py`. Keep commits focused and describe the behavioral change, not just the file touched. PRs should include a concise summary, linked issue or task, test evidence (`uv run pytest`, `uv run ruff check .`), and example API calls or terminal output when endpoints or CLI behavior change.

## Repository-Specific Rules
Read `.agents/rules/` before substantial changes. Current rules require `uv`, `django-ninja`, ChromaDB for vector search, ladybugdb for graph storage, HTTP-first integration between TUI and backend, type hints with `django-stubs`, and functional endpoint tests without mocks.
