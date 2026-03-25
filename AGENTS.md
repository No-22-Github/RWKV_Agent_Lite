# Repository Guidelines

## Project Structure & Module Organization
`rvoone/` is the main package. Keep feature code grouped by domain: `agent/` for runtime loop and tools, `cli/` for Typer commands, `config/` for TOML-backed settings, `providers/` for model integrations, `channels/` for delivery backends, and `templates/` or `skills/` for packaged markdown assets. Tests live in `tests/` and are mostly `pytest` suites, with one shell-based Docker smoke test at `tests/test_docker.sh`. Deployment files sit at the root (`Dockerfile`, `docker-compose.yml`, `pyproject.toml`).

## Build, Test, and Development Commands
Use a project-local `venv` and install all related packages inside that environment, not against the system Python. Install locally with `python -m pip install -e ".[dev]"` to get the CLI plus test and lint tools. Run the app with `python -m rvoone` or `rvoone --help`; use `rvoone onboard` to create local config and workspace scaffolding. Run tests with `pytest`, or target a file such as `pytest tests/test_cron_service.py`. Lint and format imports with `ruff check .`; if you need import normalization, run `ruff check . --fix`.

## Coding Style & Naming Conventions
Target Python 3.11+ and follow the existing code style: 4-space indentation, type hints on public functions, concise docstrings, and `snake_case` for modules, functions, and variables. Classes use `PascalCase`; constants use `UPPER_SNAKE_CASE`. Keep lines within Ruff’s configured 100-character limit where practical. Prefer small, focused modules that align with the existing package boundaries rather than adding catch-all helpers.

## Testing Guidelines
Use `pytest` for unit and async tests; `pytest-asyncio` is already configured with automatic async handling. Name new tests `test_*.py` and keep test functions descriptive, for example `test_gateway_reuses_loaded_config`. Add or update tests alongside behavior changes, especially for CLI flows, config loading, provider timeouts, and channel integrations. Use `bash tests/test_docker.sh` only for Docker packaging checks.

## Commit & Pull Request Guidelines
Recent history favors short, imperative subjects, often with conventional prefixes such as `feat(agent): ...`, `refactor: ...`, or `Fix ...`. Keep commits scoped to one logical change. Pull requests should explain the user-visible impact, call out config or template changes, link related issues, and include terminal output or screenshots when changing CLI behavior.
