# Repository Guidelines

## Project Structure & Module Organization
The root contains hospital-oriented entry scripts (`main__epurun.py`, `analysis__hana_ent.py`) that orchestrate ingestion and embedding. Core configuration and domain models live in `core/`, while feature engineering and data loaders reside under `models/data_container` and `models/container`. Embedding logic is centralized in `models/embed/openai_embedding.py`, and gradient boosting assets sit in `models/gradient_boosting/`. Experimental routines live in `pipeline/backtesting_experiment.py` and supporting utilities in `utils/` (authentication, logging). Raw hospital files should be staged in `data/external/<hospital>/`. MLflow tracking code is in `mlflow_integration/`, and generated runs land in `mlruns/`—keep this directory lightweight in commits.

## Build, Test, and Development Commands
Run `uv sync` to install locked dependencies, then `source .venv/bin/activate` for the local environment. Use `uv run python main__epurun.py` or the corresponding hospital script to execute end-to-end ingestion and embedding. `uv run python pipeline/backtesting_experiment.py` spins up the clustering workflow against the latest configuration. To launch the MLflow UI locally, run `uv run mlflow ui --backend-store-uri mlruns`.

## Coding Style & Naming Conventions
Target Python 3.12 with 4-space indentation, type hints, and docstrings for public interfaces. Follow the existing double-underscore naming (`main__<hospital>.py`, `analysis__<hospital>.py`, `patient_<hospital>.py`) for hospital-specific assets. Keep configuration centralized through `core.config.PipelineConfig` rather than ad-hoc dictionaries, and prefer descriptive logging via `utils.logging`.

## Testing Guidelines
Tests should sit in a `tests/` tree mirroring the module layout (e.g., `tests/models/test_openai_embedding.py`). Use `pytest` style names (`test_<behavior>`) and parametrization for hospital variants. Run `uv run pytest -q` before pushing. When MLflow artifacts are involved, isolate them via temporary directories so they do not pollute `mlruns/`.

## Commit & Pull Request Guidelines
Commit messages follow the `<type>: <imperative>` style (`add: new patient loader`, `fix: handle empty charts`). Group related changes, and avoid mixing formatting with logic. PRs need a concise summary, reproduction steps or commands, linked issues, and screenshots or MLflow run IDs when behavior changes. Highlight any schema updates or new environment variables so reviewers can refresh their `.env`.

## Security & Data Handling
Never commit protected health information or live API keys. Use `.env` for secrets and share sample values via `.env.example` when necessary. Remove large or derived artifacts from commits—coordinate long-term storage through the data engineering team.
