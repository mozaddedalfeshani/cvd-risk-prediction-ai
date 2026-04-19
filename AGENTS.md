# Repository Guidelines

## Project Structure & Module Organization
- `frontend/` contains the Next.js 15 app: routes in `frontend/src/app`, reusable UI in `frontend/src/components`, helpers in `frontend/src/lib`, and static assets in `frontend/public`.
- `api/` contains the Flask prediction service. Entrypoints are `api/app/app.py` and `api/app/app_dual.py`.
- `ml-models/` holds training, preprocessing, evaluation, raw data, and saved model artifacts.
- `docs/research/` and `notebooks/` store analysis outputs. Root files such as `test_backend.py`, `test_cases.json`, and `high_risk_test.json` act as integration fixtures.

## Build, Test, and Development Commands
- `pip install -r requirements.txt` installs tooling; use `api/requirements.txt` and `ml-models/requirements.txt` when needed.
- `python api/app/app.py` starts the mock prediction API. Use `python api/app/app_dual.py` to run the dual-model backend.
- `cd frontend && npm install && npm run dev` starts the frontend on `http://localhost:3000`.
- `cd frontend && npm run build` creates a production build; `cd frontend && npm run lint` runs ESLint.
- `python test_backend.py` runs the repository’s backend integration script; start the API first.
- `docker compose -f docker/docker-compose.yml up --build` runs the frontend and API together.

## Coding Style & Naming Conventions
- Python follows PEP 8: 4-space indentation, `snake_case` for modules/functions, `PascalCase` for classes.
- TypeScript and React files use `PascalCase` for components such as `PredictionResult.tsx`.
- Follow the surrounding file style when editing, because formatting is not fully uniform across the repository.
- Preferred quality tools are `black`, `isort`, and `flake8` for Python, plus ESLint in `frontend/`.

## Testing Guidelines
- Add focused tests close to the area you change when practical; for backend behavior, extend the root fixtures and `test_backend.py`.
- Name Python tests `test_*.py`.
- Before opening a PR, run `cd frontend && npm run lint` and execute `python test_backend.py` against a running backend.
- There is no enforced coverage gate yet, so prioritize meaningful scenario coverage.

## Commit & Pull Request Guidelines
- Follow the existing Conventional Commit pattern from history: `feat: ...`, `fix: ...`, `docs: ...`.
- Keep commits scoped to one change set and write messages in the imperative mood.
- PRs should include a short summary, affected directories, validation commands, and screenshots for UI changes.
- Call out any model, dataset, or API contract changes clearly so reviewers can validate them quickly.

## Security & Configuration Tips
- Do not commit secrets or private patient data. Use environment files such as `frontend/.env.local` for runtime configuration.
- Treat `ml-models/data/raw/` and `ml-models/models/` as high-impact assets; update them only when required.
