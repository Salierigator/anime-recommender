# service/

The web app: FastAPI backend + React frontend. Read-only consumer of `artifacts/` — it loads the
exported model, never trains, and picks up a retrain automatically because the file contract
doesn't change.

```
backend/
├── recommend.py   # CLI entry point
├── app/
│   ├── main.py    # create_app() + lifespan (mock → MockService, real → RealService)
│   ├── api/       # routes: health, recommend, posters, map
│   ├── services/  # business logic (mock + real behind one interface)
│   ├── ml/        # serving core: Recommender, AnimeMap
│   └── clients/   # MAL v2 + Jikan
└── fixtures/      # sample payloads for mock mode
frontend/          # React + Vite + TypeScript
```

## Run

```bash
cd service/backend && pip install -r requirements.txt        # once, inside the venv

MOCK_MODE=1 uvicorn app.main:app --reload --port 8000        # mock: fixtures, no model loaded
MOCK_MODE=0 uvicorn app.main:app --port 8000                 # real: loads artifacts/ (~5s)
#   GET /api/health · POST /api/recommend · GET /api/map · Swagger at /docs

cd service/frontend && npm run dev
```

Mock mode is the one to use for frontend work: it returns `fixtures/recommend_sample.json` without
pulling torch/lightgbm into the process. Real mode needs `artifacts/` present, plus
`MAL_CLIENT_ID` in `service/.env` to resolve live usernames.

## CLI

```bash
venv/bin/python service/backend/recommend.py <username> [--top-k 20] [--cold-k 10] [--live]
venv/bin/python service/backend/recommend.py --mal-ids fixtures/dummy_mal_ids.txt   # no MAL key needed
```

API request/response shapes: `service/API_CONTRACT.md`.
