# service/

The deployable web app — the final stage of the pipeline, turning a MyAnimeList username into anime recommendations. It's a **read-only consumer of `artifacts/`**: it loads the exported model and picks up a retrain automatically because the file contract doesn't change.

Two halves, each with its own README:

| | |
|---|---|
| [`backend/`](backend/) | FastAPI service + CLI. Loads the trained model and serves recommendations. |
| [`frontend/`](frontend/) | React + Vite + TypeScript single-page UI. |

The two talk over a JSON API.

## Run both locally

```bash
# Backend — mock mode needs no model, best for frontend work
cd service/backend && MOCK_MODE=1 uvicorn app.main:app --reload --port 8000

# Frontend — in another terminal
cd service/frontend && npm install && npm run dev
#   → http://localhost:5173
```

For real recommendations, the CLI, the full endpoint list, and the UI source layout, see [`backend/README.md`](backend/README.md) and [`frontend/README.md`](frontend/README.md).
