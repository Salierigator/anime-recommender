# service/backend/

FastAPI service that turns a MyAnimeList username (or a hand-picked list of anime) into recommendations, plus a small CLI that does the same thing without a server. It's a **read-only consumer of the trained model in `artifacts/`** — it loads the export and serves; it never trains, and it picks up a newly trained model with no code change.

## Mock vs. real mode

The one setting that matters is `MOCK_MODE`:

- **Mock mode** (`MOCK_MODE=1`, the default) — returns canned fixtures and **loads no model**. This is the mode to use for frontend work: it starts instantly and pulls in none of the heavy ML dependencies (torch / lightgbm).
- **Real mode** (`MOCK_MODE=0`) — loads `artifacts/` on startup (~5s) and serves real recommendations. A `username` is always looked up live on MyAnimeList (needs `MAL_CLIENT_ID` in `service/.env`); a list of `mal_ids` goes straight through the model and needs no MAL key.

## Run (HTTP API)

```bash
cd service/backend
pip install -r requirements.txt                        # once, inside the project venv

# Mock mode — fixtures, no model. Best for frontend work.
MOCK_MODE=1 uvicorn app.main:app --reload --port 8000

# Real mode — loads artifacts/; MAL_CLIENT_ID in service/.env for live usernames
MOCK_MODE=0 uvicorn app.main:app --port 8000
```

Interactive API docs (Swagger UI) are served at `http://localhost:8000/docs`.

## Run (CLI, no server)

The CLI gives you the same recommendations in the terminal. Run it from the repo root:

```bash
# by MyAnimeList username
venv/bin/python service/backend/recommend.py <username> [--top-k 20] [--cold-k 10] [--live]

# by a file of MAL ids — no MAL key needed, works fully offline
venv/bin/python service/backend/recommend.py --mal-ids service/backend/fixtures/dummy_mal_ids.txt
```

## Endpoints

Full request/response shapes are in the interactive API docs — Swagger UI at `/docs` once the server is running. Every route is under the `/api` prefix.

| Endpoint | What it does |
|---|---|
| `GET /api/health` | Server status + current mode |
| `POST /api/recommend` | The main call: a username or a list of anime → `{ main, cold, meta }` |
| `POST /api/posters` | Batch poster + fresh score/members for a list of ids |
| `GET /api/anime/{mal_id}` | Full details for one anime (used by the detail modal) |
| `GET /api/search?q=` | Title autocomplete for the guest picker |
| `GET /api/users/{username}/exists` | Quick check that a MAL username exists |
| `GET /api/map` · `GET /api/map/territory.png` | 2-D catalog map (optional map view) |

Recommendations come back in **two lists**: `main` (reranked by the LightGBM model) and `cold` (newly released titles the model never trained on, ranked by cosine). The frontend shows them as separate sections.

## Layout

```
app/
├── main.py          # create_app() + startup: picks the mock or real service
├── config.py        # settings (mock_mode, CORS, …)
├── api/routes/      # one file per endpoint group (health, recommend, posters, anime, users, map)
├── schemas/         # Pydantic request/response models
├── services/        # business logic — mock + real behind one interface
├── ml/              # serving core: Recommender (retriever + ranker), AnimeMap
└── clients/         # external APIs: MAL v2 + Jikan
recommend.py         # CLI entry point
fixtures/            # sample payloads for mock mode
tests/               # API tests (run in mock mode)
```

## Tests

```bash
MOCK_MODE=1 venv/bin/python -m pytest service/backend/tests -q
```
