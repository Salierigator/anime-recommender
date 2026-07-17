# Crawler

Collects a fresh anime + user-rating snapshot from MyAnimeList into `data/raw/`
(gitignored). Four scripts share one SQLite state file — every one is rate-limited,
resumable (stop with Ctrl-C, rerun to continue), and dedup-safe.

## Scripts & sources

| Script | Source | Output | Full-crawl time |
|---|---|---|---|
| `collect_usernames.py` | scrape `users.php` (recently-online, ~20/poll) | state DB | background, days |
| `crawl_details.py` | Jikan v4 `/anime?page=N` (~1215 pages) | `details.jsonl.gz` | ~40 min @ 1 req/s |
| `crawl_ratings.py` | MAL API v2 `/users/{u}/animelist` (official, 1000/page) | `ratings.csv` | ~1–2 days / 300k users @ 3 req/s |
| `crawl_profiles.py` | Jikan `/users/{u}/full`; full-parity HTML fallback | `profiles.csv` | ~4 days / 300k users |

Why these sources (measured 2026-07):
- **Jikan for details** — schema matches the old `details.csv` exactly (genres /
  themes / demographics / studios pre-split) and has `favorites`, which the ranker
  uses and MAL v2 lacks. Catalog pages are cache-served, surviving Jikan↔MAL outages.
- **MAL v2 for ratings** — 1000 entries/request, sustained 4.5 req/s without 429s,
  and returns `updated_at` / `start_date` / `finish_date` timestamps (new vs the old
  snapshot; kept when present, often empty on old entries).
- **users.php polling** — measured 240 unique usernames/min with zero duplicates;
  yields *active* accounts by construction.
- **Jikan for profiles, full-parity HTML fallback** — Jikan is primary for IP
  diversification: users.php polling and the ratings crawler already hit MAL from
  our IP, and everything runs in parallel for days; through Jikan, MAL sees
  Jikan's servers instead. But Jikan proxies MAL live and mass-504s during
  outages (observed for hours), so the fallback parses the profile page directly
  — every field is server-rendered there (sidebar, anime-stats block, anime
  favorites), so both paths yield identical columns and `source` records which
  produced each row. A circuit breaker skips Jikan for 5 min after 10
  consecutive failures. HTML stats are cross-checkable against `ratings.csv`
  (`total_entries` matched exactly on every smoke user with a public list);
  a MAL profile redesign would break the regexes → rows come back empty, which
  is immediately visible.

## State & crash-safety

`data/raw/crawl_state.sqlite` (WAL): table `users` tracks per-username
ratings/profile status (`NULL`=pending, `ok`, `http_403` private, `http_404` gone,
`error` — re-queue with `--retry-errors`); table `kv` tracks the details page cursor.

Blacklist: users whose list turns out private (403), deleted (404) or empty are
marked once and never touched again — `crawl_profiles.py` only crawls users with
`ratings_status='ok'` (no ratings ⇒ no training value ⇒ profile request would be
waste). Re-discovery by `collect_usernames.py` can't resurrect them (INSERT OR
IGNORE keeps the existing status row).

Shared rule: **data is appended first, state marked second**. A crash between the
two re-crawls that unit on resume → possible duplicate rows/lines, so cleaning must
dedup (details by `mal_id` keep-last, ratings/profiles by username keep-last).
Inspect progress any time:

```bash
sqlite3 data/raw/crawl_state.sqlite \
  "SELECT ratings_status, profile_status, COUNT(*) FROM users GROUP BY 1,2"
```

## Running a snapshot

```bash
# 1. background for days (nohup/tmux) — feeds the other two:
venv/bin/python data/crawler/collect_usernames.py
# 2. anytime, ~40 min:
venv/bin/python data/crawler/crawl_details.py
# 3+4. in parallel with 1, rerun until pending=0:
venv/bin/python data/crawler/crawl_ratings.py
venv/bin/python data/crawler/crawl_profiles.py
```

Smoke tests: `--iterations 3` / `--pages 2` / `--limit 5`.

## Output schemas

- `details.jsonl.gz` — raw Jikan anime objects, one per line ("crawl wide, clean
  later"); cleaning flattens to the `details.csv` columns.
- `ratings.csv` — `username, anime_id, status, score, num_episodes_watched,
  is_rewatching, updated_at, start_date, finish_date` (old schema + timestamps).
- `profiles.csv` — `username, gender, birthday, location, joined(YYYY-MM-DD),
  favorites_anime` (space-separated mal_ids), `days_watched, mean_score, watching,
  completed, on_hold, dropped, plan_to_watch, total_entries, rewatched,
  episodes_watched, source` (jikan | html).

## After the crawl

`data/cleaning/` gets a new pipeline written against these schemas (the old
notebook is reference only), writing `data/cleaned/` and refreshing
`data/samples/`. Then replace the temporary `data/cleaned/*.csv` symlinks with
real files.
