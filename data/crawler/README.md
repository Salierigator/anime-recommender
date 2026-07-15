# Crawler (TODO — not implemented yet)

Collects a fresh anime + user-rating snapshot from MyAnimeList. The model currently
in `artifacts/` was trained on a mid-2025 snapshot obtained elsewhere; this crawler
will produce its replacement.

## Scope

- **Output** → `data/raw/` (gitignored): anime details, user profiles, user ratings.
  Schema mirrors the three files in `data/samples/` unless a better one is chosen.
- **Source** — to be decided:
  - Jikan v4 — unofficial, no API key, ~1 req/s, exposes `/users/{username}/animelist`.
  - MAL API v2 — official, needs a client id, public animelists only.
  - Scraping — last resort.
- **Hard part**: enumerating enough usernames. The rating table is only as good as
  the user list feeding it.
- **Must have**: rate limiting, resumable progress (crawls span days), and a
  manifest recording when the snapshot was taken.

## After the crawl

`data/cleaning/` gets a new cleaning pipeline written against the raw schema
(the old notebook there is reference only), writing `data/cleaned/`. At that point
replace the temporary `data/cleaned/*.csv` symlinks with real files and refresh
`data/samples/`.
