# data/

Three stages: **crawler → `raw/` → cleaning → `cleaned/`**.

| Path | Committed | What |
|---|---|---|
| `crawler/` | yes | Fetches a fresh MAL snapshot. Not implemented yet — see its README. |
| `raw/` | no | Crawler output. |
| `cleaning/` | yes | `cleaning.ipynb` — the pipeline that produced the current model's data. |
| `cleaned/` | no | Training/serving input: `details.csv`, `profiles.csv`, `ratings.csv`. |
| `samples/` | yes | First rows of each file — the schema reference. |

The CSVs are MAL-derived and gigabytes in size (`ratings.csv` alone is ~3 GB), so nothing under
`raw/` or `cleaned/` is committed. Read schemas from `samples/` instead of opening the real files.

## Cleaning (current data, mid-2025 snapshot)

`cleaning/cleaning.ipynb` took 124.3M ratings / 29.0k anime / 337k profiles down to
**120.0M ratings / 22.8k anime / 292.6k users** (98.20% sparsity). Every rule is quantitative
rather than a hand-picked threshold: drop unknown-status and orphan rows, dedup, four bot rules
(implausible rate counts, near-constant scores, impossible watch volume, mass-add non-raters),
then iterative k-core (user ≥ 10 ratings, anime ≥ 20) until it converges.

⚠️ `cleaning.ipynb` is **reference only** — it is written against the old scrape's schema and its
paths still point at the old layout. A new cleaning pipeline gets written against the crawler's
schema once the 2026 crawl exists, writing to `cleaned/` and refreshing `samples/`.

> **Note:** `cleaned/details.csv` and `cleaned/profiles.csv` are currently symlinks into
> `legacy/` so the service keeps working on the old snapshot. Deleting `legacy/` before the new
> data is ready breaks real mode (mock mode still runs). Replace them with real files after the
> new crawl.
