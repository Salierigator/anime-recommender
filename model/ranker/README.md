# model/ranker/

Stage 2 of the recommender: a **LightGBM `lambdarank`** model that reranks the retriever's top-200 candidates per user. The retriever is fast but coarse (a single cosine over content+collaborative embeddings); the ranker is a gradient-boosted tree that fuses **29 hand-built features** — cosine, history similarity, genre/theme affinity, and MAL metadata — into a sharper ordering. A tree model is the right tool here: the features are heterogeneous tabular signals on very different scales, several are missing for some items, and boosted trees handle both natively (no scaling, native categorical splits, NaN-aware) while staying cheap on a 200-item shortlist.

```
retriever top-200 (warm)  ──►  29 features per (user, candidate)  ──►  LightGBM  ──►  score
                                                                              │
                                        final order = blend( α·rank_norm(score) + (1-α)·cosine )
```

## Input: the candidate pool

Candidates are produced by exactly the same code path at train, eval, and serve time (`pool.py`), so the score distribution the model trains on matches what it sees in production:

1. Encode the user vector `U` from their watch history (reusing the retriever's `UserEncoder` from `artifacts/`).
2. `cosine(U, every item)`, mask out PAD/OOV and already-seen items.
3. Take the **top-200 warm items** as the pool (cold items are filtered out here — see below).

Each candidate then becomes one row of 29 features; all 200 rows for a user form one `lambdarank` group.

## The 29 features

Feature order is fixed in `FEATURE_NAMES` and frozen into `ranker_meta.json`, so the service builds the exact same matrix. Four groups:

| Group | Features | Built from |
|---|---|---|
| **Cross** (user × item, 9) | `cos_uv`, `pool_rank`, `hist_cos_max`, `hist_cos_mean`, `hist_cos_top5_mean`, `genre_aff`, `theme_aff`, `genre_overlap`, `score_gap` | computed per (user, candidate) at pool time |
| **User** (5) | `u_n_rated`, `u_mean_score`, `u_std_score`, `u_account_age`, `support_len` | user's rating history + profile |
| **Item numeric** (10) | `mal_score` (+`mal_score_missing`), `log_scored_by`, `log_members`, `log_favorites`, `popularity`, `rank` (+`rank_missing`), `episodes`, `recency_years` | `details.csv`, gathered by `anime_idx` |
| **Item categorical** (5) | `type_code`, `source_code`, `rating_code`, `demo_code`, `era_code` | `details.csv`, native LightGBM categorical |

## Feature processing (the interesting part)

Everything is **deterministic from `details.csv`** — same input, same code, so train / eval / serve never drift.

**Item features** (`ItemFeatures.load`, one row per `anime_idx`, aligned with `item_vectors`):
- **Categorical → int codes**: each string column is sorted and mapped to a stable integer (`NaN → 0`), then handed to LightGBM as a *native categorical* — no one-hot, the tree splits on category subsets directly. `era_code` buckets `start_year` into eras (≤1989, 90s, 00s, 2010–17, 2018+).
- **Heavy-tailed counts → `log1p`**: `scored_by`, `members`, `favorites` span several orders of magnitude, so they're log-compressed.
- **Median-impute + missing flag**: `mal_score` and `rank` are median-filled, but paired with a `*_missing` boolean so the tree can tell "average" from "unknown" rather than being fooled by the fill value.
- **`recency_years`**: `REF_YEAR (2024) − start_year`.

**Cross features** (`pool.cross_features`, computed per (user, candidate)):
- `cos_uv` = retriever cosine; `pool_rank` = the candidate's absolute cosine rank (so item features stay invariant to pool depth).
- **History similarity**: cosine of the candidate against the user's top-256 history vectors, summarized as `max`, `mean`, and `top-5 mean` — "how close is this to the things you already love".
- **Genre / theme affinity**: the user's history is averaged into a multi-hot preference vector; `genre_aff` / `theme_aff` are the candidate's dot product with it, `genre_overlap` counts shared genres.
- `score_gap` = item's `mal_score` − user's mean score.

**User features** (`user_stats_from_support`): counts and score stats over the user's rated support. `u_account_age` is intentionally left as `NaN` when the join date is unknown — LightGBM handles it natively; do **not** impute it.

**Cold policy (no-leak)** — this is why cold items are handled specially. A newly released anime has genres/type/episodes from day one, but no accumulated `mal_score` / `scored_by` / `members` / `favorites` / `popularity` / `rank`. So for any `is_cold` item those maturity stats are **wiped to missing before imputation** (and median is computed on warm items only). Content and recency features are kept. This happens inside `ItemFeatures.load`, so every downstream path is consistent.

## Labels & training

Positives are graded from the user's raw score into relevance levels for `lambdarank`:

```
grade:  10 → 4,   9 → 3,   7–8 → 2,   {0,5,6} → 1,   non-target → 0
```

- A fraction (`TARGET_FRAC = 0.2`) of each user's positives is held out as the ranking *targets*; the rest stays in history. Group size is fixed at 200.
- **Train pool excludes cold items** (so the model never learns to bury them); the eval pool keeps them (to mirror the retriever's full-catalog recall).
- Selection isn't LightGBM's internal nDCG — it's the two-stage `metrics.py` (mirroring the retriever). The winner is chosen by a **Pareto test vs. the cosine baseline** on `{recall, nDCG}@{10,100}`, after a blend-`α` sweep (`α=0` = pure cosine, `α=1` = pure ranker).

## Cold serving (separate channel)

Cold items are **not** rescored by the ranker. The `α=1` ranker, trained on warm pools, pushes cold items to the bottom — so at serve time cold items are returned as their **own section ordered by retriever cosine**. This split (verified against two alternatives) keeps both warm ranking and cold discovery near-optimal. The policy lives in `ranker_meta.json::cold_serving`; the service reads it, hardcodes nothing.

## Output

Exactly two files, written by `export.py`: `artifacts/ranker.txt` (the model) and `artifacts/ranker_meta.json` (feature order, blend α, retrieval depth, cold policy). The service reads these and needs no code change across retrains.

## Files

| File | Purpose |
|---|---|
| `src/config.py` | Paths + knobs (seed, pool depth, K, …) and label grading |
| `src/user_encode.py` | Rebuilds the user encoder from `artifacts/` — **shared with the service** |
| `src/features.py` | The 29 `FEATURE_NAMES`, item feature table, cold policy, frame building |
| `src/pool.py` | Ragged history, user encoding, top-K masking, cross features |
| `src/metrics.py` | Two-stage recall/nDCG (mirrors the retriever's) + blend — pure numpy |
| `src/train_lgbm.py` | Local sweep → `models/<run>/` + leaderboard |
| `data_prep/build_eval.py` / `build_train.py` | Eval pools / training dataset from `artifacts/` |
| `eval.py` | Sanity gate, blend sweep, Pareto selection over cosine |
| `export.py` | Winner → `artifacts/ranker.txt` + `ranker_meta.json` |

## Commands

Run from the repo root, in this order:

```bash
venv/bin/python model/ranker/data_prep/build_eval.py
venv/bin/python model/ranker/eval.py --baseline-only    # gate: cosine must reproduce eval_reference.json
venv/bin/python model/ranker/data_prep/build_train.py
venv/bin/python model/ranker/src/train_lgbm.py          # ~40 min local, 4 threads (--smoke for a slice)
venv/bin/python model/ranker/eval.py                    # select winner → models/eval_selection.json
venv/bin/python model/ranker/export.py
venv/bin/python -m pytest model/ranker/tests -q
```

The `--baseline-only` gate is not optional: it checks that the cosine ordering of the freshly built pool reproduces the retriever's own numbers. If it fails, the harness has drifted from the retriever's protocol and any model trained on that pool is measuring the wrong thing.
