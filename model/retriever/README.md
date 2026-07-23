# model/retriever/

Stage 1 of the recommender: a **two-tower model** that maps users and anime into a shared 128-d embedding space where cosine similarity means "good match". At serve time, recommending is just a brute-force top-K cosine search of one user vector against all ~22.8k item vectors — cheap, recall-oriented, and meant to hand a candidate shortlist to the ranker (stage 2).

## Architecture

Two independent encoders ("towers") that never share weights, only the output space. Both L2-normalize their output, so a dot product is a cosine.

```
            ITEM TOWER                                 USER TOWER
  ┌─────────────────────────────┐          ┌─────────────────────────────┐
  │ 6 categorical embeddings     │          │ watch history → item vecs    │
  │ (type, source, rating,       │          │ → masked mean-pool           │
  │  demographics, year, eps)    │          │            ⊕                 │
  │            ⊕                 │          │ gender emb ⊕ joined-year emb │
  │ genres (22→8), themes (53→8) │          │                              │
  │ studios (emb, pooled)        │          │                              │
  │ [+ anime-id emb]             │          │                              │
  │            │                 │          │            │                 │
  │       MLP [256] → L2 → V (d) │          │       MLP [256] → L2 → U (d) │
  └─────────────────────────────┘          └─────────────────────────────┘
                 │                                        │
                 └──────────►  cosine(U, V)  ◄────────────┘
```

**Item tower** — content-based, so a brand-new anime with zero interactions still gets a sensible vector:
- 6 categorical features → embeddings: `type`, `source`, `rating`, `demographics`, `start_year` (bucketed), `episodes` (bucketed).
- `genres` (22-dim multi-hot) and `themes` (53-dim multi-hot) → linear projections.
- `studios` (variable-length) → embedding table, masked-mean pooled.
- Optional **anime-id embedding** (a collaborative residual on top of content). Enabled in the shipped model.
- Concatenate everything → MLP `[256]` → 128-d → L2-normalize.

**User tower** — the user is represented mostly by *what they've watched*:
- `history` = the ids of anime the user rated positively. Each id is looked up as an item vector and the set is **mean-pooled** into one 128-d "taste" vector. Empty history (cold user / private list) falls back to a learned `h_empty` vector, so the model still returns popularity-leaning recs.
- Two profile features: `gender` and `joined`-year (bucketed) embeddings.
- Concatenate → MLP `[256]` → 128-d → L2-normalize.

Pooling and the history vector source are configurable (`history_pool` mean/attention, `score_pool`, `history_source` cache/trainable-embedding); the shipped config is **mean-pool over a trainable history embedding table**, with the id embedding on.

## Input → output

| | Shape | Notes |
|---|---|---|
| **Item input** | one row per anime | the content features above, indexed by `anime_idx` (0 = PAD, 1 = OOV, real ≥ 2) |
| **User input** | `history_ids` + `history_mask` (+ scores), `gender`, `joined` | history is a padded ragged list |
| **Output** | one 128-d unit vector per item, one per user | shared space; `score = cosine(U, V)` |
| **Serving** | top-K over the item matrix | mask out already-seen items; hand top-200 to the ranker |

## Training

In-batch **sampled-softmax (InfoNCE)** with a temperature (`tau = 0.07`):

- **Positives**: `status ∈ {completed, watching}` and `score ∉ [1,4]` — i.e. things the user finished/kept watching and didn't pan.
- **In-batch negatives**: every other anchor's positive in the batch. A **logQ correction** subtracts each item's log-frequency so popular titles aren't over-penalized as negatives; false negatives (two anchors sharing the same positive) are masked out.
- **Hard negatives** (`m = 3` per anchor): `dropped ∪ (score ∈ [1,4])` — things the user actively disliked. Added to the denominator, no logQ.

Checkpoint selection metric is **recall@200 on the warm validation set**.

## Cold-start (the point of a content tower)

- **Cold items** (~5% newest anime, held out by `start_date`) never appear in training interactions. They're isolated from train examples, every user's history, hard-negs, and eval support.
- **id-dropout**: during training the anime-id is randomly replaced with OOV, forcing the tower to produce a usable vector from content alone. At eval/serve, cold items are encoded with `id → OOV`, exactly mimicking an anime the model has never seen.
- A **separate cold eval** (`test_cold` = a one-shot final exam) measures how well pure content retrieves new anime.

## Data flow

```
data/cleaned/                data_prep/ 01..06                train-data/            export.py            artifacts/
(details, profiles,   ─────►  re-index ids, encode    ─────►  feature_spec.json,  ─────►  best.pt +   ─────►  item_vectors.npy,
 ratings ~3 GB)               features, split,                item_features,              spec           user_tower.pt,
                              build history + hard-negs,      users, examples,                           item_index.parquet, …
                              compute logQ                    logq
                                    │
                                    ▼  train on Colab (train.ipynb, GPU)
                              checkpoints/best.pt
```

`data_prep/` streams the ~3 GB `ratings.csv` with polars (never loads it whole), re-indexes ids, encodes features, builds the user/item/cold splits, samples history + hard-negs, and computes logQ. `prep_config.py` is the single source of truth for seeds, caps, and label definitions. Training runs on Colab; the winning `best.pt` is exported into `artifacts/`, which the ranker and service **read only** — they never import training code, so serving can't silently drift from what was trained.

## Files

| File | Purpose |
|---|---|
| `data_prep/01..06` + `99_verify.py` | `data/cleaned/` → `train-data/`: re-index ids, encode features, split, history, hard-negs, logQ |
| `data_prep/prep_config.py` | Single source of truth for seeds/caps/label definitions |
| `src/` | Model + training: `config`, `data`, `model`, `loss`, `metrics`, `train`, `search` |
| `export.py` | `checkpoints/best.pt` → `artifacts/` (run whenever best.pt changes) |
| `test_export.py` | Rebuilds the encoder from `artifacts/` and re-runs eval — catches export drift |
| `train.ipynb` | Colab training notebook (needs GPU) |
| `tests/` | Padding/masking invariants; no train-data required |

## Commands

```bash
# prep (streams the ~3 GB ratings.csv with polars — never loads it whole)
for s in model/retriever/data_prep/0[1-6]_*.py; do venv/bin/python "$s"; done
venv/bin/python model/retriever/data_prep/99_verify.py

# local smoke train (subset, MPS) — writes checkpoints/smoke/, never touches best.pt
cd model/retriever/src && ../../../venv/bin/python train.py --smoke

# export + validate (always run as a pair)
venv/bin/python model/retriever/export.py && venv/bin/python model/retriever/test_export.py

venv/bin/python -m pytest model/retriever/tests -q
```

Real training happens on Colab; download the winning `best.pt` into `checkpoints/`, then export.
