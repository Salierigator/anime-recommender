# model/retriever/

Stage 1: a two-tower model that embeds users and anime into a shared 128-d cosine space, so
serving a candidate list is a brute-force top-K over the ~22.8k-title catalog.

Trained with in-batch sampled-softmax (InfoNCE) + logQ correction + hard negatives. Item features
are content-based (genres, themes, studios, type, source, year, …), which is what lets a brand-new
anime with no interactions still get a sensible vector.

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
