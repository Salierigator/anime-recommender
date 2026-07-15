# model/ranker/

Stage 2: LightGBM (lambdarank) reranks the retriever's top-200 candidates. Reading the full
catalog per user would be far too slow for 29 features and a boosted tree — reranking a shortlist
is not.

Reads `artifacts/` + `data/cleaned/{details,profiles}.csv`; writes exactly two files,
`artifacts/ranker.txt` and `artifacts/ranker_meta.json`. Everything the service needs to serve
this model (feature order, blend α, retrieval depth, cold policy) is in the meta file — the
service hardcodes none of it.

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

The `--baseline-only` gate is not optional: it checks that the cosine ordering of the freshly
built pool reproduces the retriever's own numbers. If it fails, the harness has drifted from the
retriever's protocol and any model trained on that pool is measuring the wrong thing.
