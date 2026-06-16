"""train_lgbm.py — train LightGBM ranker (Colab notebook import; --smoke local).

Path EXPLICIT (data_dir/out_dir) để chạy y hệt trên Colab (/content/ranker_data) lẫn local
(ranker/train-data). Train = datasets/train.parquet (label graded 0-4, group = user, top-200 pool);
valid early-stopping = pools/eval_val.parquet slice 200 (label binary — LightGBM ndcg nội bộ
CHỈ để early-stop; số chính thức = two-stage metrics.py trên eval_val, tính sau mỗi run).

Fix so với ranker cũ: num_boost_round 4000 + early_stopping(100) (cũ cap 1000 chưa hội tụ).

    venv/bin/python ranker/src/train_lgbm.py --smoke     # slice nhỏ local, kiểm tra end-to-end
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch  # noqa: F401  (TRƯỚC lightgbm — segfault 2 OpenMP runtime trên mac)
import lightgbm as lgb
import polars as pl

import config
from features import CAT_COLS, FEATURE_NAMES
from metrics import load_pool_arrays, sweep_best_alpha

BASE = dict(
    objective="lambdarank",
    boosting_type="gbdt",
    learning_rate=0.05,
    num_leaves=63,
    min_data_in_leaf=100,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    metric="ndcg",
    ndcg_eval_at=[10, 100],
    verbosity=-1,
    seed=42,
    num_threads=config.NUM_THREADS,   # giảm thread → train local đỡ nóng máy
)
NUM_ROUNDS = 4000
EARLY_STOP = 100
ALPHAS = [0.25, 0.4, 0.5, 0.6, 0.75, 1.0]
KS = [10, 50, 100, 200]


def _groups(qid: np.ndarray) -> np.ndarray:
    _, counts = np.unique(qid, return_counts=True)
    return counts


def load_datasets(data_dir: Path, max_groups: int | None = None):
    """(train Dataset, valid Dataset, valid arrays cho two-stage). data_dir chứa
    datasets/train.parquet + pools/eval_val{,_users}.parquet."""
    tr = pl.read_parquet(data_dir / "datasets" / "train.parquet")
    if max_groups is not None:
        tr = tr.filter(pl.col("qid") < max_groups)
    dtrain = lgb.Dataset(
        tr.select(FEATURE_NAMES).to_pandas(), label=tr["label"].to_numpy(),
        group=_groups(tr["qid"].to_numpy()),
        feature_name=FEATURE_NAMES, categorical_feature=CAT_COLS, free_raw_data=True)

    va_df, va_cos, va_lab, va_off, va_rt, _ = load_pool_arrays(
        data_dir / "pools" / "eval_val.parquet",
        data_dir / "pools" / "eval_val_users.parquet", k=200, max_groups=max_groups)
    X_val = va_df.select(FEATURE_NAMES).to_pandas()
    dvalid = lgb.Dataset(X_val, label=va_lab, group=np.diff(va_off),
                         feature_name=FEATURE_NAMES, categorical_feature=CAT_COLS,
                         reference=dtrain, free_raw_data=False)
    return dtrain, dvalid, (X_val, va_cos, va_lab, va_off, va_rt)


def train_one(run_name: str, out_dir: Path, dtrain, dvalid, valid_arrays,
              **overrides) -> dict:
    """Train 1 config → save model.txt + row.json (leaderboard = two-stage val @ best α)."""
    params = {**BASE, **overrides}
    t0 = time.time()
    booster = lgb.train(
        params, dtrain, num_boost_round=NUM_ROUNDS, valid_sets=[dvalid],
        callbacks=[lgb.early_stopping(EARLY_STOP, verbose=True), lgb.log_evaluation(200)])

    X_val, cos, lab, off, rt = valid_arrays
    pred = booster.predict(X_val, num_iteration=booster.best_iteration)
    best_a, best_m, _ = sweep_best_alpha(cos, pred, lab, off, rt, KS, ALPHAS)

    run_dir = out_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(run_dir / "model.txt"), num_iteration=booster.best_iteration)
    row = {
        "run": run_name, "model_type": "lightgbm",
        "objective": params["objective"],
        "truncation": params.get("lambdarank_truncation_level"),
        "learning_rate": params["learning_rate"], "num_leaves": params["num_leaves"],
        "best_iteration": booster.best_iteration,
        "best_alpha": best_a,
        **{f"val_{m}": round(best_m[m], 5)
           for m in ("recall@10", "recall@100", "ndcg@10", "ndcg@100")},
        "train_sec": round(time.time() - t0),
        "importance_gain": dict(zip(FEATURE_NAMES,
                                    booster.feature_importance("gain").round(1).tolist())),
    }
    (run_dir / "row.json").write_text(json.dumps(row, indent=2))
    print(f"[{run_name}] iter={row['best_iteration']} α={best_a} "
          f"ndcg@10={best_m['ndcg@10']:.4f} r@10={best_m['recall@10']:.4f} "
          f"({row['train_sec']}s)")
    return row


# Sweep grid LightGBM (mirror notebook cell 4 cũ): trục objective × lr × leaves.
SWEEP = [
    ("xendcg_lr05_l63",  dict(objective="rank_xendcg")),                       # winner CHỐT
    ("lrank10_lr05_l63",  dict(lambdarank_truncation_level=10)),
    ("lrank30_lr05_l63",  dict(lambdarank_truncation_level=30)),
    ("lrank200_lr05_l63", dict(lambdarank_truncation_level=200)),
    ("xendcg_lr10_l63",   dict(objective="rank_xendcg", learning_rate=0.1)),
    ("xendcg_lr05_l127",  dict(objective="rank_xendcg", num_leaves=127)),
]


def leaderboard(models_dir: Path) -> None:
    """Gom models/*/row.json → in bảng + ghi models/leaderboard.csv (sort val_ndcg@10)."""
    import pandas as pd

    rows = [json.loads(p.read_text()) for p in sorted(models_dir.glob("*/row.json"))]
    if not rows:
        return
    df = pd.DataFrame(rows).drop(columns=["importance_gain"], errors="ignore") \
        .sort_values("val_ndcg@10", ascending=False)
    df.to_csv(models_dir / "leaderboard.csv", index=False)
    print("\n=== leaderboard (val_ndcg@10) ===")
    print(df.to_string(index=False))


def main() -> None:
    """Sweep grid LightGBM LOCAL → models/<run>/{model.txt,row.json} + leaderboard.csv.
    --smoke: slice 3k + 1 config (sanity end-to-end, KHÔNG phải số thật)."""
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    max_groups = 3_000 if args.smoke else None

    dtrain, dvalid, valid_arrays = load_datasets(config.DATA, max_groups=max_groups)
    global NUM_ROUNDS
    if args.smoke:
        NUM_ROUNDS = 50
        sweep = [("smoke_xendcg", dict(objective="rank_xendcg"))]
    else:
        sweep = SWEEP

    for run_name, overrides in sweep:
        train_one(run_name, config.MODELS, dtrain, dvalid, valid_arrays, **overrides)
    leaderboard(config.MODELS)


if __name__ == "__main__":
    main()
