"""baseline_linear.py — logistic regression trên numeric features (baseline rẻ, chạy LOCAL).

Mục đích: định vị giá trị của GBDT — nếu LightGBM không thắng nổi linear thì feature set
có vấn đề. Label binary (grade > 0), subsample row để fit nhanh (~1 phút CPU).

Output: ranker/data/models/linear.npz (mean/std/coef/features — eval.py đọc trực tiếp)
        + row.json cùng format leaderboard.

    venv/bin/python ranker/src/baseline_linear.py
"""
from __future__ import annotations

import json
import time

import numpy as np
import torch  # noqa: F401  (convention: torch trước mọi lib OpenMP)
import polars as pl
from sklearn.linear_model import LogisticRegression

import config
from features import CAT_COLS, FEATURE_NAMES
from metrics import load_pool_arrays, sweep_best_alpha

NUM_COLS = [f for f in FEATURE_NAMES if f not in CAT_COLS]
N_SAMPLE = 2_000_000
ALPHAS = [0.25, 0.4, 0.5, 0.6, 0.75, 1.0]
KS = [10, 50, 100, 200]


def main() -> None:
    t0 = time.time()
    rng = np.random.default_rng(config.SEED)
    tr = pl.read_parquet(config.DATASETS / "train.parquet",
                         columns=NUM_COLS + ["label"])
    idx = rng.choice(tr.height, size=min(N_SAMPLE, tr.height), replace=False)
    X = tr.select(NUM_COLS).to_numpy().astype(np.float64)[idx]
    y = (tr["label"].to_numpy()[idx] > 0).astype(np.int8)
    mean, std = np.nanmean(X, 0), np.nanstd(X, 0) + 1e-9
    X = np.where(np.isnan(X), mean, X)
    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit((X - mean) / std, y)
    print(f"fit {len(idx):,} rows, pos_rate {y.mean():.4f} ({time.time() - t0:.0f}s)")

    va_df, cos, lab, off, rt, _ = load_pool_arrays(
        config.POOLS / "eval_val.parquet", config.POOLS / "eval_val_users.parquet", k=200)
    Xv = (va_df.select(NUM_COLS).to_numpy().astype(np.float64) - mean) / std
    pred = Xv @ clf.coef_.ravel() + clf.intercept_[0]
    best_a, best_m, _ = sweep_best_alpha(cos, pred, lab, off, rt, KS, ALPHAS)

    run_dir = config.MODELS / "linear"
    run_dir.mkdir(parents=True, exist_ok=True)
    np.savez(run_dir / "model.npz", mean=mean, std=std, coef=clf.coef_.ravel(),
             intercept=clf.intercept_[0], features=np.asarray(NUM_COLS))
    row = {
        "run": "linear", "model_type": "linear", "n_sample": len(idx),
        "best_alpha": best_a,
        **{f"val_{m}": round(best_m[m], 5)
           for m in ("recall@10", "recall@100", "ndcg@10", "ndcg@100")},
        "train_sec": round(time.time() - t0),
    }
    (run_dir / "row.json").write_text(json.dumps(row, indent=2))
    print(f"[linear] α={best_a} ndcg@10={best_m['ndcg@10']:.4f} "
          f"r@10={best_m['recall@10']:.4f} -> {run_dir / 'model.npz'}")


if __name__ == "__main__":
    main()
