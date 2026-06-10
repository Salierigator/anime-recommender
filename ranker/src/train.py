"""train.py — LightGBM ranker trên feature matrix, sweep nhiều objective.

Chạy 3 biến thể (cùng group-Dataset + graded label, chỉ đổi objective):
  - lambdarank30  : lambdarank truncation=30  (v1, top-heavy)
  - lambdarank200 : lambdarank truncation=200 (quan tâm cả list)
  - xendcg        : rank_xendcg (cross-entropy NDCG, mượt/robust hơn)
Lưu model trung gian vào ranker/data/ranker_<obj>.txt. eval.py chọn cấu hình tốt nhất (objective ×
blend α) rồi ghi artifacts/ranker.txt + ranker_meta.json.

Firewall: chỉ ghi ranker/data/ (trung gian) — artifacts/ranker.txt do eval.py chốt. Không cần torch.
"""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import pandas as pd

from features import CAT_COLS, FEATURE_NAMES

ROOT = Path(__file__).resolve().parent.parent          # ranker/
DATA = ROOT / "data"

BASE = {
    "metric": "ndcg",
    "ndcg_eval_at": [10, 50, 100],
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "seed": 42,
    "verbosity": -1,
}
OBJECTIVES = {
    "lambdarank30": {"objective": "lambdarank", "lambdarank_truncation_level": 30},
    "lambdarank200": {"objective": "lambdarank", "lambdarank_truncation_level": 200},
    "xendcg": {"objective": "rank_xendcg"},
}


def make_dataset(df: pd.DataFrame, ref: lgb.Dataset | None = None) -> lgb.Dataset:
    group = df.groupby("qid", sort=False).size().to_numpy()      # row cùng qid liền nhau
    return lgb.Dataset(df[FEATURE_NAMES], label=df["label"], group=group,
                       categorical_feature=CAT_COLS, reference=ref, free_raw_data=False)


def main() -> None:
    train = pd.read_parquet(DATA / "train.parquet")
    valid = pd.read_parquet(DATA / "valid.parquet")
    print(f"train {len(train):,} rows / {train['qid'].nunique():,} groups | "
          f"valid {len(valid):,} rows / {valid['qid'].nunique():,} groups")
    dtrain = make_dataset(train)
    dvalid = make_dataset(valid, ref=dtrain)

    best_iters = {}
    for name, override in OBJECTIVES.items():
        print(f"\n=== {name} ===")
        res: dict = {}
        booster = lgb.train(
            BASE | override, dtrain, num_boost_round=1000,
            valid_sets=[dvalid], valid_names=["valid"],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100), lgb.record_evaluation(res)],
        )
        booster.save_model(str(DATA / f"ranker_{name}.txt"), num_iteration=booster.best_iteration)
        best_iters[name] = booster.best_iteration
        print(f"  saved ranker_{name}.txt (best_iter={booster.best_iteration})")
    (DATA / "train_meta.json").write_text(json.dumps({"best_iters": best_iters}, indent=2))
    print(f"\nDONE train. best_iters={best_iters}")


if __name__ == "__main__":
    main()
