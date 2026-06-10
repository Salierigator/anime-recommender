"""06 — logQ correction table + final feature_spec.json (v2).

logQ: từ TRAIN examples warm (KHÔNG đụng val/test, chống leak). Q(item)=count/total,
log_q = log(max(count,1)/total) (floor để real item chưa thấy — gồm toàn bộ cold H —
vẫn finite và vẫn là candidate lúc eval). Dense theo anime_idx 0..N+1; PAD/OOV
(idx<2) -> log_q=-inf, is_candidate=False.

feature_spec.json = merge _spec_item + _spec_user + params từ prep_config (echo labels,
cold meta, history_store="full", eval_history_cap default, protocol_version="v2").
Single source of truth để model dựng nn.Embedding + đọc đúng protocol.

In:  train-data/examples/split={train,val_cold,test_cold}/, train-data/{_spec_item,_spec_user}.json,
     train-data/user_id_map.parquet
Out: train-data/logq.parquet (+ logq.npy), train-data/feature_spec.json

Usage:
    venv/bin/python retriever/data_prep/06_logq_and_spec.py
"""
import json
import pathlib

import numpy as np
import polars as pl

from prep_config import (
    EVAL_HISTORY_CAP, EVAL_MIN_POS, EVAL_QUERY_FRAC, HARD_NEG_CAP, LABELS_SPEC, SEED,
)

ROOT = pathlib.Path(__file__).parent.parent.parent
OUT = ROOT / "retriever" / "train-data"


def main():
    spec_item = json.loads((OUT / "_spec_item.json").read_text())
    spec_user = json.loads((OUT / "_spec_user.json").read_text())
    num_items = spec_item["num_items"]
    num_users = pl.read_parquet(OUT / "user_id_map.parquet").height

    # --- logQ từ TRAIN examples (warm — H không bao giờ xuất hiện ở đây) ---
    cnt = (
        pl.scan_parquet(OUT / "examples" / "split=train" / "*.parquet")
        .group_by("anime_idx")
        .agg(pl.len().alias("count"))
        .collect(engine="streaming")
    )
    full = pl.DataFrame({"anime_idx": np.arange(num_items, dtype=np.int32)})
    logq = (
        full.join(cnt, on="anime_idx", how="left")
        .with_columns(pl.col("count").fill_null(0).cast(pl.Int64))
        .sort("anime_idx")
    )
    total = int(logq["count"].sum())
    logq = logq.with_columns(
        (pl.col("anime_idx") >= 2).alias("is_candidate"),
        pl.when(pl.col("anime_idx") < 2)
        .then(pl.lit(float("-inf")))
        .otherwise((pl.max_horizontal("count", pl.lit(1)) / total).log())
        .cast(pl.Float32)
        .alias("log_q"),
    )
    logq.write_parquet(OUT / "logq.parquet")
    np.save(OUT / "logq.npy", logq["log_q"].to_numpy())

    total_check = float(logq.filter(pl.col("count") > 0).select(pl.col("log_q").exp().sum()).item())
    n_zero_real = logq.filter((pl.col("anime_idx") >= 2) & (pl.col("count") == 0)).height
    print(f"logq: total train positives = {total:,}")
    print(f"  real items count==0 (floored, gồm {spec_item['cold_items']['n_cold']} cold): {n_zero_real:,}")
    print(f"  sum(exp(log_q)) over count>0 = {total_check:.8f}  (target 1.0)")
    assert abs(total_check - 1.0) < 1e-6

    # --- merge feature_spec.json (v2) ---
    cold_examples = {
        name: pl.read_parquet(OUT / "examples" / f"split={name}" / "part-0.parquet").height
        for name in ["val_cold", "test_cold"]
    }
    spec = {
        "protocol_version": "v2",
        "seed": SEED,
        "labels": LABELS_SPEC,
        "history_store": "full",
        "eval_history_cap_default": EVAL_HISTORY_CAP,
        "hard_neg_cap": HARD_NEG_CAP,
        "eval_min_positives": EVAL_MIN_POS,
        "eval_query_frac": EVAL_QUERY_FRAC,
        "split_ratios": {"train": 0.90, "val": 0.05, "test": 0.05},
        "num_users": num_users,
        "num_items": num_items,
        "cold_items": spec_item["cold_items"],
        "cold_examples": cold_examples,
        "special_idx": spec_item["special_idx"],
        "item_features": spec_item["item_features"],
        "user_features": spec_user["user_features"],
    }
    (OUT / "feature_spec.json").write_text(json.dumps(spec, ensure_ascii=False, indent=2))
    print(f"\nfeature_spec.json (v2): num_users={num_users:,}  num_items={num_items:,}")
    print(f"  labels: {LABELS_SPEC['positive']} | hard-neg: {LABELS_SPEC['hard_negative']}")
    print(f"  cold: {spec['cold_items']['n_cold']} items, examples {cold_examples}")
    print("DONE 06.")


if __name__ == "__main__":
    main()
