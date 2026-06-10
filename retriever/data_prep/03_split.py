"""03 — Cold-by-user split (90/5/5) + user_idx re-index.

Hold out trọn user. Eval (val/test) chỉ nhận user có n_pos >= 11 (đủ chia
support/query ở script 05); user 1..10 positive luôn về train. Drop n_pos==0.
Split tất định bằng stable-hash(username, SEED) (reproducible, không RNG).

In:  train-data/_user_stats.parquet
Out: train-data/_user_split.parquet (username, user_idx, split, n_pos, n_dropped),
     train-data/user_id_map.parquet (username, user_idx)

Usage:
    python scripts/build_train_data/03_split.py
"""
import pathlib

import polars as pl

ROOT = pathlib.Path(__file__).parent.parent.parent
OUT = ROOT / "retriever" / "train-data"

SEED = 42
EVAL_MIN_POS = 11  # đủ support + query


def main():
    stats = pl.read_parquet(OUT / "_user_stats.parquet")
    kept = stats.filter(pl.col("n_pos") >= 1)

    bucket = pl.col("username").hash(seed=SEED) % 100
    eligible = pl.col("n_pos") >= EVAL_MIN_POS
    split = (
        pl.when(~eligible).then(pl.lit("train"))      # 1..10 pos -> train
        .when(bucket < 90).then(pl.lit("train"))
        .when(bucket < 95).then(pl.lit("val"))
        .otherwise(pl.lit("test"))
    )

    kept = (
        kept.with_columns(split.alias("split"))
        .sort("username")
        .with_row_index("user_idx")
        .with_columns(pl.col("user_idx").cast(pl.Int32))
    )
    kept.write_parquet(OUT / "_user_split.parquet")
    kept.select("username", "user_idx").write_parquet(OUT / "user_id_map.parquet")

    n = kept.height
    print(f"kept users (n_pos>=1): {n:,}")
    for s in ["train", "val", "test"]:
        c = kept.filter(pl.col("split") == s).height
        print(f"  {s:<6} {c:>9,}  ({c / n * 100:5.2f}%)")
    # assert: mọi eval user >= EVAL_MIN_POS
    bad = kept.filter((pl.col("split") != "train") & (pl.col("n_pos") < EVAL_MIN_POS)).height
    assert bad == 0, f"{bad} eval users có n_pos < {EVAL_MIN_POS}"
    print(f"  eval-eligibility OK (mọi val/test user >= {EVAL_MIN_POS} positive)")
    print("DONE 03.")


if __name__ == "__main__":
    main()
