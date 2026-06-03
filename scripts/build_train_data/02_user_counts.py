"""02 — Pass-1: per-user positive/dropped counts (streaming over ratings.csv).

positive = status=="completed" & score ∉ [1,4]  (giữ score==0 và 5..10).
hard-neg = status=="dropped".
Output _user_stats.parquet phục vụ chia split (script 03). ratings.csv ~3.2GB
-> polars lazy + streaming, KHÔNG collect full.

Usage:
    python scripts/build_train_data/02_user_counts.py
"""
import pathlib

import polars as pl

ROOT = pathlib.Path(__file__).parent.parent.parent
SRC = ROOT / "cleaned-data" / "ratings.csv"
OUT = ROOT / "train-data"
OUT.mkdir(exist_ok=True)


def main():
    score = pl.col("score").cast(pl.Int64, strict=False)
    is_pos = (pl.col("status") == "completed") & ~score.is_between(1, 4)
    is_dropped = pl.col("status") == "dropped"

    stats = (
        pl.scan_csv(SRC)
        .group_by("username")
        .agg(
            is_pos.sum().alias("n_pos"),
            is_dropped.sum().alias("n_dropped"),
        )
        .collect(engine="streaming")
        .with_columns(pl.col("n_pos").cast(pl.Int32), pl.col("n_dropped").cast(pl.Int32))
    )
    stats.write_parquet(OUT / "_user_stats.parquet")

    n = stats.height
    n_zero = stats.filter(pl.col("n_pos") == 0).height
    n_eval = stats.filter(pl.col("n_pos") >= 11).height
    print(f"users total: {n:,}")
    print(f"  n_pos == 0  (drop khỏi train):    {n_zero:>9,}  ({n_zero / n * 100:5.2f}%)")
    print(f"  n_pos >= 1  (keep):               {n - n_zero:>9,}")
    print(f"  n_pos >= 11 (eval-eligible):      {n_eval:>9,}  ({n_eval / n * 100:5.2f}%)")
    print(f"  total positives: {int(stats['n_pos'].sum()):,}")
    print(f"  total dropped:   {int(stats['n_dropped'].sum()):,}")
    print("DONE 02.")


if __name__ == "__main__":
    main()
