"""05 — Pass-2 (core): history + hard_neg per user + examples table.

Stream-filter ratings xuống positives∪dropped, map id, join split; rồi build eager:
  - support/query (chống leak): train -> mọi positive vừa history vừa example
    (anchor gỡ ở runtime). val/test -> query (random theo tie hash) = example,
    support = history. n_query = clip(round(0.2*n_pos), 1, n_pos-1).
  - history_ids/scores: top-K=30 theo (score desc, tie asc) trên history-eligible,
    2 list song song cùng order.
  - hard_neg_ids: dropped item của user (dedup, sort, cap 64).

In:  cleaned-data/ratings.csv, train-data/{anime_id_map,_user_split,_user_feats}.parquet
Out: train-data/users.parquet, train-data/examples/split={train,val,test}/part-0.parquet

Usage:
    python scripts/build_train_data/05_history_examples.py
"""
import pathlib

import polars as pl

ROOT = pathlib.Path(__file__).parent.parent.parent
SRC = ROOT / "cleaned-data" / "ratings.csv"
OUT = ROOT / "train-data"

SEED = 42
K = 30          # top-k history
HARD_NEG_CAP = 64
SPLIT_CODE = {"train": 0, "val": 1, "test": 2}


def build_base():
    """Streaming filter -> eager frame: user_idx, anime_idx, is_pos, is_dropped, score, split_code."""
    amap = pl.scan_parquet(OUT / "anime_id_map.parquet")          # mal_id, anime_idx
    split = (
        pl.scan_parquet(OUT / "_user_split.parquet")
        .select("username", "user_idx", "split")
        .with_columns(
            pl.col("split").replace_strict(SPLIT_CODE, return_dtype=pl.Int8).alias("split_code")
        )
        .select("username", "user_idx", "split_code")
    )
    score = pl.col("score").cast(pl.Int64, strict=False)
    is_pos = (pl.col("status") == "completed") & ~score.is_between(1, 4)
    is_dropped = pl.col("status") == "dropped"

    base = (
        pl.scan_csv(SRC)
        .with_columns(score.alias("score"))
        .filter(is_pos | is_dropped)
        .with_columns(is_pos.alias("is_pos"), is_dropped.alias("is_dropped"))
        .join(amap, left_on="anime_id", right_on="mal_id", how="inner")
        .join(split, on="username", how="inner")
        .select(
            pl.col("user_idx").cast(pl.Int32),
            pl.col("anime_idx").cast(pl.Int32),
            "is_pos", "is_dropped",
            pl.col("score").cast(pl.Int8),
            "split_code",
        )
        .collect(engine="streaming")
    )
    return base


def main():
    print("Pass-2: streaming filter positives∪dropped ...")
    base = build_base()
    print(f"  base rows (positives + dropped): {base.height:,}")

    pos = base.filter("is_pos").select("user_idx", "anime_idx", "score", "split_code")
    dropped = base.filter("is_dropped").select("user_idx", "anime_idx")
    del base

    # tie hash (reproducible random tie-break + query selection)
    pos = pos.with_columns(
        pl.struct("user_idx", "anime_idx").hash(seed=SEED).alias("tie")
    ).with_columns(
        pl.col("tie").rank("ordinal").over("user_idx").alias("r_tie"),
        pl.len().over("user_idx").alias("n_pos_u"),
    )

    n_query = pl.min_horizontal(
        pl.max_horizontal((0.2 * pl.col("n_pos_u")).round(0), pl.lit(1.0)),
        (pl.col("n_pos_u") - 1).cast(pl.Float64),
    )
    is_eval = pl.col("split_code") > 0
    pos = pos.with_columns((is_eval & (pl.col("r_tie") <= n_query)).alias("is_query"))
    pos = pos.with_columns(
        pl.when(is_eval).then(pl.col("is_query")).otherwise(True).alias("is_example"),
        pl.when(is_eval).then(~pl.col("is_query")).otherwise(True).alias("is_history"),
    ).drop("r_tie", "n_pos_u", "is_query")  # free RAM trước sort history

    # ---- examples: partition theo split ----
    EX = OUT / "examples"
    print("examples per split:")
    for name, code in SPLIT_CODE.items():
        d = EX / f"split={name}"
        d.mkdir(parents=True, exist_ok=True)
        ex = pos.filter(pl.col("is_example") & (pl.col("split_code") == code)).select(
            "user_idx", "anime_idx"
        )
        ex.write_parquet(d / "part-0.parquet")
        print(f"  {name:<6} {ex.height:>10,}")
        assert ex.filter(pl.col("anime_idx") < 2).height == 0, "example chứa PAD/OOV"

    # ---- history: top-K (score desc, tie asc) trên history-eligible ----
    hist = (
        pos.filter("is_history")
        .sort(["user_idx", "score", "tie"], descending=[False, True, False])
        .group_by("user_idx", maintain_order=True)
        .agg(
            pl.col("anime_idx").head(K).alias("history_ids"),
            pl.col("score").head(K).alias("history_scores"),
        )
    )

    # ---- hard_neg: dedup, sort, cap ----
    hard = (
        dropped.group_by("user_idx")
        .agg(pl.col("anime_idx").unique().sort().head(HARD_NEG_CAP).alias("hard_neg_ids"))
    )

    # ---- users.parquet ----
    empty_i32 = pl.lit([], dtype=pl.List(pl.Int32))
    empty_i8 = pl.lit([], dtype=pl.List(pl.Int8))
    users = (
        pl.read_parquet(OUT / "_user_feats.parquet")  # user_idx, split, gender_id, joined_bucket
        .join(hist, on="user_idx", how="left")
        .join(hard, on="user_idx", how="left")
        .with_columns(
            pl.col("history_ids").fill_null(empty_i32),
            pl.col("history_scores").fill_null(empty_i8),
            pl.col("hard_neg_ids").fill_null(empty_i32),
        )
        .sort("user_idx")
    )
    users.write_parquet(OUT / "users.parquet")
    print(f"\nusers.parquet: {users.height:,} rows")

    # ---- verify ----
    hlen = users.select(
        (pl.col("history_ids").list.len() == pl.col("history_scores").list.len()).all().alias("aligned"),
        (pl.col("history_ids").list.len() <= K).all().alias("le_k"),
        (pl.col("history_ids").list.len() == 0).sum().alias("empty_hist"),
    )
    print("  history aligned:", hlen["aligned"].item(), "| <=K:", hlen["le_k"].item(),
          "| empty-history users:", hlen["empty_hist"].item())
    assert hlen["aligned"].item() and hlen["le_k"].item()

    # full leak check: eval history ∩ eval examples = ∅
    eval_ex = pl.concat([
        pl.read_parquet(EX / "split=val" / "part-0.parquet"),
        pl.read_parquet(EX / "split=test" / "part-0.parquet"),
    ])
    hist_exp = users.select("user_idx", pl.col("history_ids").alias("aid")).explode("aid")
    leak = eval_ex.join(hist_exp, left_on=["user_idx", "anime_idx"],
                        right_on=["user_idx", "aid"], how="inner")
    print(f"  leak (eval example ∈ history): {leak.height}")
    assert leak.height == 0, "LEAK: eval example nằm trong history!"
    print("DONE 05.")


if __name__ == "__main__":
    main()
