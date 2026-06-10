"""05 — Pass-2 (core): history FULL + hard_neg + examples (warm + cold) + eval_seen.

Stream-filter ratings 1 pass: (positive ∪ hard-neg theo prep_config) ∪ (MỌI row của
eval user — nguồn seen-mask). Rồi build eager:
  - Cold-item H (cold_items.parquet, chọn ở 01) cách ly 4 chỗ: train examples,
    history MỌI user, hard_neg, support eval. Positive-H của EVAL user -> cold query
    (examples/split={val,test}_cold); positive-H của TRAIN user vứt hẳn.
  - Warm support/query (chống leak, cơ chế tie-hash như v1): train -> mọi warm
    positive vừa history vừa example (anchor gỡ ở runtime); val/test -> query
    (EVAL_QUERY_FRAC trên warm pool) = example, support (phần còn lại) = history.
    n_warm < 2 -> không có warm query (user vẫn có thể có cold query).
  - history_ids/scores: FULL (sort score desc, tie asc) — KHÔNG cap K; cap nằm ở
    src lúc train/eval (train_hist_len / eval_history_cap).
  - hard_neg_ids: hard-neg − H (dedup, sort, cap HARD_NEG_CAP).
  - eval_seen.parquet: eval user -> unique anime_idx MỌI status (kể cả PTW/on_hold)
    — protocol v2 mask = seen − query_đang_chấm.

In:  cleaned-data/ratings.csv,
     train-data/{anime_id_map,_user_split,_user_feats,cold_items}.parquet
Out: train-data/users.parquet, train-data/eval_seen.parquet,
     train-data/examples/split={train,val,test,val_cold,test_cold}/part-0.parquet

Usage:
    venv/bin/python retriever/data_prep/05_history_examples.py
"""
import pathlib

import polars as pl

from prep_config import EVAL_QUERY_FRAC, HARD_NEG_CAP, SEED, is_hardneg_expr, is_pos_expr

ROOT = pathlib.Path(__file__).parent.parent.parent
SRC = ROOT / "cleaned-data" / "ratings.csv"
OUT = ROOT / "retriever" / "train-data"

SPLIT_CODE = {"train": 0, "val": 1, "test": 2}


def build_base():
    """Streaming 1 pass -> eager frame: (pos ∪ hardneg) ∪ (mọi row của eval user).
    Cột: user_idx, anime_idx, is_pos, is_hardneg, score, split_code."""
    amap = pl.scan_parquet(OUT / "anime_id_map.parquet")          # mal_id, anime_idx
    split = (
        pl.scan_parquet(OUT / "_user_split.parquet")
        .select("username", "user_idx", "split")
        .with_columns(
            pl.col("split").replace_strict(SPLIT_CODE, return_dtype=pl.Int8).alias("split_code")
        )
        .select("username", "user_idx", "split_code")
    )
    base = (
        pl.scan_csv(SRC)
        .with_columns(
            pl.col("score").cast(pl.Int64, strict=False).alias("score"),
            is_pos_expr().alias("is_pos"),
            is_hardneg_expr().alias("is_hardneg"),
        )
        .join(amap, left_on="anime_id", right_on="mal_id", how="inner")
        .join(split, on="username", how="inner")
        .filter(pl.col("is_pos") | pl.col("is_hardneg") | (pl.col("split_code") > 0))
        .select(
            pl.col("user_idx").cast(pl.Int32),
            pl.col("anime_idx").cast(pl.Int32),
            "is_pos", "is_hardneg",
            pl.col("score").cast(pl.Int8),
            "split_code",
        )
        .collect(engine="streaming")
    )
    return base


def main():
    print("Pass-2: streaming (pos ∪ hardneg ∪ eval-user mọi status) ...")
    base = build_base()
    print(f"  base rows: {base.height:,}")

    cold = pl.read_parquet(OUT / "cold_items.parquet")["anime_idx"].to_list()
    print(f"  cold items H: {len(cold):,}")

    # ---- eval_seen: TRƯỚC khi vứt row nào (mọi status của eval user) ----
    eval_seen = (
        base.filter(pl.col("split_code") > 0)
        .group_by("user_idx")
        .agg(pl.col("anime_idx").unique().sort().alias("seen_ids"))
        .sort("user_idx")
    )
    eval_seen.write_parquet(OUT / "eval_seen.parquet")
    print(f"eval_seen.parquet: {eval_seen.height:,} eval users, "
          f"{int(eval_seen.select(pl.col('seen_ids').list.len().sum()).item()):,} seen pairs")

    pos = base.filter("is_pos").select("user_idx", "anime_idx", "score", "split_code")
    hard = base.filter("is_hardneg").select("user_idx", "anime_idx")
    del base

    EX = OUT / "examples"
    is_cold = pl.col("anime_idx").is_in(cold)

    # ---- cold queries: positive-H của eval user; positive-H của train user vứt ----
    cold_total = 0
    for name, code in [("val_cold", 1), ("test_cold", 2)]:
        d = EX / f"split={name}"
        d.mkdir(parents=True, exist_ok=True)
        ex = pos.filter(is_cold & (pl.col("split_code") == code)).select("user_idx", "anime_idx")
        ex.write_parquet(d / "part-0.parquet")
        assert ex.filter(pl.col("anime_idx") < 2).height == 0, f"{name} chứa PAD/OOV"
        nu = ex["user_idx"].n_unique()
        cold_total += ex.height
        print(f"  {name:<10} {ex.height:>9,} pairs / {nu:,} users")
    print(f"  cold pairs tổng: {cold_total:,}  (gate D9: <20k -> nâng COLD_FRAC, re-run 01->06)")
    pos = pos.filter(~is_cold)                       # warm pool (H rớt khỏi train + support)

    # ---- tie hash (reproducible) + warm query selection trên WARM pool ----
    pos = pos.with_columns(
        pl.struct("user_idx", "anime_idx").hash(seed=SEED).alias("tie")
    ).with_columns(
        pl.col("tie").rank("ordinal").over("user_idx").alias("r_tie"),
        pl.len().over("user_idx").alias("n_warm"),
    )
    n_query = pl.min_horizontal(
        pl.max_horizontal((EVAL_QUERY_FRAC * pl.col("n_warm")).round(0), pl.lit(1.0)),
        (pl.col("n_warm") - 1).cast(pl.Float64),     # n_warm<2 -> n_query<=0 -> không có query
    )
    is_eval = pl.col("split_code") > 0
    pos = pos.with_columns((is_eval & (pl.col("r_tie") <= n_query)).alias("is_query"))
    pos = pos.with_columns(
        pl.when(is_eval).then(pl.col("is_query")).otherwise(True).alias("is_example"),
        pl.when(is_eval).then(~pl.col("is_query")).otherwise(True).alias("is_history"),
    ).drop("r_tie", "n_warm", "is_query")  # free RAM trước sort history

    # ---- warm examples: partition theo split ----
    print("examples per split (warm):")
    for name, code in SPLIT_CODE.items():
        d = EX / f"split={name}"
        d.mkdir(parents=True, exist_ok=True)
        ex = pos.filter(pl.col("is_example") & (pl.col("split_code") == code)).select(
            "user_idx", "anime_idx"
        )
        ex.write_parquet(d / "part-0.parquet")
        print(f"  {name:<6} {ex.height:>10,}")
        assert ex.filter(pl.col("anime_idx") < 2).height == 0, "example chứa PAD/OOV"
        assert ex.filter(pl.col("anime_idx").is_in(cold)).height == 0, f"{name} chứa cold item"

    # ---- history: FULL (score desc, tie asc) trên history-eligible — KHÔNG head(K) ----
    hist = (
        pos.filter("is_history")
        .sort(["user_idx", "score", "tie"], descending=[False, True, False])
        .group_by("user_idx", maintain_order=True)
        .agg(
            pl.col("anime_idx").alias("history_ids"),
            pl.col("score").alias("history_scores"),
        )
    )

    # ---- hard_neg: − H, dedup, sort, cap ----
    hard = (
        hard.filter(~is_cold)
        .group_by("user_idx")
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
    hlen = users.select(pl.col("history_ids").list.len().alias("l"))["l"]
    print(f"\nusers.parquet: {users.height:,} rows | history len "
          f"min {hlen.min()} p50 {int(hlen.median())} max {hlen.max()} "
          f"| empty {int((hlen == 0).sum()):,}")

    # ---- verify ----
    chk = users.select(
        (pl.col("history_ids").list.len() == pl.col("history_scores").list.len()).all().alias("aligned"),
    )
    assert chk["aligned"].item(), "history_ids/scores lệch độ dài"

    hist_exp = users.select("user_idx", pl.col("history_ids").alias("aid")).explode("aid")
    assert hist_exp.filter(pl.col("aid").is_in(cold)).height == 0, "LEAK: history chứa cold item!"
    hn_exp = users.select(pl.col("hard_neg_ids").alias("aid")).explode("aid").drop_nulls()
    assert hn_exp.filter(pl.col("aid").is_in(cold)).height == 0, "LEAK: hard_neg chứa cold item!"

    # leak: eval example (warm + cold) ∩ history = ∅
    eval_ex = pl.concat([
        pl.read_parquet(EX / f"split={s}" / "part-0.parquet")
        for s in ["val", "test", "val_cold", "test_cold"]
    ])
    leak = eval_ex.join(hist_exp, left_on=["user_idx", "anime_idx"],
                        right_on=["user_idx", "aid"], how="inner")
    print(f"  leak (eval example ∈ history): {leak.height}")
    assert leak.height == 0, "LEAK: eval example nằm trong history!"

    # seen ⊇ history của eval user (mask v2 phải phủ được mọi thứ user từng chạm)
    seen_exp = eval_seen.explode("seen_ids")
    eval_hist = (
        users.filter(pl.col("split") != "train")
        .select("user_idx", pl.col("history_ids").alias("aid")).explode("aid").drop_nulls()
    )
    not_seen = eval_hist.join(seen_exp, left_on=["user_idx", "aid"],
                              right_on=["user_idx", "seen_ids"], how="anti")
    assert not_seen.height == 0, "eval history có item ngoài seen!"
    print("  seen ⊇ history (eval users) OK")
    print("DONE 05.")


if __name__ == "__main__":
    main()
