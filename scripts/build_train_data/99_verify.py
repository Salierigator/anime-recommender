"""99 — Verify toàn bộ artifacts train-data/ (chỉ đọc train-data, KHÔNG đụng cleaned-data).

Check: schema/dtype, id ∈ [0,vocab) khớp feature_spec, multihot width, examples
anime_idx>=2 & user_idx hợp lệ, history aligned, logq length & candidate. Raise nếu sai.

Usage:
    python scripts/build_train_data/99_verify.py
"""
import json
import pathlib

import polars as pl

ROOT = pathlib.Path(__file__).parent.parent.parent
OUT = ROOT / "train-data"


def main():
    spec = json.loads((OUT / "feature_spec.json").read_text())
    num_items, num_users = spec["num_items"], spec["num_users"]
    ok = []

    # ---- item_features ----
    items = pl.read_parquet(OUT / "item_features.parquet")
    assert items.height == num_items, f"item rows {items.height} != {num_items}"
    scalar = {"type": "type_id", "source": "source_id", "rating": "rating_id",
              "demographics": "demographics_id", "start_year": "startyear_bucket",
              "episodes": "episodes_bucket"}
    for feat, col in scalar.items():
        vocab = spec["item_features"][feat]["vocab"]
        lo, hi = items[col].min(), items[col].max()
        assert lo >= 0 and hi < vocab, f"{col}: [{lo},{hi}] ngoài [0,{vocab})"
    for feat, col, w in [("genres", "genres_multihot", 22), ("themes", "themes_multihot", 53)]:
        assert spec["item_features"][feat]["width"] == w
        lens = items.select(pl.col(col).list.len().alias("l"))["l"]
        assert lens.min() == w and lens.max() == w, f"{col} width != {w}"
        mx = items.select(pl.col(col).list.max().alias("m"))["m"].max()
        assert mx <= 1, f"{col} không phải 0/1 (max {mx})"
    svocab = spec["item_features"]["studios"]["vocab"]
    smax = items.select(pl.col("studio_ids").list.max().alias("m"))["m"].max()
    smin = items.select(pl.col("studio_ids").list.min().alias("m"))["m"].min()
    assert 0 <= smin and smax < svocab, f"studio id ngoài [0,{svocab})"
    ok.append(f"item_features: {items.height:,} rows, mọi id ∈ [0,vocab), multihot 22/53 OK")

    # ---- users ----
    users = pl.read_parquet(OUT / "users.parquet")
    assert users.height == num_users
    chk = users.select(
        (pl.col("history_ids").list.len() == pl.col("history_scores").list.len()).all().alias("a"),
        (pl.col("history_ids").list.len() <= spec["k_history"]).all().alias("k"),
        pl.col("gender_id").max().alias("gmax"), pl.col("joined_bucket").max().alias("jmax"),
        pl.col("hard_neg_ids").list.len().max().alias("hncap"),
    )
    assert chk["a"].item() and chk["k"].item()
    assert chk["gmax"].item() < spec["user_features"]["gender"]["vocab"]
    assert chk["jmax"].item() < spec["user_features"]["joined"]["vocab"]
    assert chk["hncap"].item() <= spec["hard_neg_cap"]
    ok.append(f"users: {users.height:,} rows, history aligned & <=K, gender/joined/hardneg in-range OK")

    # ---- examples ----
    total_ex = 0
    for name in ["train", "val", "test"]:
        ex = pl.read_parquet(OUT / "examples" / f"split={name}" / "part-0.parquet")
        assert ex.filter(pl.col("anime_idx") < 2).height == 0, f"{name}: PAD/OOV làm target"
        assert ex.filter((pl.col("user_idx") < 0) | (pl.col("user_idx") >= num_users)).height == 0
        total_ex += ex.height
    ok.append(f"examples: {total_ex:,} rows tổng, anime_idx>=2 & user_idx hợp lệ OK")

    # ---- logq ----
    logq = pl.read_parquet(OUT / "logq.parquet")
    assert logq.height == num_items
    assert logq.filter(~pl.col("is_candidate")).select((pl.col("anime_idx") < 2).all()).item()
    real = logq.filter(pl.col("anime_idx") >= 2)
    assert real.select(pl.col("log_q").is_finite().all()).item(), "real item có log_q không finite"
    s = float(logq.filter(pl.col("count") > 0).select(pl.col("log_q").exp().sum()).item())
    assert abs(s - 1.0) < 1e-6
    ok.append(f"logq: {logq.height:,} rows, real finite, sum(exp)=1.0 OK")

    print("=" * 64)
    for line in ok:
        print("  [OK] " + line)
    print("=" * 64)
    print("ALL CHECKS PASSED.")


if __name__ == "__main__":
    main()
