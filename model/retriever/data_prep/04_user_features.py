"""04 — User features: gender_id + joined_bucket (cho user tower).

gender: value (Female/Male/Non-Binary) -> 1..3 (sorted); NaN/unseen -> 0 (OOV). vocab 4.
joined: year bucket COHORT_BINS (copy verbatim từ profiles_audit/audit_joined.py);
        5 cohort 0..4. NULL/unparseable -> cohort mới nhất (2022+, id 4) — NULL
        chỉ 0.54% nên gộp vào bucket lớn nhất, bỏ slot NULL đông cứng. vocab 5.

In:  data/cleaned/profiles.csv (pandas), train-data/_user_split.parquet
Out: train-data/_user_feats.parquet (user_idx, split, gender_id, joined_bucket),
     train-data/_spec_user.json (gender/joined vocab+dim+map)

Usage:
    python scripts/build_train_data/04_user_features.py
"""
import json
import pathlib

import numpy as np
import pandas as pd
import polars as pl

RETRIEVER = pathlib.Path(__file__).parent.parent          # model/retriever/
ROOT = RETRIEVER.parent.parent                            # repo root
PROFILES = ROOT / "data" / "cleaned" / "profiles.csv"
OUT = RETRIEVER / "train-data"

COHORT_LABELS = ["<=2012", "2013-16", "2017-19", "2020-21", "2022+"]
COHORT_BINS = [-np.inf, 2012, 2016, 2019, 2021, np.inf]
DIMS = {"gender": 4, "joined": 4}


def main():
    prof = pd.read_csv(PROFILES, usecols=["username", "gender", "joined"])

    gvals = sorted(prof["gender"].dropna().unique())
    gmap = {v: i + 1 for i, v in enumerate(gvals)}  # 0 = OOV
    prof["gender_id"] = prof["gender"].map(
        lambda v: gmap.get(v, 0) if pd.notna(v) else 0).astype(np.int16)

    jyear = pd.to_datetime(prof["joined"], errors="coerce").dt.year
    jcode = pd.cut(jyear, bins=COHORT_BINS, labels=False, right=True)  # 0..4
    # NULL/unparseable -> cohort mới nhất (2022+, id 4): không còn slot NULL riêng
    prof["joined_bucket"] = jcode.fillna(len(COHORT_LABELS) - 1).astype(np.int16)

    feats = pl.from_pandas(prof[["username", "gender_id", "joined_bucket"]])
    split = pl.read_parquet(OUT / "_user_split.parquet").select("username", "user_idx", "split")

    user_feats = (
        split.join(feats, on="username", how="left")
        .with_columns(
            pl.col("gender_id").fill_null(0).cast(pl.Int8),  # 0 = gender OOV
            # join-miss (user không có trong profiles) -> newest cohort, khớp NULL ở line 42
            pl.col("joined_bucket").fill_null(len(COHORT_LABELS) - 1).cast(pl.Int8),
        )
        .select("user_idx", "split", "gender_id", "joined_bucket")
        .sort("user_idx")
    )
    user_feats.write_parquet(OUT / "_user_feats.parquet")

    assert len(gmap) + 1 == 4, f"gender vocab {len(gmap) + 1} != 4"
    print(f"user_feats rows: {user_feats.height:,}")
    print(f"  gender map: {gmap}  (vocab {len(gmap) + 1}, 0=OOV)")
    print("  gender_id dist:\n" + str(user_feats["gender_id"].value_counts().sort("gender_id")))
    print("  joined_bucket dist:\n" + str(user_feats["joined_bucket"].value_counts().sort("joined_bucket")))

    spec_user = {
        "user_features": {
            "gender": {"kind": "cat", "vocab": len(gmap) + 1, "dim": DIMS["gender"],
                       "oov_id": 0, "map": gmap},
            "joined": {"kind": "bucket", "vocab": len(COHORT_LABELS), "dim": DIMS["joined"],
                       "bins": [str(b) for b in COHORT_BINS], "labels": COHORT_LABELS},
        }
    }
    (OUT / "_spec_user.json").write_text(json.dumps(spec_user, ensure_ascii=False, indent=2))
    print("DONE 04.")


if __name__ == "__main__":
    main()
