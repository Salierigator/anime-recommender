"""build_dataset.py — cleaned-data + artifacts → ma trận feature ranker (parquet, group theo user).

Luồng (xem docs/RANKER.md §4-6):
  1. chọn user (train + val từ user_split), stream ratings → positives(`completed & score∉[1,4]`)+dropped.
  2. per user: split profile/target chống leak; history top-30 (score desc, tie asc); dựng U.
  3. candidate = target(label graded) ∪ hard-neg retriever ∪ random-neg; tính feature (§6).
  4. ghi train.parquet (train user) + valid.parquet (val user, cho early-stopping).

Firewall: ĐỌC artifacts/ + cleaned-data/, import định nghĩa model. ratings.csv ~3.2GB → polars
streaming, filter user TRƯỚC. KHÔNG đọc retriever/train-data. CWD chạy = ranker/src.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import torch

from features import ItemFeatures, build_frame, FEATURE_NAMES
from user_encode import ARTIFACTS, ROOT, encode_gender_joined, load_user_encoder

CLEANED = ROOT.parent / "cleaned-data"
OUT = ROOT / "data"
SEED = 42
K_HISTORY = 30
N_TRAIN_USERS = 100_000
N_VAL_USERS = 12_000
TARGET_MAX = 8            # tối đa target (label) mỗi user
HARD_NEG = 40            # hard-neg retriever mỗi user
HARD_SKIP = 5            # bỏ top-K hard nhất (dễ là false-neg: item retriever thích, user chưa xem)
RANDOM_NEG = 40          # random-neg mỗi user
CHUNK = 2_000            # số user mỗi chunk encode/score


def grade(score: int) -> int:
    """relevance graded theo score. Phủ TẤT CẢ positive (khớp 'relevant' của eval = completed &
    score∉[1,4]): score 0/5/6 → 1 (relevant thấp nhất), 7-8 → 2, 9 → 3, 10 → 4."""
    if score >= 10:
        return 4
    if score >= 9:
        return 3
    if score >= 7:
        return 2
    return 1


def select_users(rng) -> tuple[list[str], list[str]]:
    split = pl.read_parquet(ARTIFACTS / "user_split.parquet")
    tr = split.filter(pl.col("split") == "train")["username"].to_list()
    va = split.filter(pl.col("split") == "val")["username"].to_list()
    rng.shuffle(tr)
    rng.shuffle(va)
    return tr[:N_TRAIN_USERS], va[:N_VAL_USERS]


def stream_ratings(usernames: list[str]) -> pd.DataFrame:
    """Stream ratings.csv (filter user trước), giữ positives∪dropped, map anime_id→anime_idx."""
    amap = pl.scan_parquet(ARTIFACTS / "item_index.parquet")           # anime_idx, mal_id
    score = pl.col("score").cast(pl.Int64, strict=False)
    is_pos = (pl.col("status") == "completed") & ~score.is_between(1, 4)
    is_dropped = pl.col("status") == "dropped"
    base = (
        pl.scan_csv(CLEANED / "ratings.csv")
        .filter(pl.col("username").is_in(usernames))
        .with_columns(score.alias("score"))
        .filter(is_pos | is_dropped)
        .with_columns(is_pos.alias("is_pos"), is_dropped.alias("is_dropped"))
        .join(amap, left_on="anime_id", right_on="mal_id", how="inner")
        .select("username", pl.col("anime_idx").cast(pl.Int32),
                "is_pos", "is_dropped", pl.col("score").cast(pl.Int16))
        .collect(engine="streaming")
    )
    return base.to_pandas()


def account_age(usernames: list[str]) -> dict[str, float]:
    prof = pd.read_csv(CLEANED / "profiles.csv", usecols=["username", "joined"])
    prof = prof[prof["username"].isin(set(usernames))]
    yr = pd.to_datetime(prof["joined"], errors="coerce").dt.year
    age = (2024 - yr)
    return dict(zip(prof["username"], age))


def main() -> None:
    t0 = time.time()
    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    enc, meta = load_user_encoder("cpu")
    V = enc.item_cache.numpy()                                # [N, 128] (row==anime_idx)
    N = V.shape[0]
    itemfeat = ItemFeatures.load(ARTIFACTS, CLEANED)
    G, T = itemfeat.genres.shape[1], itemfeat.themes.shape[1]

    train_users, val_users = select_users(rng)
    all_users = train_users + val_users
    is_train = np.array([True] * len(train_users) + [False] * len(val_users))
    print(f"users: train={len(train_users):,} val={len(val_users):,}")

    print("streaming ratings.csv ...")
    base = stream_ratings(all_users)
    print(f"  base rows (positives+dropped): {len(base):,}  ({time.time()-t0:.0f}s)")
    age_map = account_age(all_users)
    guf = encode_gender_joined(
        pd.read_csv(CLEANED / "profiles.csv", usecols=["username", "gender", "joined"]), meta)
    guf = guf.set_index("username")

    # group per user
    base_g = {u: g for u, g in base.groupby("username", sort=False)}

    # ---- Pass A: history/target/pref per user (chống leak) ----
    rows = []           # mỗi phần tử = dict thông tin user hợp lệ
    for u, is_tr in zip(all_users, is_train):
        g = base_g.get(u)
        if g is None:
            continue
        pos = g[g["is_pos"].to_numpy()]
        dropped_idx = g.loc[g["is_dropped"].to_numpy(), "anime_idx"].to_numpy()
        p_idx = pos["anime_idx"].to_numpy()
        p_score = pos["score"].to_numpy()
        if len(p_idx) < 2:
            continue
        tie = rng.random(len(p_idx))                          # tie-break + holdout reproducible
        # target = held-out positives (TẤT CẢ, khớp 'relevant' của eval), graded theo score.
        # Chọn ngẫu nhiên theo tie — cùng cơ chế support/query split của eval (train↔eval khớp).
        n_target = int(min(TARGET_MAX, len(p_idx) - 1))
        tgt_local = np.argsort(tie)[:n_target]
        is_tgt = np.zeros(len(p_idx), bool)
        is_tgt[tgt_local] = True
        # history = positives ngoài target, top-K theo (score desc, tie asc)
        h_local = np.where(~is_tgt)[0]
        order = sorted(h_local, key=lambda i: (-p_score[i], tie[i]))[:K_HISTORY]
        hist_idx = p_idx[order]
        hist_score = p_score[order]
        if len(hist_idx) == 0:
            continue
        rated = hist_score[hist_score >= 1]
        rows.append({
            "u": u, "is_train": bool(is_tr),
            "hist_idx": hist_idx, "hist_score": hist_score,
            "tgt_idx": p_idx[tgt_local], "tgt_score": p_score[tgt_local],
            "seen": np.unique(np.concatenate([p_idx, dropped_idx])),
            "g_pref": itemfeat.genres[hist_idx].mean(0), "t_pref": itemfeat.themes[hist_idx].mean(0),
            "gender_id": int(guf.loc[u, "gender_id"]) if u in guf.index else 0,
            "joined_bucket": int(guf.loc[u, "joined_bucket"]) if u in guf.index else len(meta["user_features"]["joined"]["bins"]) - 2,
            "u_n_rated": float(len(rated)),
            "u_mean_score": float(rated.mean()) if len(rated) else 0.0,
            "u_std_score": float(rated.std()) if len(rated) else 0.0,
            "u_age": float(age_map.get(u, np.nan)),
        })
    print(f"  valid users: {len(rows):,}  ({time.time()-t0:.0f}s)")

    # ---- Pass B: encode U theo chunk, dựng candidate + feature ----
    acc = {k: [] for k in ["cand", "label", "qid", "is_train"] + FEATURE_NAMES[:10]}
    pad = lambda arr, L: np.pad(arr, (0, L - len(arr)))[:L]
    for s in range(0, len(rows), CHUNK):
        chunk = rows[s:s + CHUNK]
        hid = torch.tensor(np.stack([pad(r["hist_idx"], K_HISTORY) for r in chunk]), dtype=torch.long)
        hsc = torch.tensor(np.stack([pad(r["hist_score"], K_HISTORY) for r in chunk]), dtype=torch.long)
        gid = torch.tensor([r["gender_id"] for r in chunk], dtype=torch.long)
        jb = torch.tensor([r["joined_bucket"] for r in chunk], dtype=torch.long)
        U = enc.encode(hid, hid != 0, hsc, gid, jb)                       # [c, 128]
        scores = (U @ enc.item_cache.t()).numpy()                         # [c, N] = cos_uv toàn item

        for ci, r in enumerate(chunk):
            qid = s + ci
            srow = scores[ci]
            # hard-neg: top retrieval ngoài seen + PAD/OOV
            masked = srow.copy()
            masked[r["seen"]] = -np.inf
            masked[:2] = -np.inf
            top = np.argpartition(masked, -(HARD_NEG + HARD_SKIP))[-(HARD_NEG + HARD_SKIP):]
            top = top[np.argsort(-masked[top])]               # sort desc, bỏ HARD_SKIP top nhất
            hard = top[HARD_SKIP:HARD_SKIP + HARD_NEG]
            # random-neg: ngoài seen ∪ hard
            exclude = set(r["seen"].tolist()) | set(hard.tolist())
            rand_pool = rng.integers(2, N, size=RANDOM_NEG * 3)
            rand = np.array([x for x in rand_pool if x not in exclude][:RANDOM_NEG])
            tgt = r["tgt_idx"]
            cand = np.concatenate([tgt, hard, rand]).astype(np.int64)
            label = np.concatenate([
                [grade(int(x)) for x in r["tgt_score"]],
                np.zeros(len(hard) + len(rand), int)]).astype(np.int32)

            hv = V[r["hist_idx"]]
            sims = V[cand] @ hv.T                                          # [c, h]
            ga, ta, go = itemfeat.affinity(cand, r["g_pref"], r["t_pref"])
            n = len(cand)
            acc["cand"].append(cand)
            acc["label"].append(label)
            acc["qid"].append(np.full(n, qid))
            acc["is_train"].append(np.full(n, r["is_train"]))
            acc["cos_uv"].append(srow[cand])
            acc["hist_cos_max"].append(sims.max(1))
            acc["hist_cos_mean"].append(sims.mean(1))
            acc["genre_aff"].append(ga); acc["theme_aff"].append(ta); acc["genre_overlap"].append(go)
            acc["u_n_rated"].append(np.full(n, r["u_n_rated"]))
            acc["u_mean_score"].append(np.full(n, r["u_mean_score"]))
            acc["u_std_score"].append(np.full(n, r["u_std_score"]))
            acc["u_account_age"].append(np.full(n, r["u_age"]))
        print(f"  chunk {s//CHUNK + 1}/{(len(rows)-1)//CHUNK + 1}  ({time.time()-t0:.0f}s)")

    cat = {k: np.concatenate(v) for k, v in acc.items()}
    cross = {name: cat[name] for name in FEATURE_NAMES[:10]}
    df = build_frame(itemfeat, cat["cand"], cross)
    df["u_account_age"] = df["u_account_age"].fillna(df["u_account_age"].median())
    df["label"] = cat["label"]
    df["qid"] = cat["qid"]

    for name, mask in [("train", cat["is_train"]), ("valid", ~cat["is_train"])]:
        sub = df[mask].reset_index(drop=True)
        sub.to_parquet(OUT / f"{name}.parquet")
        n_grp = sub["qid"].nunique()
        print(f"{name}: {len(sub):,} rows, {n_grp:,} groups, "
              f"pos_rate={ (sub['label']>0).mean():.3f}")
    print(f"DONE build_dataset ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
