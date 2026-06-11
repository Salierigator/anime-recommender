"""build_train.py — dataset train ranker từ TRAIN users (local, re-run khi artifacts đổi).

Per train user (sample N_TRAIN_USERS, n_pos ≥ MIN_POS_TRAIN, RNG per-user deterministic):
  1. split positives support 80% / target 20% (tie random — CÙNG cơ chế support/query của
     protocol eval → train khớp eval khớp serve).
  2. U từ support → candidates = top-K_POOL cosine, mask = support ∪ hard_neg ∪ cold
     (target vẫn retrievable — mirror seen−query; cold loại vì train user không bao giờ
     có positive cold H → giữ lại = dạy model dìm cold).
     KHÔNG inject target, KHÔNG random-neg — đúng phân phối serve.
  3. label = grade(score) nếu cand ∈ target (10→4, 9→3, 7-8→2, 0/5/6→1), còn lại 0.
     Drop group không dính positive nào (không có signal lambdarank).

Valid cho early-stopping = pools/eval_val.parquet (label binary) — KHÔNG build riêng.

Output: ranker/data/datasets/train.parquet + train_users.parquet + build_meta.json
        (+ in danh sách file cần upload Drive cho Colab).

    venv/bin/python ranker/src/build_train.py [--n-users 100000] [--smoke]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone

import numpy as np
import polars as pl
import torch  # noqa: F401

import config
from features import ItemFeatures, build_frame
from pool import (PoolWriter, UsersHistory, account_age_by_user, cross_features,
                  encode_users, topk_pool, user_stats_from_support)
from user_encode import load_user_encoder


def select_train_users(uh: UsersHistory, n_users: int) -> np.ndarray:
    split = pl.read_parquet(config.ARTIFACTS / "user_split.parquet")
    train_ids = split.filter(pl.col("split") == "train")["user_idx"].to_numpy()
    eligible = train_ids[uh.history_lens()[train_ids] >= config.MIN_POS_TRAIN]
    rng = np.random.default_rng(config.SEED)
    pick = rng.choice(eligible, size=min(n_users, len(eligible)), replace=False)
    return np.sort(pick).astype(np.int64)


def split_support_target(ids: np.ndarray, scores: np.ndarray, u: int):
    """Tie random per-user (deterministic theo (SEED, user_idx)) → 20% target, còn lại support
    (giữ nguyên thứ tự gốc = vẫn sort score desc)."""
    rng = np.random.default_rng([config.SEED, int(u)])
    tie = rng.random(len(ids))
    n_target = int(np.clip(round(config.TARGET_FRAC * len(ids)), 1, len(ids) - 1))
    is_tgt = np.zeros(len(ids), bool)
    is_tgt[np.argsort(tie)[:n_target]] = True
    return ids[~is_tgt], scores[~is_tgt], ids[is_tgt], scores[is_tgt]


def main() -> None:
    ap = argparse.ArgumentParser(description="Build train dataset cho ranker")
    ap.add_argument("--n-users", type=int, default=config.N_TRAIN_USERS)
    ap.add_argument("--smoke", action="store_true", help="2k user (kiểm tra end-to-end)")
    args = ap.parse_args()
    n_users = 2_000 if args.smoke else args.n_users
    t0 = time.time()

    enc, meta = load_user_encoder("cpu")
    cap = meta.get("eval_history_cap", 1024)
    V = enc.item_cache.numpy()
    itemfeat = ItemFeatures.load(config.ARTIFACTS, config.CLEANED)
    cold_idx = np.flatnonzero(itemfeat.is_cold)
    uh = UsersHistory()
    ages = account_age_by_user()
    uids = select_train_users(uh, n_users)
    print(f"train users: {len(uids):,} (n_pos≥{config.MIN_POS_TRAIN}, seed {config.SEED})")

    writer = PoolWriter(config.DATASETS / "train.parquet",
                        config.DATASETS / "train_users.parquet")
    n_groups_in = 0
    grade_hist = np.zeros(5, dtype=np.int64)
    for s in range(0, len(uids), config.CHUNK):
        chunk = uids[s : s + config.CHUNK]
        supp_ids, supp_sc, tgt_maps, mask_lists, r_total = [], [], [], [], []
        for u in chunk:
            ids, sc = uh.history(int(u))
            si, ss, ti, ts = split_support_target(ids, sc, int(u))
            supp_ids.append(si)
            supp_sc.append(ss)
            tgt_maps.append(dict(zip(ti.tolist(), config.grade(ts).tolist())))
            mask_lists.append(np.union1d(si, uh.hard_neg(int(u))).astype(np.int64))
            r_total.append(len(ti))
        U = encode_users(enc, supp_ids, supp_sc,
                         uh.gender_id[chunk], uh.joined_bucket[chunk], cap)
        cand, cos = topk_pool(U, enc.item_cache, mask_lists, config.K_POOL, cold_idx=cold_idx)
        labels = np.zeros_like(cand, dtype=np.int8)
        for i, tm in enumerate(tgt_maps):
            labels[i] = [tm.get(int(a), 0) for a in cand[i]]
        keep = (labels > 0).any(axis=1)
        stats = user_stats_from_support(
            supp_sc, np.asarray([ages.get(int(u), np.nan) for u in chunk]))
        cross = cross_features(V, itemfeat, cand, cos, supp_ids, stats)
        frame = build_frame(itemfeat, cand.ravel(), cross)
        writer.add_chunk(chunk, cand, labels, frame, U.numpy(), supp_ids,
                         np.asarray(r_total), keep=keep)
        n_groups_in += len(chunk)
        vals, cnts = np.unique(labels[keep], return_counts=True)
        grade_hist[vals] += cnts
        print(f"  {min(s + config.CHUNK, len(uids)):,}/{len(uids):,} users, "
              f"kept {writer.next_qid:,} ({time.time() - t0:.0f}s)", flush=True)
    n_kept = writer.close()

    rows = n_kept * config.K_POOL
    bmeta = {
        "n_users_sampled": n_groups_in, "n_groups_kept": n_kept,
        "drop_rate_no_pos": round(1 - n_kept / n_groups_in, 4),
        "rows": rows, "k_pool": config.K_POOL,
        "pos_rate": round(float(grade_hist[1:].sum()) / rows, 4),
        "grade_hist": {str(g): int(c) for g, c in enumerate(grade_hist)},
        "grade_map": config.GRADE_MAP, "seed": config.SEED,
        "git_rev": subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                  capture_output=True, text=True,
                                  cwd=config.ROOT).stdout.strip(),
        "source_checkpoint": next(
            (l.strip("- ") for l in (config.ARTIFACTS / "CONTRACT.md").read_text().splitlines()
             if "Source checkpoint" in l), "?"),
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (config.DATASETS / "build_meta.json").write_text(json.dumps(bmeta, indent=2))
    print(f"\n[done] {n_kept:,} groups ({rows / 1e6:.1f}M rows), "
          f"drop {bmeta['drop_rate_no_pos']:.1%}, pos_rate {bmeta['pos_rate']:.4f}, "
          f"grades {bmeta['grade_hist']}  ({time.time() - t0:.0f}s)")
    print("\nUpload Drive (DRIVE/ranker_data/) cho ranker/train.ipynb:")
    for f in ["datasets/train.parquet", "datasets/train_users.parquet",
              "datasets/build_meta.json", "pools/eval_val.parquet",
              "pools/eval_val_users.parquet"]:
        print(f"  - ranker/data/{f}")
    print("  - artifacts/item_vectors.npy")


if __name__ == "__main__":
    main()
