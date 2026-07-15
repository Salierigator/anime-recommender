"""build_eval.py — dựng eval pool val/test/val_cold ĐÚNG protocol retriever (local, re-run
mỗi khi artifacts đổi).

Per eval user: U từ support (= history trong users_history, đã trừ query+H) → top-POOL_DEPTH
cosine với mask = seen − query (KHÔNG mask query — đáp án đang chấm; KHÔNG loại cold — warm
eval cho cold cạnh tranh y như retriever) → features. Pool lưu sâu 500, eval slice 200/500.

Output: model/ranker/train-data/pools/eval_{split}.parquet (qid contiguous, trong group sort cosine desc,
label binary = cand ∈ query) + eval_{split}_users.parquet (qid, r_total, U, hist_top64).

    venv/bin/python model/ranker/data_prep/build_eval.py [--splits val test val_cold]
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone

import numpy as np
import torch  # noqa: F401  (trước lightgbm ở downstream; giữ convention)

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))   # lib chung ở ../src

import config  # noqa: E402
from features import ItemFeatures, build_frame  # noqa: E402
from pool import (PoolWriter, UsersHistory, account_age_by_user, cross_features,  # noqa: E402
                  encode_users, load_eval_seen, load_queries, topk_pool,
                  user_stats_from_support)
from user_encode import load_user_encoder  # noqa: E402


def build_split(split: str, enc, cap: int, itemfeat, V, uh: UsersHistory,
                seen: dict, ages: dict) -> int:
    t0 = time.time()
    queries, query_scores = load_queries(split)
    uids = np.asarray(sorted(queries), dtype=np.int64)
    writer = PoolWriter(config.POOLS / f"eval_{split}.parquet",
                        config.POOLS / f"eval_{split}_users.parquet")
    for s in range(0, len(uids), config.CHUNK):
        chunk = uids[s : s + config.CHUNK]
        hist_ids = [uh.history(u)[0] for u in chunk]
        hist_scores = [uh.history(u)[1] for u in chunk]
        U = encode_users(enc, hist_ids, hist_scores,
                         uh.gender_id[chunk], uh.joined_bucket[chunk], cap)
        mask_lists = [np.setdiff1d(seen[u], queries[u]) for u in chunk]
        cand, cos = topk_pool(U, enc.item_cache, mask_lists, config.POOL_DEPTH)
        labels = np.stack([np.isin(c, queries[u]) for c, u in zip(cand, chunk)]).astype(np.int8)
        stats = user_stats_from_support(
            hist_scores, np.asarray([ages.get(int(u), np.nan) for u in chunk]))
        # liked = query item có score>=1 & score>u_mean (per-user above-own-mean, mirror retriever);
        # u_n_rated==0 (không có support rated) -> u_mean vô nghĩa -> không có liked.
        liked_sets = []
        for i, u in enumerate(chunk):
            if stats["u_n_rated"][i] > 0:
                qa, qsc = queries[u], query_scores[u]
                liked_sets.append(qa[(qsc >= 1) & (qsc > stats["u_mean_score"][i])])
            else:
                liked_sets.append(np.empty(0, dtype=np.int64))
        label_liked = np.stack([np.isin(c, ls) for c, ls in zip(cand, liked_sets)]).astype(np.int8)
        r_liked = np.asarray([len(ls) for ls in liked_sets], dtype=np.int64)
        cross = cross_features(V, itemfeat, cand, cos, hist_ids, stats)
        frame = build_frame(itemfeat, cand.ravel(), cross)
        r_total = np.asarray([len(queries[u]) for u in chunk])
        writer.add_chunk(chunk, cand, labels, frame, U.numpy(), hist_ids, r_total,
                         label_liked=label_liked, r_liked=r_liked)
        print(f"  [{split}] {min(s + config.CHUNK, len(uids)):,}/{len(uids):,} users "
              f"({time.time() - t0:.0f}s)", flush=True)
    n = writer.close()
    print(f"[done] eval_{split}: {n:,} users, depth {config.POOL_DEPTH} "
          f"({time.time() - t0:.0f}s)")
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description="Build eval pools (val/test/val_cold)")
    ap.add_argument("--splits", nargs="+", default=["val", "test", "val_cold"])
    args = ap.parse_args()
    assert "test_cold" not in args.splits or \
        (config.ARTIFACTS / "eval_queries_test_cold.parquet").exists(), \
        "test_cold = final exam — cần export.py --final-exam (chấm 1 lần lúc chốt pipeline)"

    enc, meta = load_user_encoder("cpu")
    cap = meta.get("eval_history_cap", 1024)
    V = enc.item_cache.numpy()
    itemfeat = ItemFeatures.load(config.ARTIFACTS, config.CLEANED)
    uh = UsersHistory()
    seen = load_eval_seen()
    ages = account_age_by_user()

    counts = {s: build_split(s, enc, cap, itemfeat, V, uh, seen, ages) for s in args.splits}

    meta_path = config.POOLS / "build_meta.json"
    old = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    contract = (config.ARTIFACTS / "CONTRACT.md").read_text().splitlines()
    old.update({
        "pool_depth": config.POOL_DEPTH, "hist_feat_cap": config.HIST_FEAT_CAP,
        "n_users": {**old.get("n_users", {}), **counts},
        "source_checkpoint": next((l.strip("- ") for l in contract if "Source checkpoint" in l), "?"),
        "artifacts_generated": next((l.strip("- ") for l in contract if "Generated" in l), "?"),
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })
    meta_path.write_text(json.dumps(old, indent=2))
    print(f"meta -> {meta_path}")


if __name__ == "__main__":
    main()
