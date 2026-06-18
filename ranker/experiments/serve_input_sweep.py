"""serve_input_sweep.py — thí nghiệm cách DỰNG vector U lúc serve (KHÔNG retrain/export).

Mirror ranker/data_prep/build_eval.py::build_split nhưng chấm metric IN-MEMORY (không ghi pool
parquet, không đụng artifacts/). Chỉ ĐỌC artifacts/ + cleaned-data/; ghi 1 file leaderboard.csv
ở repo ROOT.

Hai phương pháp, mỗi cái đo cả retriever-only + two-stage (α=1) trên val (warm) + val_cold:
  1. z-score subset cho U: chỉ lấy item history có (score-mean)/std >= t (loại score=0; mean/std
     trên rated score>=1), quét t. SCOPE = chỉ đổi vector U (encode_users) — ranker hist-features
     + u_mean/u_std VẪN dùng full history.
  2. MAL-sort: pool retriever (U full-history) xếp theo mal_score desc (retriever-only).

Headline = liked_ndcg@10. Sort theo split, trong split theo liked_ndcg@10 desc.

CAVEAT:
- MAL-sort trên val_cold: item cold bị cold-policy ép mal_score -> median impute
  (features.py ItemFeatures.load), nên cold query xếp ~đồng hạng giữa bảng -> recall thấp;
  là kết quả trung thực, không phải bug.
- z@t cao -> nhiều user subset rỗng -> h_empty (vector generic); cột n_empty cho biết bao nhiêu.

    venv/bin/python ranker/experiments/serve_input_sweep.py

⚠ THỨ TỰ IMPORT load-bearing (giữ nguyên — service/CLAUDE.md §4):
  (1) torch TRƯỚC lightgbm (2 OpenMP runtime -> segfault mac);
  (2) features/pool/metrics TRƯỚC user_encode (user_encode chèn retriever/src vào sys.path[0],
      cũng có config.py -> import sau nó lấy NHẦM config retriever).
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

import torch  # noqa: E402  (TRƯỚC lightgbm)

NTHREAD = os.cpu_count() or 4
torch.set_num_threads(NTHREAD)
import config  # noqa: E402
from features import ItemFeatures, FEATURE_NAMES, build_frame  # noqa: E402
from metrics import blend, eval_pool  # noqa: E402
from pool import (UsersHistory, account_age_by_user, cross_features,  # noqa: E402
                  encode_users, load_eval_seen, load_queries,
                  topk_pool, user_stats_from_support)
from user_encode import load_user_encoder  # noqa: E402
import lightgbm as lgb  # noqa: E402  (SAU torch)

ROOT = config.ROOT.parent
D = config.POOL_DEPTH                    # 500
T_GRID = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
KS_FULL = [10, 50, 100, 200, 500]        # retriever / mal-sort (depth 500)
KS_RERANK = [10, 50, 100, 200]           # two-stage (rerank top-k_retrieve)
REF_KEY = {"val": "val_warm", "val_cold": "val_cold"}


def z_select(t: float):
    """Trả hàm lọc history theo z-score: giữ item rated (score>=1) có (score-mean)/std >= t.
    mean/std trên rated; std==0 -> z=0 (t=0 giữ hết rated, t>0 bỏ hết). Rỗng -> ([], [])."""
    def f(ids: np.ndarray, sc: np.ndarray):
        rated = sc[sc >= 1]
        if len(rated) == 0:
            return ids[:0], sc[:0]
        mu, sd = float(rated.mean()), float(rated.std())   # ddof=0, khớp user_stats_from_support
        z = (sc.astype(np.float64) - mu) / sd if sd > 1e-9 else np.zeros(len(sc))
        keep = (sc >= 1) & (z >= t)
        return ids[keep], sc[keep]
    return f


def build_pool(split: str, hist_select, enc, cap: int, itemfeat, V, uh, seen, ages, booster, kr):
    """Mirror build_eval.build_split nhưng IN-MEMORY. hist_select áp CHỈ cho U (encode_users);
    cross-features + user_stats dùng FULL history. Pool sâu D=500 cho retriever/mal-sort; feature
    + booster.predict CHỈ trên top-kr (=k_retrieve, two-stage chỉ rerank ngần đó). Trả dict arrays."""
    queries, query_scores = load_queries(split)
    uids = np.asarray(sorted(queries), dtype=np.int64)
    cos5_l, lab5_l, ll5_l, cand5_l = [], [], [], []
    cos2_l, pred2_l, lab2_l, ll2_l = [], [], [], []
    rtot_l, rlik_l = [], []
    n_empty = 0
    t0 = time.time()
    for s in range(0, len(uids), config.CHUNK):
        chunk = uids[s : s + config.CHUNK]
        hist_ids = [uh.history(u)[0] for u in chunk]
        hist_scores = [uh.history(u)[1] for u in chunk]
        sel = [hist_select(i, sc) for i, sc in zip(hist_ids, hist_scores)]
        n_empty += sum(len(i) == 0 for i, _ in sel)
        U = encode_users(enc, [i for i, _ in sel], [sc for _, sc in sel],
                         uh.gender_id[chunk], uh.joined_bucket[chunk], cap)
        mask_lists = [np.setdiff1d(seen[u], queries[u]) for u in chunk]
        cand, cos = topk_pool(U, enc.item_cache, mask_lists, D)         # [n, 500]
        labels = np.stack([np.isin(c, queries[u]) for c, u in zip(cand, chunk)]).astype(np.int8)
        stats = user_stats_from_support(
            hist_scores, np.asarray([ages.get(int(u), np.nan) for u in chunk]))
        liked_sets = []
        for i, u in enumerate(chunk):
            if stats["u_n_rated"][i] > 0:
                qa, qsc = queries[u], query_scores[u]
                liked_sets.append(qa[(qsc >= 1) & (qsc > stats["u_mean_score"][i])])
            else:
                liked_sets.append(np.empty(0, dtype=np.int64))
        label_liked = np.stack([np.isin(c, ls) for c, ls in zip(cand, liked_sets)]).astype(np.int8)
        # two-stage chỉ rerank top-kr -> feature + predict CHỈ trên slice đó
        candk, cosk = cand[:, :kr], cos[:, :kr]
        cross = cross_features(V, itemfeat, candk, cosk, hist_ids, stats)
        frame = build_frame(itemfeat, candk.ravel(), cross)
        pred = booster.predict(frame[FEATURE_NAMES], num_threads=NTHREAD)
        cos5_l.append(cos.ravel()); lab5_l.append(labels.ravel())
        ll5_l.append(label_liked.ravel()); cand5_l.append(cand.ravel())
        cos2_l.append(cosk.ravel()); pred2_l.append(np.asarray(pred))
        lab2_l.append(labels[:, :kr].ravel()); ll2_l.append(label_liked[:, :kr].ravel())
        rtot_l.append(np.asarray([len(queries[u]) for u in chunk]))
        rlik_l.append(np.asarray([len(ls) for ls in liked_sets]))
        print(f"  [{split}] {min(s + config.CHUNK, len(uids)):,}/{len(uids):,} "
              f"({time.time() - t0:.0f}s)", flush=True)
    G = len(uids)
    return {
        "G": G, "n_empty": n_empty,
        "cos5": np.concatenate(cos5_l).astype(np.float64),
        "lab5": np.concatenate(lab5_l).astype(np.int8),
        "ll5": np.concatenate(ll5_l).astype(np.int8),
        "cand5": np.concatenate(cand5_l).astype(np.int64),
        "off5": np.arange(G + 1, dtype=np.int64) * D,
        "cos2": np.concatenate(cos2_l).astype(np.float64),
        "pred2": np.concatenate(pred2_l).astype(np.float64),
        "lab2": np.concatenate(lab2_l).astype(np.int8),
        "ll2": np.concatenate(ll2_l).astype(np.int8),
        "off2": np.arange(G + 1, dtype=np.int64) * kr,
        "r_total": np.concatenate(rtot_l), "r_liked": np.concatenate(rlik_l),
    }


def metrics_retriever(p, scores5, pooled):
    return eval_pool(scores5, p["lab5"], p["off5"], p["r_total"], KS_FULL,
                     pooled=pooled, label_liked=p["ll5"], r_liked=p["r_liked"])


def metrics_twostage(p, pooled, alpha):
    return eval_pool(blend(p["cos2"], p["pred2"], p["off2"], alpha), p["lab2"], p["off2"],
                     p["r_total"], KS_RERANK, pooled=pooled, label_liked=p["ll2"],
                     r_liked=p["r_liked"])


def main():
    enc, meta = load_user_encoder("cpu")
    cap = meta.get("eval_history_cap", 1024)
    V = enc.item_cache.numpy()
    itemfeat = ItemFeatures.load(config.ARTIFACTS, config.CLEANED)
    mal_score = itemfeat.item["mal_score"]
    uh = UsersHistory()
    seen = load_eval_seen()
    ages = account_age_by_user()
    booster = lgb.Booster(model_file=str(config.ARTIFACTS / "ranker.txt"))
    assert booster.feature_name() == FEATURE_NAMES, "ranker.txt lệch FEATURE_NAMES"
    rmeta = json.loads((config.ARTIFACTS / "ranker_meta.json").read_text())
    alpha, k_retrieve = float(rmeta["blend_alpha"]), int(rmeta["k_retrieve"])
    ref = json.loads((config.ARTIFACTS / "eval_reference.json").read_text())
    print(f"alpha={alpha} k_retrieve={k_retrieve} cap={cap}")

    rows = []

    def add(split, method, stage, t, m, n_empty):
        row = {"split": split, "method": method, "stage": stage,
               "t": "" if t is None else t, "n_users": m.get("n_users", ""),
               "n_users_liked": m.get("n_users_liked", ""), "n_empty": n_empty,
               "n_pairs": m.get("n_pairs", "")}
        for k in KS_FULL:
            for pre in ("recall", "ndcg", "liked_recall", "liked_ndcg", "hitrate"):
                key = f"{pre}@{k}"
                row[key] = round(m[key], 5) if key in m else ""
        rows.append(row)

    for split in ("val", "val_cold"):
        pooled = split.endswith("cold")
        print(f"\n=== {split} (pooled={pooled}) ===")

        # full-history pool: cosine + two-stage + mal-sort
        full = build_pool(split, lambda i, sc: (i, sc), enc, cap, itemfeat, V, uh,
                          seen, ages, booster, k_retrieve)
        m_cos = metrics_retriever(full, full["cos5"], pooled)
        add(split, "full", "retriever", None, m_cos, full["n_empty"])
        add(split, "full", "two-stage", None,
            metrics_twostage(full, pooled, alpha), full["n_empty"])
        add(split, "malsort", "retriever", None,
            metrics_retriever(full, mal_score[full["cand5"]], pooled), full["n_empty"])

        # sanity: cosine_full phải khớp eval_reference.json (lệch > 2e-3 = pool sai protocol)
        rk = ref.get(REF_KEY[split], {})
        for k in KS_FULL:
            key = f"recall@{k}"
            if key in rk and abs(m_cos[key] - rk[key]) > 2e-3:
                print(f"  ⚠ SANITY {split} {key}: got {m_cos[key]:.4f} vs ref {rk[key]:.4f}")

        # z-score subset cho U
        for t in T_GRID:
            pz = build_pool(split, z_select(t), enc, cap, itemfeat, V, uh, seen, ages,
                            booster, k_retrieve)
            add(split, f"z@{t}", "retriever", t, metrics_retriever(pz, pz["cos5"], pooled),
                pz["n_empty"])
            add(split, f"z@{t}", "two-stage", t,
                metrics_twostage(pz, pooled, alpha), pz["n_empty"])

    # sort: theo split, trong split theo liked_ndcg@10 desc
    def lik(r):
        v = r.get("liked_ndcg@10", "")
        return v if isinstance(v, (int, float)) else -1.0
    rows.sort(key=lambda r: (r["split"], -lik(r)))

    fields = ["split", "method", "stage", "t", "n_users", "n_users_liked", "n_empty", "n_pairs"]
    for k in KS_FULL:
        fields += [f"recall@{k}", f"ndcg@{k}", f"liked_recall@{k}", f"liked_ndcg@{k}",
                   f"hitrate@{k}"]
    out = ROOT / "leaderboard.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(rows)
    print(f"\n-> {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
