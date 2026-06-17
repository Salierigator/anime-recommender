"""Vòng eval chung cho baselines — protocol v2 (khớp y hệt metrics.evaluate).

v2: mask = seen − query (eval_seen.parquet, KHÔNG chỉ history như v1); history cho
score_fn = prefix eval_history_cap (top-by-score) pad 0; 2 slice: warm (val/test)
và cold ({val,test}_cold — chỉ baseline cold-capable mới chạy); cold thêm pooled
hitrate@K. Mỗi baseline chỉ cấp `score_fn(u[E] np.int64, hist[E,W] long tensor
pad-0 trên device) -> scores [E, N]`.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
import torch

import config as cfg_mod
import data as data_mod
import metrics


def setup(split: str = "test"):
    """Load chung: cfg/spec/logq/users + protocol v2 (queries + mask cho 2 slice) +
    query scores cho liked-metric (qs_warm/qs_cold, song song queries từng slice)."""
    cfg = cfg_mod.TwoTowerConfig()
    spec = data_mod.load_feature_spec(cfg.train_data)
    logq = data_mod.load_logq(cfg.train_data).to(cfg.device)
    users = data_mod.UserTable(cfg.train_data, spec["hard_neg_cap"])
    q_warm, m_warm, q_cold, m_cold = metrics.load_eval_protocol(cfg.train_data, split)
    qs_warm = metrics.load_query_scores(cfg.train_data, split)
    qs_cold = metrics.load_query_scores(cfg.train_data, f"{split}_cold")
    return cfg, spec, logq, users, q_warm, m_warm, q_cold, m_cold, qs_warm, qs_cold


def rank_eval(cfg, users, queries: Dict[int, List[int]], logq, score_fn: Callable, ks,
              mask_ids: Dict[int, np.ndarray], batch: int = 512, pooled: bool = False,
              query_scores: Dict[int, List[int]] = None):
    """Trả (out_dict, n_users, n_cand). Mask candidate (logq=-inf) + seen−query
    y hệt metrics.evaluate (cùng discount/idcg/pooled).

    query_scores (song song queries): khi có -> thêm liked-recall@K/liked-ndcg@K
    report-only (mirror metrics.evaluate; liked = query score>=1 & score>u_mean
    per-user). Ranking/topk KHÔNG đổi -> so trực tiếp với binary."""
    device = cfg.device
    kmax = max(ks)
    candidate_mask = torch.isfinite(logq).to(device)
    n_cand = int(candidate_mask.sum())

    discount = 1.0 / np.log2(np.arange(2, kmax + 2))
    idcg_cum = np.cumsum(discount)

    uids = list(queries.keys())
    sums = {f"recall@{k}": 0.0 for k in ks}
    sums.update({f"ndcg@{k}": 0.0 for k in ks})
    pooled_hits = {k: 0.0 for k in ks}
    total_rel = 0
    n = 0

    liked_on = query_scores is not None
    u_mean = metrics.support_mean(users, uids) if liked_on else {}
    sums_liked = {f"liked_recall@{k}": 0.0 for k in ks}
    sums_liked.update({f"liked_ndcg@{k}": 0.0 for k in ks})
    n_liked = 0

    for s in range(0, len(uids), batch):
        chunk = uids[s : s + batch]
        u = np.asarray(chunk, dtype=np.int64)
        ids, _, _ = users.eval_history_batch(u, cfg.eval_history_cap)
        hist = torch.from_numpy(ids).long().to(device)           # [E,W] pad 0
        scores = score_fn(u, hist)                               # [E, N] on device
        scores[:, ~candidate_mask] = float("-inf")
        mpad = torch.from_numpy(metrics._pad_mask_ids(mask_ids, chunk)).to(device)
        scores.scatter_(1, mpad, float("-inf"))                  # mask seen − query
        topk = torch.topk(scores, kmax, dim=1).indices.cpu().numpy()

        for row, uid in enumerate(chunk):
            rel = set(queries[uid])
            R = len(rel)
            if R == 0:
                continue
            ranked = topk[row]
            hit = np.array([a in rel for a in ranked], dtype=np.float64)
            for k in ks:
                h = hit[:k]
                n_hit = h.sum()
                sums[f"recall@{k}"] += n_hit / R
                sums[f"ndcg@{k}"] += (h * discount[:k]).sum() / idcg_cum[min(R, k) - 1]
                if pooled:
                    pooled_hits[k] += n_hit
            total_rel += R
            n += 1

            if liked_on:
                mu = u_mean[uid]
                liked = set() if np.isnan(mu) else {
                    a for a, sc in zip(queries[uid], query_scores[uid]) if sc >= 1 and sc > mu}
                R_liked = len(liked)
                if R_liked:
                    hit_l = np.array([a in liked for a in ranked], dtype=np.float64)
                    for k in ks:
                        hl = hit_l[:k]
                        sums_liked[f"liked_recall@{k}"] += hl.sum() / R_liked
                        sums_liked[f"liked_ndcg@{k}"] += (hl * discount[:k]).sum() \
                            / idcg_cum[min(R_liked, k) - 1]
                    n_liked += 1

    out = {m: (v / n if n else 0.0) for m, v in sums.items()}
    if pooled:
        for k in ks:
            out[f"hitrate@{k}"] = pooled_hits[k] / total_rel if total_rel else 0.0
        out["n_pairs"] = total_rel
    if liked_on:
        out.update({m: (v / n_liked if n_liked else 0.0) for m, v in sums_liked.items()})
        out["n_users_liked"] = n_liked
    return out, n, n_cand


def section(title: str, out: Dict[str, float], ks, n: int, pooled: bool = False) -> List[str]:
    lines = [f"## {title}  (users: {n:,}" + (f", pairs: {out['n_pairs']:,})" if pooled else ")")]
    for k in ks:
        lines.append(f"recall@{k:<4} = {out[f'recall@{k}']:.6f}")
    lines.append("")
    for k in ks:
        lines.append(f"ndcg@{k:<6} = {out[f'ndcg@{k}']:.6f}")
    if pooled:
        lines.append("")
        for k in ks:
            lines.append(f"hitrate@{k:<3} = {out[f'hitrate@{k}']:.6f}")
    if f"liked_recall@{ks[0]}" in out:
        lines.append(f"\n# liked-metric (report-only; users w/ >=1 liked query: {out['n_users_liked']:,})")
        for k in ks:
            lines.append(f"liked_recall@{k:<4} = {out[f'liked_recall@{k}']:.6f}")
        lines.append("")
        for k in ks:
            lines.append(f"liked_ndcg@{k:<6} = {out[f'liked_ndcg@{k}']:.6f}")
    lines.append("")
    return lines


def header(name: str, cfg, split: str, n_cand: int, extra: str = "") -> List[str]:
    return [
        f"# {name} — split={split} — protocol v2 (mask seen−query, history cap={cfg.eval_history_cap})",
        f"# generated {datetime.now().isoformat(timespec='seconds')}  device={cfg.device}"
        + (f"  {extra}" if extra else ""),
        f"# candidates (finite logq): {n_cand:,}",
        "",
    ]


def write_result(path: Path, lines: List[str]):
    text = "\n".join(lines) + "\n"
    print(text)
    path.write_text(text)
    print(f"saved {path}")
