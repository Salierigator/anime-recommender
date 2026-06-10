"""Vòng eval chung + ghi kết quả cho các baseline retrieval (cold-by-user).

Tách phần lặp của baseline (rank + mask non-candidate/seen + recall@K/ndcg@K + ghi .txt)
ra đây; mỗi baseline chỉ cần cấp `score_fn(u, hist) -> [E, N]`. Protocol khớp y hệt
metrics.evaluate / popular_baseline / random_baseline (cùng discount, idcg, mask).
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
import torch


def rank_eval(cfg, users, queries, logq, score_fn: Callable, ks, batch: int = 512):
    """score_fn(u[E] int64 np, hist[E,Hk] long tensor on device) -> scores [E, N] on device.

    Trả (out_dict, n_users, n_cand). Mask non-candidate (logq=-inf) + item đã seen y hệt
    metrics.evaluate.
    """
    device = cfg.device
    kmax = max(ks)
    candidate_mask = torch.isfinite(logq).to(device)
    n_cand = int(candidate_mask.sum())

    discount = 1.0 / np.log2(np.arange(2, kmax + 2))
    idcg_cum = np.cumsum(discount)

    uids = list(queries.keys())
    sums = {f"recall@{k}": 0.0 for k in ks}
    sums.update({f"ndcg@{k}": 0.0 for k in ks})
    n = 0

    for s in range(0, len(uids), batch):
        chunk = uids[s : s + batch]
        u = np.asarray(chunk, dtype=np.int64)
        hist = torch.from_numpy(users.history_pad[u]).long().to(device)
        scores = score_fn(u, hist)                               # [E, N] on device
        scores[:, ~candidate_mask] = float("-inf")
        scores.scatter_(1, hist, float("-inf"))                  # mask item đã seen
        topk = torch.topk(scores, kmax, dim=1).indices.cpu().numpy()

        for row, uid in enumerate(chunk):
            rel = set(queries[uid])
            R = len(rel)
            if R == 0:
                continue
            hit = np.array([a in rel for a in topk[row]], dtype=np.float64)
            for k in ks:
                h = hit[:k]
                sums[f"recall@{k}"] += h.sum() / R
                sums[f"ndcg@{k}"] += (h * discount[:k]).sum() / idcg_cum[min(R, k) - 1]
            n += 1

    out = {m: (v / n if n else 0.0) for m, v in sums.items()}
    return out, n, n_cand


def write_result(path: Path, header: List[str], out: Dict[str, float], ks):
    lines = list(header) + [""]
    for k in ks:
        lines.append(f"recall@{k:<4} = {out[f'recall@{k}']:.6f}")
    lines.append("")
    for k in ks:
        lines.append(f"ndcg@{k:<6} = {out[f'ndcg@{k}']:.6f}")
    text = "\n".join(lines) + "\n"
    print(text)
    path.write_text(text)
    print(f"saved {path}")
