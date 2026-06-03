"""Eval cold-by-user (headline metric): recall@K, ndcg@K (xem plan.md §2, TRAIN_DATA §5.3).

User eval: build U từ history (support, đã chống leak ở pipeline) -> score vs toàn item_cache
-> mask non-candidate + mask item đã có trong history -> rank -> so với query items (examples).
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch


def group_examples(user_idx: np.ndarray, anime_idx: np.ndarray) -> Dict[int, List[int]]:
    """user_idx -> list anime_idx (query items)."""
    out: Dict[int, List[int]] = {}
    for u, a in zip(user_idx.tolist(), anime_idx.tolist()):
        out.setdefault(u, []).append(a)
    return out


@torch.no_grad()
def evaluate(model, users, queries: Dict[int, List[int]], logq, ks, batch=512):
    """model.item_cache phải được refresh trước khi gọi. Trả dict {recall@K, ndcg@K}."""
    model.eval()
    device = model.item_cache.device
    kmax = max(ks)
    candidate_mask = torch.isfinite(logq).to(device)             # [N] True = item được rank

    uids = list(queries.keys())
    sums = {f"recall@{k}": 0.0 for k in ks}
    sums.update({f"ndcg@{k}": 0.0 for k in ks})
    n = 0

    # IDCG[k] precompute (relevant đứng đầu): sum_{i=1..k} 1/log2(i+1)
    discount = 1.0 / np.log2(np.arange(2, kmax + 2))
    idcg_cum = np.cumsum(discount)                               # idcg_cum[r-1] = IDCG cho r relevant trong topK

    for s in range(0, len(uids), batch):
        chunk = uids[s : s + batch]
        u = np.asarray(chunk, dtype=np.int64)
        hist = torch.from_numpy(users.history_pad[u]).long().to(device)
        hmask = hist != 0
        ub = {
            "history_ids": hist,
            "history_mask": hmask,
            "gender_id": torch.from_numpy(users.gender_id[u]).to(device),
            "joined_bucket": torch.from_numpy(users.joined_bucket[u]).to(device),
        }
        U = model.encode_users(ub)                              # [E, d]
        scores = U @ model.item_cache.t()                      # [E, N]
        scores[:, ~candidate_mask] = float("-inf")
        # mask item đã seen trong history (đừng recommend lại)
        scores.scatter_(1, hist, float("-inf"))                # hist pad=0 cũng bị -inf (non-candidate sẵn rồi)
        topk = torch.topk(scores, kmax, dim=1).indices.cpu().numpy()  # [E, kmax]

        for row, uid in enumerate(chunk):
            rel = set(queries[uid])
            R = len(rel)
            if R == 0:
                continue
            ranked = topk[row]
            hit = np.array([a in rel for a in ranked], dtype=np.float64)  # [kmax]
            for k in ks:
                h = hit[:k]
                n_hit = h.sum()
                sums[f"recall@{k}"] += n_hit / R
                dcg = (h * discount[:k]).sum()
                idcg = idcg_cum[min(R, k) - 1]
                sums[f"ndcg@{k}"] += dcg / idcg
            n += 1

    model.train()
    return {m: (v / n if n else 0.0) for m, v in sums.items()} | {"n_users": n}
