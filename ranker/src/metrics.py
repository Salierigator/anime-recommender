"""metrics.py — two-stage recall/ndcg trên pool + rank_norm/blend. Pure numpy (không torch),
dùng được cả local lẫn Colab notebook.

Mirror ĐÚNG công thức retriever/src/metrics.py::evaluate (binary relevance, mean-per-user,
IDCG chuẩn hoá min(R_total, K)) — chỉ khác: rank trong pool top-D thay vì full catalog.
recall@K / ndcg@K với K ≤ D cho giá trị Y HỆT full ranking (top-K của full ranking ⊆ pool).
R_total = TỔNG số query của user (kể cả query không lọt pool) — mẫu số không đổi.

Pool format (parquet, build_eval/build_train ghi): rows sort theo qid CONTIGUOUS,
mỗi group = 1 user, trong group đã sort theo cosine desc (pool_rank tăng).
"""
from __future__ import annotations

import numpy as np


def group_offsets(qid: np.ndarray) -> np.ndarray:
    """qid contiguous (0..G-1, sort tăng) -> offsets [G+1]. Assert để bắt pool ghi sai."""
    change = np.r_[True, qid[1:] != qid[:-1]]
    starts = np.flatnonzero(change)
    assert (qid[starts] == np.arange(len(starts))).all(), "qid phải dense 0..G-1 contiguous"
    return np.r_[starts, len(qid)]


def rank_norm(x: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Thứ hạng chuẩn hoá [0,1] per group (giá trị lớn → gần 1). Ties phá tuỳ ý (đủ cho blend)."""
    out = np.empty(len(x), dtype=np.float64)
    for s, e in zip(offsets[:-1], offsets[1:]):
        n = e - s
        r = np.empty(n, dtype=np.float64)
        r[np.argsort(x[s:e], kind="stable")] = np.arange(n)
        out[s:e] = r / max(n - 1, 1)
    return out


def blend(cos: np.ndarray, pred: np.ndarray, offsets: np.ndarray, alpha: float) -> np.ndarray:
    """score = (1-α)·rank_norm(cos) + α·rank_norm(pred) per group — serve áp Y HỆT công thức này."""
    if alpha == 0.0:
        return cos.astype(np.float64)
    return (1 - alpha) * rank_norm(cos, offsets) + alpha * rank_norm(pred, offsets)


def eval_pool(scores: np.ndarray, labels: np.ndarray, offsets: np.ndarray,
              r_total: np.ndarray, ks, pooled: bool = False) -> dict:
    """Rerank từng group theo scores desc rồi đo recall@K/ndcg@K mean-per-user (labels binary
    0/1 = candidate ∈ query). pooled=True: thêm hitrate@K pooled trên mọi (user,query) pairs
    (cho slice cold mỏng). Trả {recall@K, ndcg@K, n_users[, hitrate@K, n_pairs]}."""
    G = len(offsets) - 1
    kmax = max(ks)
    discount = 1.0 / np.log2(np.arange(2, kmax + 2))
    idcg_cum = np.cumsum(discount)

    sums = {f"recall@{k}": 0.0 for k in ks}
    sums.update({f"ndcg@{k}": 0.0 for k in ks})
    pooled_hits = {k: 0.0 for k in ks}
    n = 0
    total_rel = 0
    for g in range(G):
        s, e = offsets[g], offsets[g + 1]
        R = int(r_total[g])
        if R == 0:
            continue
        order = np.argsort(-scores[s:e], kind="stable")[:kmax]
        hit = labels[s:e][order].astype(np.float64)
        for k in ks:
            h = hit[:k]
            n_hit = h.sum()
            sums[f"recall@{k}"] += n_hit / R
            dcg = (h * discount[: len(h)][:k]).sum()
            sums[f"ndcg@{k}"] += dcg / idcg_cum[min(R, k) - 1]
            if pooled:
                pooled_hits[k] += n_hit
        total_rel += R
        n += 1

    out = {m: (v / n if n else 0.0) for m, v in sums.items()}
    out["n_users"] = n
    if pooled:
        for k in ks:
            out[f"hitrate@{k}"] = pooled_hits[k] / total_rel if total_rel else 0.0
        out["n_pairs"] = total_rel
    return out


def load_pool_arrays(pool_path, users_path, k: int, max_groups: int | None = None):
    """Đọc pool parquet + slice top-k theo pool_rank. Trả (df polars, cos f64, label i8,
    offsets, r_total). Path explicit → dùng được cả local (config.POOLS) lẫn Colab.
    max_groups: chỉ lấy qid < N (smoke)."""
    import polars as pl

    df = pl.read_parquet(pool_path).filter(pl.col("pool_rank") < k)
    users = pl.read_parquet(users_path).sort("qid")
    if max_groups is not None:
        df = df.filter(pl.col("qid") < max_groups)
        users = users.filter(pl.col("qid") < max_groups)
    offsets = group_offsets(df["qid"].to_numpy())
    assert len(offsets) - 1 == users.height
    return (df, df["cos_uv"].to_numpy().astype(np.float64),
            df["label"].to_numpy().astype(np.int8), offsets,
            users["r_total"].to_numpy(), users)


def sweep_best_alpha(cos, pred, labels, offsets, r_total, ks, alphas) -> tuple[float, dict, dict]:
    """Sweep blend α, trả (best_alpha theo ndcg@10, metrics best, {alpha: metrics})."""
    all_m = {a: eval_pool(blend(cos, pred, offsets, a), labels, offsets, r_total, ks)
             for a in alphas}
    best = max((a for a in alphas if a > 0), key=lambda a: all_m[a]["ndcg@10"])
    return best, all_m[best], all_m


def fmt(m: dict, ks) -> str:
    cols = " ".join(f"r@{k}={m[f'recall@{k}']:.4f}" for k in ks)
    nd = " ".join(f"ndcg@{k}={m[f'ndcg@{k}']:.4f}" for k in ks if k in (10, 100))
    return f"{cols}  {nd}  (n_users={m['n_users']:,})"
