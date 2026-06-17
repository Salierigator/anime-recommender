"""ItemKNN (item-item cosine co-occurrence) baseline — CF "thật" cho cold-by-user.

CF chuẩn cho setting hold-out trọn user: KHÔNG học tham số per-user (user lạ lúc eval vẫn
score được). "Train" = dựng ma trận user×item từ TRAIN positives -> similarity item-item
cosine (top-K neighbor mỗi item, qua implicit.CosineRecommender, chỉ đếm co-occurrence trên
train -> không leak test).

User (cold): score(i) = Σ_{j∈history} cosine(i, j). Protocol v2: mask seen−query,
history prefix cap. Cold slice: N/A by construction — H không có co-occurrence train,
similarity với H = 0. Output -> retriever/baselines/itemknn.txt.

Usage: venv/bin/python retriever/baselines/itemknn.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
from implicit.nearest_neighbours import CosineRecommender

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))                 # import flat config/data/metrics
import data as data_mod
import _eval

SPLIT = "test"
KNN_GRID = [50, 100, 200, 500, 1000]       # số neighbor giữ mỗi item — sweep trên val
KNN_GRID_SMOKE = [100, 200]


def build_user_items(cfg, num_users, num_items, subset=None) -> sp.csr_matrix:
    """[U, N] binary từ TRAIN positives (đếm co-occurrence trên train -> không leak test)."""
    ds = data_mod.ExamplesDataset(cfg.train_data, "train", subset=subset)
    data = np.ones(len(ds.user_idx), dtype=np.float32)
    return sp.csr_matrix(
        (data, (ds.user_idx.astype(np.int64), ds.anime_idx.astype(np.int64))),
        shape=(num_users, num_items),
    )


def fit_sim(user_items, K) -> sp.csr_matrix:
    model = CosineRecommender(K=K)
    model.fit(user_items)
    return model.similarity.tocsr().astype(np.float32)                       # [N, N] top-K item-item


def make_score_fn(cfg, S, N):
    def score_fn(u, hist):
        h = hist.cpu().numpy()                                              # [E, W]
        E, W = h.shape
        rows = np.repeat(np.arange(E), W)
        cols = h.reshape(-1)
        keep = cols != 0                                                    # bỏ pad
        H = sp.csr_matrix(
            (np.ones(keep.sum(), np.float32), (rows[keep], cols[keep])),
            shape=(E, N),
        )
        scores = np.asarray((H @ S).todense(), dtype=np.float32)            # [E, N]
        return torch.from_numpy(scores).to(cfg.device)
    return score_fn


def main():
    smoke = "--smoke" in sys.argv
    grid = KNN_GRID_SMOKE if smoke else KNN_GRID

    # --- sweep K trên VAL (chọn theo recall@headline_k) ---
    cfg, spec, logq, users, qv, mv, _, _, _, _ = _eval.setup("val")
    N = logq.shape[0]
    HK = cfg.headline_k
    user_items = build_user_items(cfg, spec["num_users"], N, subset=200_000 if smoke else None)

    sweep = {}
    sims = {}
    for K in grid:
        sims[K] = fit_sim(user_items, K)
        out_v, _, _ = _eval.rank_eval(cfg, users, qv, logq, make_score_fn(cfg, sims[K], N),
                                      cfg.eval_ks, mv)
        sweep[K] = out_v[f"recall@{HK}"]
    best_k = max(sweep, key=sweep.get)
    S = sims[best_k]

    # --- report TEST với K thắng (+ liked) ---
    _, _, _, usersT, q_warm, m_warm, _, _, qs_warm, _ = _eval.setup(SPLIT)
    out_w, n_w, n_cand = _eval.rank_eval(cfg, usersT, q_warm, logq, make_score_fn(cfg, S, N),
                                         cfg.eval_ks, m_warm, query_scores=qs_warm)

    lines = _eval.header(
        f"ItemKNN (item-item cosine, K={best_k}, selected on val)" + (" [SMOKE]" if smoke else ""),
        cfg, SPLIT, n_cand, extra=f"co-occurrence on split=train, sim nnz={S.nnz:,}")
    lines += [f"## val sweep (recall@{HK}): "
              + "  ".join(f"K={K}: {sweep[K]:.6f}" for K in grid), ""]
    lines += _eval.section("warm (test)", out_w, cfg.eval_ks, n_w)
    lines += ["## cold (test_cold)",
              "N/A — H không có co-occurrence train -> similarity = 0, không score được.",
              "Comparator cold: content_based / meta_popular.",
              ""]
    _eval.write_result(HERE / ("itemknn_smoke.txt" if smoke else "itemknn.txt"), lines)


if __name__ == "__main__":
    main()
