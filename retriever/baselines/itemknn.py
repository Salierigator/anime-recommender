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
KNN = 200                       # số neighbor giữ mỗi item


def build_user_items(cfg, num_users, num_items, subset=None) -> sp.csr_matrix:
    """[U, N] binary từ TRAIN positives (đếm co-occurrence trên train -> không leak test)."""
    ds = data_mod.ExamplesDataset(cfg.train_data, "train", subset=subset)
    data = np.ones(len(ds.user_idx), dtype=np.float32)
    return sp.csr_matrix(
        (data, (ds.user_idx.astype(np.int64), ds.anime_idx.astype(np.int64))),
        shape=(num_users, num_items),
    )


def main():
    cfg, spec, logq, users, q_warm, m_warm, q_cold, m_cold = _eval.setup(SPLIT)
    N = logq.shape[0]
    smoke = "--smoke" in sys.argv

    user_items = build_user_items(cfg, spec["num_users"], N, subset=200_000 if smoke else None)
    model = CosineRecommender(K=KNN)
    model.fit(user_items)
    S = model.similarity.tocsr().astype(np.float32)                          # [N, N] top-K item-item

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

    out_w, n_w, n_cand = _eval.rank_eval(cfg, users, q_warm, logq, score_fn, cfg.eval_ks, m_warm)

    lines = _eval.header(
        f"ItemKNN (item-item cosine, K={KNN})" + (" [SMOKE]" if smoke else ""),
        cfg, SPLIT, n_cand, extra=f"co-occurrence on split=train, sim nnz={S.nnz:,}")
    lines += _eval.section("warm (test)", out_w, cfg.eval_ks, n_w)
    lines += ["## cold (test_cold)",
              "N/A — H không có co-occurrence train -> similarity = 0, không score được.",
              "Comparator cold: content_based / meta_popular.",
              ""]
    _eval.write_result(HERE / ("itemknn_smoke.txt" if smoke else "itemknn.txt"), lines)


if __name__ == "__main__":
    main()
