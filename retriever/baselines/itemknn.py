"""ItemKNN (item-item cosine co-occurrence) baseline — CF "thật" cho cold-by-user.

CF chuẩn cho setting hold-out trọn user: KHÔNG học tham số per-user (user lạ lúc eval vẫn
score được). "Train" = dựng ma trận user×item từ TRAIN positives -> similarity item-item
cosine (top-K neighbor mỗi item, qua implicit.CosineRecommender, chỉ đếm co-occurrence trên
train -> không leak test).

User (cold): score(i) = Σ_{j∈history} cosine(i, j). Mask non-candidate (logq) + item đã seen,
top-K, đo recall@K/ndcg@K y hệt protocol. Output -> model/baselines/itemknn.txt.

Usage: venv/bin/python model/baselines/itemknn.py
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
from implicit.nearest_neighbours import CosineRecommender

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))                 # model/ -> import flat config/data/metrics
import config as cfg_mod
import data as data_mod
from metrics import group_examples
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
    cfg = cfg_mod.TwoTowerConfig()
    spec = data_mod.load_feature_spec(cfg.train_data)
    logq = data_mod.load_logq(cfg.train_data).to(cfg.device)
    users = data_mod.UserTable(cfg.train_data, spec["k_history"], spec["hard_neg_cap"])
    N = logq.shape[0]
    smoke = "--smoke" in sys.argv

    user_items = build_user_items(cfg, spec["num_users"], N, subset=200_000 if smoke else None)
    model = CosineRecommender(K=KNN)
    model.fit(user_items)
    S = model.similarity.tocsr().astype(np.float32)                          # [N, N] top-K item-item

    def score_fn(u, hist):
        h = hist.cpu().numpy()                                              # [E, Hk]
        E, Hk = h.shape
        rows = np.repeat(np.arange(E), Hk)
        cols = h.reshape(-1)
        keep = cols != 0                                                    # bỏ pad
        H = sp.csr_matrix(
            (np.ones(keep.sum(), np.float32), (rows[keep], cols[keep])),
            shape=(E, N),
        )
        scores = np.asarray((H @ S).todense(), dtype=np.float32)            # [E, N]
        return torch.from_numpy(scores).to(cfg.device)

    ds = data_mod.ExamplesDataset(cfg.train_data, SPLIT, subset=4000 if smoke else None)
    queries = group_examples(ds.user_idx, ds.anime_idx)
    out, n, n_cand = _eval.rank_eval(cfg, users, queries, logq, score_fn, cfg.eval_ks)

    header = [
        f"# ItemKNN (item-item cosine, K={KNN}) baseline — split={SPLIT}{' [SMOKE]' if smoke else ''}  (co-occurrence on split=train)",
        f"# generated {datetime.now().isoformat(timespec='seconds')}  device={cfg.device}",
        f"# users evaluated: {n:,}   candidates (finite logq): {n_cand:,}   sim nnz: {S.nnz:,}",
    ]
    out_name = "itemknn_smoke.txt" if smoke else "itemknn.txt"
    _eval.write_result(HERE / out_name, header, out, cfg.eval_ks)


if __name__ == "__main__":
    main()
