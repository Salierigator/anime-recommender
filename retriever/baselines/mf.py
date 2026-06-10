"""Matrix Factorization (implicit ALS) + fold-in baseline cho cold-by-user.

MF học user_factors × item_factors. Nhưng cold-by-user hold-out TRỌN user -> user_factors
của test user KHÔNG tồn tại. Cách hợp lệ: học item_factors trên TRAIN, rồi với mỗi test
user **fold-in** = giải lại user-vector từ support history qua công thức ALS (model.recalculate_user)
-> score = u · item_factorsᵀ. Không học tham số per-user nào của test (chỉ dùng history).

Mask non-candidate (logq) + item đã seen, top-K, recall@K/ndcg@K y hệt protocol.
Output -> model/baselines/mf.txt.

Usage: venv/bin/python model/baselines/mf.py
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
from implicit.als import AlternatingLeastSquares

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))                 # model/ -> import flat config/data/metrics
import config as cfg_mod
import data as data_mod
from metrics import group_examples
import _eval

SPLIT = "test"
FACTORS = 64
ITERS = 15
REG = 0.05


def build_user_items(cfg, num_users, num_items, subset=None) -> sp.csr_matrix:
    """[U, N] binary từ TRAIN positives (chỉ train user -> không leak test)."""
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
    model = AlternatingLeastSquares(
        factors=FACTORS, regularization=REG, iterations=ITERS, random_state=cfg.seed,
    )
    model.fit(user_items)
    item_f = torch.from_numpy(np.asarray(model.item_factors, dtype=np.float32)).to(cfg.device)  # [N, f]

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
        uf = model.recalculate_user(np.arange(E), H)                       # fold-in [E, f]
        uf = torch.from_numpy(np.asarray(uf, dtype=np.float32)).to(cfg.device)
        return uf @ item_f.t()                                             # [E, N]

    ds = data_mod.ExamplesDataset(cfg.train_data, SPLIT, subset=4000 if smoke else None)
    queries = group_examples(ds.user_idx, ds.anime_idx)
    out, n, n_cand = _eval.rank_eval(cfg, users, queries, logq, score_fn, cfg.eval_ks)

    header = [
        f"# MF (implicit ALS, factors={FACTORS}, iters={ITERS}) + fold-in — split={SPLIT}{' [SMOKE]' if smoke else ''}  (trained on split=train)",
        f"# generated {datetime.now().isoformat(timespec='seconds')}  device={cfg.device}",
        f"# users evaluated: {n:,}   candidates (finite logq): {n_cand:,}",
    ]
    out_name = "mf_smoke.txt" if smoke else "mf.txt"
    _eval.write_result(HERE / out_name, header, out, cfg.eval_ks)


if __name__ == "__main__":
    main()
