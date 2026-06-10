"""Matrix Factorization (implicit ALS) + fold-in baseline cho cold-by-user.

MF học user_factors × item_factors. Nhưng cold-by-user hold-out TRỌN user -> user_factors
của test user KHÔNG tồn tại. Cách hợp lệ: học item_factors trên TRAIN, rồi với mỗi test
user **fold-in** = giải lại user-vector từ support history qua công thức ALS (model.recalculate_user)
-> score = u · item_factorsᵀ. Không học tham số per-user nào của test (chỉ dùng history).

Protocol v2: mask seen−query, history prefix cap. Cold slice: N/A by construction —
item factors của H không được học (H cách ly khỏi train), ALS không score được item
ngoài train. Output -> retriever/baselines/mf.txt.

Usage: venv/bin/python retriever/baselines/mf.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
from implicit.als import AlternatingLeastSquares

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))                 # import flat config/data/metrics
import data as data_mod
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
    cfg, spec, logq, users, q_warm, m_warm, q_cold, m_cold = _eval.setup(SPLIT)
    N = logq.shape[0]
    smoke = "--smoke" in sys.argv

    user_items = build_user_items(cfg, spec["num_users"], N, subset=200_000 if smoke else None)
    model = AlternatingLeastSquares(
        factors=FACTORS, regularization=REG, iterations=ITERS, random_state=cfg.seed,
    )
    model.fit(user_items)
    item_f = torch.from_numpy(np.asarray(model.item_factors, dtype=np.float32)).to(cfg.device)  # [N, f]

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
        uf = model.recalculate_user(np.arange(E), H)                       # fold-in [E, f]
        uf = torch.from_numpy(np.asarray(uf, dtype=np.float32)).to(cfg.device)
        return uf @ item_f.t()                                             # [E, N]

    out_w, n_w, n_cand = _eval.rank_eval(cfg, users, q_warm, logq, score_fn, cfg.eval_ks, m_warm)

    lines = _eval.header(
        f"MF (implicit ALS, factors={FACTORS}, iters={ITERS}) + fold-in"
        + (" [SMOKE]" if smoke else ""), cfg, SPLIT, n_cand, extra="trained on split=train")
    lines += _eval.section("warm (test)", out_w, cfg.eval_ks, n_w)
    lines += ["## cold (test_cold)",
              "N/A — item factors của H không được học (H cách ly khỏi train);",
              "ALS không score được item ngoài train. Comparator cold: content_based / meta_popular.",
              ""]
    _eval.write_result(HERE / ("mf_smoke.txt" if smoke else "mf.txt"), lines)


if __name__ == "__main__":
    main()
