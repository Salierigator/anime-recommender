"""Matrix Factorization (implicit ALS) + fold-in baseline cho cold-by-user.

MF học user_factors × item_factors. Nhưng cold-by-user hold-out TRỌN user -> user_factors
của test user KHÔNG tồn tại. Cách hợp lệ: học item_factors trên TRAIN, rồi với mỗi test
user **fold-in** = giải lại user-vector từ support history qua công thức ALS (model.recalculate_user)
-> score = u · item_factorsᵀ. Không học tham số per-user nào của test (chỉ dùng history).

Protocol v2: mask seen−query, history prefix cap. Cold slice: N/A by construction —
item factors của H không được học (H cách ly khỏi train), ALS không score được item
ngoài train. Output -> retriever/baselines/mf.txt.

HP-search: TRAIN có ~67.5M interaction -> 1 full fit ~4' (factors=64) đến ~9' (factors=128),
quá tốn để sweep nhiều combo. Vì vậy: **coarse sweep trên subset ~15k user ngẫu nhiên**
(fold-in chỉ cần item_factors -> eval val đầy đủ vẫn hợp lệ), chọn best theo val recall@HK,
rồi **refit config thắng trên FULL train** để report test. Giảm nhiệt CPU: cap num_threads +
OPENBLAS_NUM_THREADS=1 (tránh oversubscribe BLAS vs implicit -> vừa chậm vừa nóng máy).

Usage: venv/bin/python retriever/baselines/mf.py   (--smoke: grid + subset rút gọn)
"""
from __future__ import annotations

import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")    # phải set TRƯỚC khi import numpy/implicit

import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import threadpoolctl
from implicit.als import AlternatingLeastSquares

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))                 # import flat config/data/metrics
import data as data_mod
import _eval

SPLIT = "test"
ITERS = 15
NUM_THREADS = 4          # nửa số core (8) -> giảm nhiệt; full fit factors=64 ~4', factors=128 ~9'
SWEEP_USERS = 15_000     # ~5% trong 291k user -> coarse sweep nhanh (~13-27s/fit trên subset)
# alpha (confidence) & factors là 2 đòn bẩy lớn nhất của implicit ALS trên data binary; reg=.05 (gốc).
# factors=256 bị bỏ: full refit của nó ~17' -> vượt budget (mở lại nếu cần capacity cao hơn).
FACTORS_GRID = [64, 128]
ALPHA_GRID = [1.0, 10.0, 40.0]
REG = 0.05
GRID_SMOKE = ([64], [1.0, 40.0])             # (factors, alpha) rút gọn


def load_train_arrays(cfg):
    """(user_idx, anime_idx) int64 của TRAIN positives — load 1 lần, tái dùng subset + full."""
    ds = data_mod.ExamplesDataset(cfg.train_data, "train")
    return ds.user_idx.astype(np.int64), ds.anime_idx.astype(np.int64)


def build_matrix(u, a, num_users, N) -> sp.csr_matrix:
    return sp.csr_matrix((np.ones(len(u), np.float32), (u, a)), shape=(num_users, N))


def subset_users_matrix(u, a, num_users, N, n_users, seed) -> sp.csr_matrix:
    """Ma trận từ n_users user NGẪU NHIÊN (đại diện, khác subset=first-N = toàn power-user)
    -> coarse sweep nhanh. item_factors học từ subset, fold-in eval dùng full val vẫn hợp lệ."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(u)
    keep = np.zeros(num_users, dtype=bool)
    keep[rng.choice(uniq, min(n_users, len(uniq)), replace=False)] = True
    m = keep[u]
    return build_matrix(u[m], a[m], num_users, N)


def fit_als(user_items, factors, reg, alpha, iters, seed, num_threads):
    model = AlternatingLeastSquares(
        factors=factors, regularization=reg, alpha=alpha, iterations=iters,
        random_state=seed, num_threads=num_threads,
    )
    model.fit(user_items)
    return model


def make_score_fn(cfg, model, N):
    item_f = torch.from_numpy(np.asarray(model.item_factors, dtype=np.float32)).to(cfg.device)  # [N,f]

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
    return score_fn


def main():
    smoke = "--smoke" in sys.argv
    threadpoolctl.threadpool_limits(1, "blas")    # khoá BLAS 1 thread (khớp OPENBLAS_NUM_THREADS=1)
    facs, alphas = (GRID_SMOKE if smoke else (FACTORS_GRID, ALPHA_GRID))
    n_sub = 5_000 if smoke else SWEEP_USERS

    cfg, spec, logq, users, qv, mv, _, _, _, _ = _eval.setup("val")
    N = logq.shape[0]
    HK = cfg.headline_k
    num_users = spec["num_users"]
    u_all, a_all = load_train_arrays(cfg)

    # --- coarse sweep (factors × alpha) trên subset user; track CẢ recall@HK lẫn ndcg@10 ---
    ui_sub = subset_users_matrix(u_all, a_all, num_users, N, n_sub, cfg.seed)
    sweep = []   # (factors, alpha, val_recall@HK, val_ndcg@10)
    for f in facs:
        for a in alphas:
            model = fit_als(ui_sub, f, REG, a, ITERS, cfg.seed, NUM_THREADS)
            out_v, _, _ = _eval.rank_eval(cfg, users, qv, logq, make_score_fn(cfg, model, N),
                                          cfg.eval_ks, mv)
            sweep.append((f, a, out_v[f"recall@{HK}"], out_v["ndcg@10"]))

    # Per-axis: config tốt nhất theo recall@HK và theo ndcg@10 (có thể trùng -> dedupe)
    best_recall = max(sweep, key=lambda r: r[2])
    best_ndcg = max(sweep, key=lambda r: r[3])
    report = [("recall-optimal", best_recall)]
    if (best_ndcg[0], best_ndcg[1]) != (best_recall[0], best_recall[1]):
        report.append(("ndcg-optimal", best_ndcg))

    # --- refit từng config thắng trên FULL train -> report TEST (+ liked) ---
    ui_full = build_matrix(u_all, a_all, num_users, N)
    _, _, _, usersT, q_warm, m_warm, _, _, qs_warm, _ = _eval.setup(SPLIT)

    n_cand = int(torch.isfinite(logq).sum())
    lines = _eval.header(
        "MF (implicit ALS) + fold-in — per-axis report" + (" [SMOKE]" if smoke else ""),
        cfg, SPLIT, n_cand,
        extra=f"refit on FULL train; HP coarse-swept on {n_sub} random users (val)")
    lines += [f"## val sweep (coarse on {n_sub} users) — factors/alpha -> recall@{HK} / ndcg@10:"]
    lines += [f"  factors={f:<4} alpha={a:<5} -> r@{HK} {rc:.6f}  ndcg@10 {nd:.6f}"
              for f, a, rc, nd in sorted(sweep, key=lambda r: r[2], reverse=True)]
    lines += [""]
    for label, (f, a, _, _) in report:
        model = fit_als(ui_full, f, REG, a, ITERS, cfg.seed, NUM_THREADS)
        out_w, n_w, _ = _eval.rank_eval(cfg, usersT, q_warm, logq, make_score_fn(cfg, model, N),
                                        cfg.eval_ks, m_warm, query_scores=qs_warm)
        lines += _eval.section(
            f"MF {label} (factors={f}, alpha={a}, reg={REG}, iters={ITERS}) — warm (test)",
            out_w, cfg.eval_ks, n_w)
    lines += ["## cold (test_cold)",
              "N/A — item factors của H không được học (H cách ly khỏi train);",
              "ALS không score được item ngoài train. Comparator cold: content_based / meta_popular.",
              ""]
    _eval.write_result(HERE / ("mf_smoke.txt" if smoke else "mf.txt"), lines)


if __name__ == "__main__":
    main()
