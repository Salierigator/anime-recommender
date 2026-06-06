"""Random baseline cho stage Retrieval — sàn so sánh với model đã train.

Dự đoán RANDOM cho mỗi user test: chấm điểm ngẫu nhiên các candidate (mask non-candidate
qua logq + mask item đã seen trong history), top-K, đo recall@K / ndcg@K y hệt protocol
metrics.evaluate. Lưu kết quả ra model/baselines/random.txt.

Usage: venv/bin/python model/baselines/random.py
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))                 # model/ -> import flat config/data/metrics
import config as cfg_mod
import data as data_mod
from metrics import group_examples

SPLIT = "test"


def random_eval(cfg, users, queries, logq, ks, seed=42, batch=512):
    device = cfg.device
    g = torch.Generator(device=device).manual_seed(seed)
    kmax = max(ks)
    candidate_mask = torch.isfinite(logq).to(device)         # [N] item được rank
    n_cand = int(candidate_mask.sum())
    N = logq.shape[0]

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
        E = len(chunk)
        scores = torch.rand(E, N, generator=g, device=device)  # điểm ngẫu nhiên
        scores[:, ~candidate_mask] = float("-inf")
        scores.scatter_(1, hist, float("-inf"))               # mask item đã seen
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


def main():
    cfg = cfg_mod.TwoTowerConfig()
    spec = data_mod.load_feature_spec(cfg.train_data)
    logq = data_mod.load_logq(cfg.train_data).to(cfg.device)
    users = data_mod.UserTable(cfg.train_data, spec["k_history"], spec["hard_neg_cap"])

    ds = data_mod.ExamplesDataset(cfg.train_data, SPLIT)
    queries = group_examples(ds.user_idx, ds.anime_idx)

    out, n, n_cand = random_eval(cfg, users, queries, logq, cfg.eval_ks, seed=cfg.seed)

    lines = [
        f"# Random baseline — split={SPLIT}",
        f"# generated {datetime.now().isoformat(timespec='seconds')}  device={cfg.device}  seed={cfg.seed}",
        f"# users evaluated: {n:,}   candidates (finite logq): {n_cand:,}",
        "",
    ]
    for k in cfg.eval_ks:
        analytic = k / n_cand
        lines.append(f"recall@{k:<4} = {out[f'recall@{k}']:.6f}   (analytic K/N_cand = {analytic:.6f})")
    lines.append("")
    for k in cfg.eval_ks:
        lines.append(f"ndcg@{k:<6} = {out[f'ndcg@{k}']:.6f}")
    text = "\n".join(lines) + "\n"

    print(text)
    path = HERE / "random.txt"
    path.write_text(text)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
