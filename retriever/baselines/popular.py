"""MostPopular baseline — popularity đếm trên TRAIN examples (protocol v2, warm only).

Cold slice: N/A by construction — mọi item H có popularity train = 0 (H bị cách ly
khỏi train) -> không bao giờ lọt top-K, recall cold = 0 đúng nghĩa đen. Xem
meta_popular.py cho prior popularity cold-capable (metadata members).

Usage: venv/bin/python retriever/baselines/popular.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))                 # import flat config/data/metrics
import data as data_mod
import _eval

SPLIT = "test"
POP_SPLIT = "train"   # đếm popularity trên train -> không leak test


def main():
    cfg, spec, logq, users, q_warm, m_warm, q_cold, m_cold = _eval.setup(SPLIT)
    N = logq.shape[0]
    ds = data_mod.ExamplesDataset(cfg.train_data, POP_SPLIT)
    pop = torch.from_numpy(
        np.bincount(ds.anime_idx, minlength=N).astype(np.float32)
    ).to(cfg.device)

    def score_fn(u, hist):
        return pop.unsqueeze(0).expand(len(u), N).clone()

    out_w, n_w, n_cand = _eval.rank_eval(cfg, users, q_warm, logq, score_fn, cfg.eval_ks, m_warm)

    lines = _eval.header("MostPopular baseline", cfg, SPLIT, n_cand,
                         extra=f"popularity on split={POP_SPLIT}")
    lines += _eval.section("warm (test)", out_w, cfg.eval_ks, n_w)
    lines += ["## cold (test_cold)",
              "N/A — popularity train của mọi item H = 0 by construction (H cách ly khỏi train),",
              "không bao giờ lọt top-K -> recall/hitrate cold = 0. Prior cold-capable: meta_popular.py.",
              ""]
    _eval.write_result(HERE / "popular.txt", lines)


if __name__ == "__main__":
    main()
