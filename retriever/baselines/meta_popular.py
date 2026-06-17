"""Meta-Popular baseline — prior popularity COLD-CAPABLE (protocol v2, warm + cold).

Điểm = log1p(members) từ metadata details.csv (số người add vào list trên MAL tại
thời điểm scrape) — KHÔNG đụng interactions train nên định nghĩa được cho cả item H.
Đây là baseline "cứ gợi ý anime mới đang hype cho mọi người" mà model phải vượt
trên cold slice. Output -> retriever/baselines/meta_popular.txt.

Usage: venv/bin/python retriever/baselines/meta_popular.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))                 # import flat config/data/metrics
import _eval

SPLIT = "test"
DETAILS = HERE.parent.parent / "cleaned-data" / "details.csv"


def members_scores(cfg, N) -> torch.Tensor:
    """Dense [N]: log1p(members) theo anime_idx (PAD/OOV = 0, sẽ bị candidate mask)."""
    df = pd.read_csv(DETAILS, usecols=["mal_id", "members"])
    amap = pl.read_parquet(cfg.train_data / "anime_id_map.parquet").to_pandas()
    df = amap.merge(df, on="mal_id", how="left")
    scores = np.zeros(N, dtype=np.float32)
    scores[df["anime_idx"].to_numpy()] = np.log1p(df["members"].fillna(0).to_numpy())
    return torch.from_numpy(scores)


def main():
    cfg, spec, logq, users, q_warm, m_warm, q_cold, m_cold, qs_warm, qs_cold = _eval.setup(SPLIT)
    N = logq.shape[0]
    pop = members_scores(cfg, N).to(cfg.device)

    def score_fn(u, hist):
        return pop.unsqueeze(0).expand(len(u), N).clone()

    out_w, n_w, n_cand = _eval.rank_eval(cfg, users, q_warm, logq, score_fn, cfg.eval_ks, m_warm,
                                         query_scores=qs_warm)
    out_c, n_c, _ = _eval.rank_eval(cfg, users, q_cold, logq, score_fn, cfg.eval_ks, m_cold,
                                    pooled=True, query_scores=qs_cold)

    lines = _eval.header("Meta-Popular baseline (log1p members, metadata)", cfg, SPLIT, n_cand)
    lines += _eval.section("warm (test)", out_w, cfg.eval_ks, n_w)
    lines += _eval.section("cold (test_cold, full-catalog)", out_c, cfg.eval_ks, n_c, pooled=True)
    _eval.write_result(HERE / "meta_popular.txt", lines)


if __name__ == "__main__":
    main()
