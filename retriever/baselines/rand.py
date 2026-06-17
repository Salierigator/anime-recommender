"""Random baseline — sàn so sánh (protocol v2, warm + cold).

Điểm ngẫu nhiên mọi candidate; mask seen−query như mọi method. Cold-capable
(random phủ cả H). Output -> retriever/baselines/random.txt.

Usage: venv/bin/python retriever/baselines/rand.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))                 # import flat config/data/metrics
import _eval

SPLIT = "test"


def main():
    cfg, spec, logq, users, q_warm, m_warm, q_cold, m_cold, qs_warm, qs_cold = _eval.setup(SPLIT)
    N = logq.shape[0]
    g = torch.Generator(device=cfg.device).manual_seed(cfg.seed)

    def score_fn(u, hist):
        return torch.rand(len(u), N, generator=g, device=cfg.device)

    out_w, n_w, n_cand = _eval.rank_eval(cfg, users, q_warm, logq, score_fn, cfg.eval_ks, m_warm,
                                         query_scores=qs_warm)
    out_c, n_c, _ = _eval.rank_eval(cfg, users, q_cold, logq, score_fn, cfg.eval_ks, m_cold,
                                    pooled=True, query_scores=qs_cold)

    lines = _eval.header("Random baseline", cfg, SPLIT, n_cand, extra=f"seed={cfg.seed}")
    lines.append("# analytic recall@K ≈ K/N_cand: "
                 + "  ".join(f"@{k}={k / n_cand:.6f}" for k in cfg.eval_ks))
    lines.append("")
    lines += _eval.section("warm (test)", out_w, cfg.eval_ks, n_w)
    lines += _eval.section("cold (test_cold, full-catalog)", out_c, cfg.eval_ks, n_c, pooled=True)
    _eval.write_result(HERE / "random.txt", lines)


if __name__ == "__main__":
    main()
