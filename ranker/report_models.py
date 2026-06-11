"""report_models.py — ghi results.txt vào từng ranker/models/<run>/ (format kiểu
retriever/baselines/*.txt) để xem nhanh kết quả từng model sau khi tải về.

Chấm two-stage trên pool đã build: warm VAL (sweep α, full K) + VAL_COLD (blend cùng α +
reference kênh cosine). KỶ LUẬT: KHÔNG chấm test ở đây — test warm chỉ qua eval.py cho model
được chọn; test_cold = final exam (eval.py --final-exam).

    venv/bin/python ranker/report_models.py [--k 200] [--models path ...]
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl
import torch  # noqa: F401  (trước lightgbm trong eval.predict)

import sys

HERE = Path(__file__).resolve().parent                 # ranker/
sys.path.insert(0, str(HERE / "src"))                  # lib chung (config/metrics)
sys.path.insert(0, str(HERE))                          # sibling eval.py (top-level)

import config  # noqa: E402
from eval import predict, run_label  # noqa: E402
from metrics import blend, eval_pool, load_pool_arrays, sweep_best_alpha  # noqa: E402

KS = [10, 50, 100, 200]


def fmt_block(m: dict, ks, pooled: bool = False) -> str:
    out = "\n".join(f"recall@{k:<4}= {m[f'recall@{k}']:.6f}" for k in ks) + "\n\n"
    out += "\n".join(f"ndcg@{k:<4}  = {m[f'ndcg@{k}']:.6f}" for k in ks) + "\n"
    if pooled:
        out += "\n" + "\n".join(f"hitrate@{k:<3}= {m[f'hitrate@{k}']:.6f}" for k in ks) + "\n"
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Ghi results.txt cho từng model trong models/")
    ap.add_argument("--k", type=int, default=config.K_POOL, choices=[200, 500])
    ap.add_argument("--models", nargs="*", type=Path)
    args = ap.parse_args()

    models = args.models or sorted(
        p for pat in ("*/model.txt", "*/model.pt", "*/model.npz") for p in config.MODELS.glob(pat))
    if not models:
        raise SystemExit(f"không có model trong {config.MODELS}")

    pools = {}
    for split in ("val", "val_cold"):
        pools[split] = load_pool_arrays(config.POOLS / f"eval_{split}.parquet",
                                        config.POOLS / f"eval_{split}_users.parquet", args.k)
    _, vc_cos, vc_lab, vc_off, vc_rt, _ = pools["val_cold"]
    cold_cosine = eval_pool(vc_cos, vc_lab, vc_off, vc_rt, KS, pooled=True)

    for mp in models:
        run = run_label(mp)
        df, cos, lab, off, rt, users = pools["val"]
        pred = predict(mp, df, users)
        best_a, best_m, all_m = sweep_best_alpha(cos, pred, lab, off, rt, KS, config.ALPHAS)
        base = all_m[0.0]

        cdf, ccos, clab, coff, crt, cusers = pools["val_cold"]
        cpred = predict(mp, cdf, cusers)
        cold_m = eval_pool(blend(ccos, cpred, coff, best_a), clab, coff, crt, KS, pooled=True)

        sweep_rows = "\n".join(
            f"alpha={a:<5} ndcg@10={m['ndcg@10']:.4f}  recall@10={m['recall@10']:.4f}  "
            f"recall@100={m['recall@100']:.4f}  ndcg@100={m['ndcg@100']:.4f}"
            + ("   <- best" if a == best_a else "")
            for a, m in all_m.items())

        text = f"""# Ranker {run} — two-stage rerank pool K={args.k} — protocol v2 (mask seen−query, pool từ artifacts hiện tại)
# generated {datetime.now(timezone.utc).isoformat(timespec='seconds')}  model={mp.name}
# blend: score = (1-α)·rank_norm(cos_uv) + α·rank_norm(pred), chọn α theo val ndcg@10
# KỶ LUẬT: test warm chấm qua eval.py khi CHỐT model; test_cold = final exam (--final-exam). File này chỉ VAL.

## sweep α (val warm)
alpha=0 = cosine baseline (retriever-only)
{sweep_rows}

## warm (val) @ α={best_a}  (users: {best_m['n_users']:,})
{fmt_block(best_m, KS)}
## warm (val) cosine baseline — để so
{fmt_block(base, KS)}
## cold (val_cold) @ blend α={best_a}  (users: {cold_m['n_users']:,}, pairs: {cold_m['n_pairs']:,})
# ⚠ chỉ là diagnostic: serving KHÔNG đưa cold qua ranker (cold_serving = tách kênh, docs/RANKER.md §7)
{fmt_block(cold_m, KS, pooled=True)}
## cold (val_cold) kênh serve thật = cosine retriever  (users: {cold_cosine['n_users']:,})
{fmt_block(cold_cosine, KS, pooled=True)}"""
        out = mp.parent / "results.txt"
        out.write_text(text)
        print(f"[{run}] α={best_a} val ndcg@10={best_m['ndcg@10']:.4f} "
              f"(cosine {base['ndcg@10']:.4f}) cold-blend {cold_m['ndcg@10']:.4f} -> {out}")


if __name__ == "__main__":
    main()
