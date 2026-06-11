"""eval.py — two-stage eval trên pool đã build (build_eval.py) + chọn cấu hình.

Modes:
  --baseline-only   SANITY GATE: xếp pool theo cos_uv → metrics PHẢI khớp
                    artifacts/eval_reference.json (~1e-3) — cùng encoder/mask/queries với
                    retriever. Fail = harness lệch protocol, KHÔNG được train tiếp.
  (default)         Load model trong ranker/models/ (*.txt LightGBM, *.pt neural) →
                    sweep blend α trên VAL → Pareto-select vs cosine → report TEST + val_cold
                    → ghi ranker/models/eval_selection.json (export.py đọc).
  --k 500           Ablation pool depth (slice 200 mặc định từ pool sâu 500).
  --final-exam      test_cold (chấm 1 LẦN lúc chốt pipeline): cần export.py --final-exam trước.

Selection (VAL only): Pareto ≥ cosine trên {r@10, r@100, ndcg@10, ndcg@100} + ndcg@10 strict >
→ max ndcg@10; fallback max ndcg@10 s.t. r@100 ≥ cosine. (r@200 = trần pool khi K=200, bỏ.)

QUAN TRỌNG: import torch (qua pool/user_encode) TRƯỚC lightgbm — segfault 2 OpenMP runtime/mac.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch  # noqa: F401  (trước lightgbm)

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))   # lib chung ở src/

import config  # noqa: E402
from features import FEATURE_NAMES  # noqa: E402
from metrics import blend, eval_pool, fmt  # noqa: E402

SEL_METRICS = ["recall@10", "recall@100", "ndcg@10", "ndcg@100"]


def run_label(p: Path) -> str:
    """models/<run_name>/model.txt -> run_name (trainer ghi theo run subdir)."""
    return p.parent.name if p.stem == "model" else p.stem


def load_pool(split: str, k: int):
    from metrics import load_pool_arrays
    return load_pool_arrays(config.POOLS / f"eval_{split}.parquet",
                            config.POOLS / f"eval_{split}_users.parquet", k)


def predict(model_path: Path, df: pl.DataFrame, users: pl.DataFrame) -> np.ndarray:
    """Score pool bằng model bất kỳ: .txt = LightGBM booster, .pt = neural (train_nn),
    .npz = linear baseline."""
    if model_path.suffix == ".txt":
        import lightgbm as lgb
        booster = lgb.Booster(model_file=str(model_path))
        X = df.select(FEATURE_NAMES).to_pandas()
        return booster.predict(X)
    if model_path.suffix == ".pt":
        from train_nn import predict_pool
        return predict_pool(model_path, df, users)
    if model_path.suffix == ".npz":
        z = np.load(model_path, allow_pickle=True)
        feats = list(z["features"])
        X = df.select(feats).to_numpy().astype(np.float64)
        X = np.where(np.isnan(X), z["mean"], X)
        return (X - z["mean"]) / z["std"] @ z["coef"] + float(z["intercept"])
    raise SystemExit(f"model không nhận dạng được: {model_path}")


def sweep(cos, pred, labels, offsets, r_total, ks, alphas):
    """{alpha: metrics} cho 1 model. alpha=0 = cosine baseline."""
    return {a: eval_pool(blend(cos, pred, offsets, a), labels, offsets, r_total, ks)
            for a in alphas}


def pareto_pick(results: dict[tuple, dict], base: dict) -> tuple | None:
    """results: {(model_name, alpha): metrics}. Pareto ≥ base mọi SEL_METRICS, ndcg@10 strict."""
    pareto = {k: v for k, v in results.items()
              if all(v[m] >= base[m] - 1e-9 for m in SEL_METRICS)
              and v["ndcg@10"] > base["ndcg@10"]}
    pool = pareto or {k: v for k, v in results.items()
                      if v["recall@100"] >= base["recall@100"] - 1e-9}
    if not pool:
        return None
    return max(pool, key=lambda k: pool[k]["ndcg@10"]), bool(pareto)


def baseline_gate() -> None:
    """Cosine-baseline pool phải tái lập eval_reference.json (đo qua artifacts) ~1e-3."""
    ref = json.loads((config.ARTIFACTS / "eval_reference.json").read_text())
    ks_full = [10, 50, 100, 200, 500]
    ok = True
    for split, key in (("val", "val_warm"), ("test", "test_warm"), ("val_cold", "val_cold")):
        if not (config.POOLS / f"eval_{split}.parquet").exists():
            print(f"[skip] eval_{split} chưa build")
            continue
        _, cos, labels, offsets, r_total, _ = load_pool(split, config.POOL_DEPTH)
        m = eval_pool(cos, labels, offsets, r_total, ks_full, pooled=split.endswith("cold"))
        print(f"[{split:8s}] {fmt(m, ks_full)}")
        for k in ks_full:
            d = abs(m[f"recall@{k}"] - ref[key][f"recall@{k}"])
            if d > 2e-3:
                print(f"  ✗ recall@{k}: pool {m[f'recall@{k}']:.4f} vs ref "
                      f"{ref[key][f'recall@{k}']:.4f} (Δ{d:.4f})")
                ok = False
        assert m["n_users"] == ref[key]["n_users"], \
            f"{split}: n_users {m['n_users']} != ref {ref[key]['n_users']}"
    if not ok:
        raise SystemExit("✗ SANITY GATE FAIL — protocol lệch retriever, dừng (đừng train).")
    print("✓ sanity gate PASS — cosine baseline tái lập eval_reference (encoder/mask/queries khớp)")


def report_split(split: str, k: int, models: list[Path], alphas, ks):
    df, cos, labels, offsets, r_total, users = load_pool(split, k)
    base = eval_pool(cos, labels, offsets, r_total, ks, pooled=split.endswith("cold"))
    print(f"\n=== {split} (K={k}, n={base['n_users']:,}) ===")
    print(f"  {'config':<28}" + "".join(f"{m:>12}" for m in SEL_METRICS))
    print(f"  {'cosine(baseline)':<28}" + "".join(f"{base[m]:>12.4f}" for m in SEL_METRICS))
    results = {}
    for mp in models:
        pred = predict(mp, df, users)
        for a, m in sweep(cos, pred, labels, offsets, r_total, ks, alphas).items():
            if a == 0.0:
                continue
            results[(run_label(mp), a)] = m
            print(f"  {f'{run_label(mp)}|a{a}':<28}"
                  + "".join(f"{m[x]:>12.4f}" for x in SEL_METRICS))
    return base, results


def main() -> None:
    ap = argparse.ArgumentParser(description="Two-stage eval + selection")
    ap.add_argument("--baseline-only", action="store_true")
    ap.add_argument("--k", type=int, default=config.K_POOL, choices=[200, 500])
    ap.add_argument("--models", nargs="*", type=Path,
                    help="default: mọi *.txt/*.pt trong ranker/models/")
    ap.add_argument("--final-exam", action="store_true",
                    help="chấm test_cold 1 LẦN (cần export.py --final-exam + selection có sẵn)")
    args = ap.parse_args()
    t0 = time.time()

    baseline_gate()
    if args.baseline_only:
        return

    if args.final_exam:
        run_final_exam()
        return

    models = args.models or sorted(
        p for pat in ("*/model.txt", "*/model.pt", "*/model.npz", "*.txt", "*.pt", "*.npz")
        for p in config.MODELS.glob(pat))
    if not models:
        raise SystemExit(f"không có model trong {config.MODELS} — tải winner từ Drive về trước")
    print(f"models: {[run_label(m) for m in models]}")

    base_val, val_res = report_split("val", args.k, models, config.ALPHAS, config.KS)
    picked = pareto_pick(val_res, base_val)
    if picked is None:
        raise SystemExit("không config nào ≥ cosine — kiểm tra model/feature")
    (model_name, alpha), is_pareto = picked
    print(f"\n>>> CHỌN (val): {model_name} α={alpha} "
          f"[{'Pareto-dominate cosine' if is_pareto else 'fallback r@100≥cosine'}]")

    chosen = next(m for m in models if run_label(m) == model_name)
    base_test, test_res = report_split("test", args.k, [chosen], [alpha], config.KS)
    test_m = test_res[(model_name, alpha)]
    print("    test Δ vs cosine: " + "  ".join(
        f"{m}={test_m[m]:.4f}({test_m[m] - base_test[m]:+.4f})" for m in SEL_METRICS))

    base_cold, cold_res = report_split("val_cold", args.k, [chosen], [alpha], config.KS)
    cold_m = cold_res[(model_name, alpha)]

    sel = {
        "model": str(chosen), "model_name": model_name, "blend_alpha": alpha,
        "k_retrieve": args.k, "pareto": is_pareto,
        "val_metrics": val_res[(model_name, alpha)], "baseline_val": base_val,
        "test_metrics": test_m, "baseline_test": base_test,
        "val_cold_metrics": cold_m, "baseline_val_cold": base_cold,
        "pool_ceiling": {"val": base_val[f"recall@{args.k}"],
                         "test": base_test[f"recall@{args.k}"]},
    }
    config.MODELS.mkdir(exist_ok=True)
    (config.MODELS / "eval_selection.json").write_text(json.dumps(sel, indent=2))
    print(f"\nselection -> {config.MODELS / 'eval_selection.json'}  ({time.time() - t0:.0f}s)")


def run_final_exam() -> None:
    """FINAL EXAM — test_cold, chấm đúng 1 lần với cấu hình đã chốt trong eval_selection.json."""
    assert (config.ARTIFACTS / "eval_queries_test_cold.parquet").exists(), \
        "cần `venv/bin/python retriever/export.py --final-exam` trước"
    sel = json.loads((config.MODELS / "eval_selection.json").read_text())
    if not (config.POOLS / "eval_test_cold.parquet").exists():
        import build_eval
        from features import ItemFeatures
        from pool import UsersHistory, account_age_by_user, load_eval_seen
        from user_encode import load_user_encoder
        enc, meta = load_user_encoder("cpu")
        itemfeat = ItemFeatures.load(config.ARTIFACTS, config.CLEANED)
        build_eval.build_split("test_cold", enc, meta.get("eval_history_cap", 1024), itemfeat,
                               enc.item_cache.numpy(), UsersHistory(), load_eval_seen(),
                               account_age_by_user())
    k, alpha = sel["k_retrieve"], sel["blend_alpha"]
    df, cos, labels, offsets, r_total, users = load_pool("test_cold", k)
    base = eval_pool(cos, labels, offsets, r_total, config.KS, pooled=True)
    pred = predict(Path(sel["model"]), df, users)
    m = eval_pool(blend(cos, pred, offsets, alpha), labels, offsets, r_total,
                  config.KS, pooled=True)
    print(f"\n=== FINAL EXAM test_cold (K={k}, α={alpha}, model={sel['model_name']}) ===")
    print(f"  cosine : {fmt(base, config.KS)}")
    print(f"  ranker : {fmt(m, config.KS)}")
    sel["test_cold_metrics"] = m
    sel["baseline_test_cold"] = base
    (config.MODELS / "eval_selection.json").write_text(json.dumps(sel, indent=2))


if __name__ == "__main__":
    main()
