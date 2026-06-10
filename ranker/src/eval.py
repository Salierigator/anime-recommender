"""eval.py — two-stage cold-by-user: sweep objective × blend-α, chọn cấu hình Pareto vs cosine.

Mỗi user (val/test, cold): hold-out query, dựng U từ support → top-200 cosine → rerank. So sánh:
  - baseline = thứ tự cosine (α=0).
  - blend = (1-α)·rank_norm(cos_uv) + α·rank_norm(pred_model), α∈{.3,.5,.7,1} × {lambdarank30/200,xendcg}.
Đo recall@{10,50,100} / ndcg@{10,50} + trần pool (tỉ lệ query lọt top-200). Chọn (objective,α)
Pareto-dominate cosine TRÊN VAL → ghi artifacts/ranker.txt + ranker_meta.json; báo cáo test.

QUAN TRỌNG: import torch (user_encode) TRƯỚC lightgbm (tránh segfault 2 OpenMP runtime trên mac).
"""
from __future__ import annotations

import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from features import ItemFeatures, K_RETRIEVE, build_frame, CAT_COLS, FEATURE_NAMES
from user_encode import ARTIFACTS, ROOT, encode_gender_joined, load_user_encoder
from build_dataset import CLEANED, account_age, stream_ratings

import lightgbm as lgb                                            # sau torch

DATA = ROOT / "data"
SEED = 42
K_HISTORY = 30
N_EVAL_USERS = 3_000
CHUNK = 2_000
KS = [10, 50, 100]
MODELS = ["lambdarank30", "lambdarank200", "xendcg"]
ALPHAS = [0.3, 0.5, 0.7, 1.0]
METRIC_ORDER = [f"{m}@{k}" for m in ["recall", "ndcg"] for k in KS]


def rank_norm(x: np.ndarray) -> np.ndarray:
    """Thứ hạng chuẩn hoá [0,1] (giá trị lớn → gần 1). Ties phá tuỳ ý (đủ cho blend)."""
    r = np.empty(len(x), dtype=np.float64)
    r[np.argsort(x)] = np.arange(len(x))
    return r / max(len(x) - 1, 1)


def metrics_row(ranked: np.ndarray, relevant: set) -> dict:
    R = len(relevant)
    hit = np.array([a in relevant for a in ranked], dtype=np.float64)
    disc = 1.0 / np.log2(np.arange(2, len(ranked) + 2))
    out = {}
    for k in KS:
        h = hit[:k]
        out[f"recall@{k}"] = h.sum() / R
        out[f"ndcg@{k}"] = (h * disc[:k]).sum() / disc[:min(R, k)].sum()
    return out


def collect_records(split: str, enc, meta, itemfeat, boosters, V, rng) -> tuple[list, float]:
    """Per user: cos[200], pred[model][200], cand[200], query set. Trả (records, trần pool@200)."""
    import polars as pl
    users = (pl.read_parquet(ARTIFACTS / "user_split.parquet")
             .filter(pl.col("split") == split)["username"].to_list())
    rng.shuffle(users)
    users = users[:N_EVAL_USERS]
    base = stream_ratings(users)
    base_g = {u: g for u, g in base.groupby("username", sort=False)}
    age_map = account_age(users)
    guf = encode_gender_joined(
        pd.read_csv(CLEANED / "profiles.csv", usecols=["username", "gender", "joined"]), meta).set_index("username")
    newest = len(meta["user_features"]["joined"]["bins"]) - 2

    recs = []
    for u in users:
        g = base_g.get(u)
        if g is None:
            continue
        pos = g[g["is_pos"].to_numpy()]
        p_idx = pos["anime_idx"].to_numpy(); p_score = pos["score"].to_numpy()
        if len(p_idx) < 2:
            continue
        tie = rng.random(len(p_idx))
        n_query = int(np.clip(round(0.2 * len(p_idx)), 1, len(p_idx) - 1))
        q_local = np.argsort(tie)[:n_query]
        is_q = np.zeros(len(p_idx), bool); is_q[q_local] = True
        s_local = np.where(~is_q)[0]
        order = sorted(s_local, key=lambda i: (-p_score[i], tie[i]))[:K_HISTORY]
        hist_idx = p_idx[order]; hist_score = p_score[order]
        if len(hist_idx) == 0:
            continue
        rated = hist_score[hist_score >= 1]
        recs.append({
            "hist_idx": hist_idx, "hist_score": hist_score,
            "query": set(p_idx[q_local].tolist()),
            "g_pref": itemfeat.genres[hist_idx].mean(0), "t_pref": itemfeat.themes[hist_idx].mean(0),
            "gender_id": int(guf.loc[u, "gender_id"]) if u in guf.index else 0,
            "joined_bucket": int(guf.loc[u, "joined_bucket"]) if u in guf.index else newest,
            "u_n_rated": float(len(rated)),
            "u_mean_score": float(rated.mean()) if len(rated) else 0.0,
            "u_std_score": float(rated.std()) if len(rated) else 0.0,
            "u_age": float(age_map.get(u, np.nan)),
        })
    age_med = np.nanmedian([r["u_age"] for r in recs])

    out, pool_hit = [], []
    pad = lambda arr, L: np.pad(arr, (0, L - len(arr)))[:L]
    for s in range(0, len(recs), CHUNK):
        chunk = recs[s:s + CHUNK]
        hid = torch.tensor(np.stack([pad(r["hist_idx"], K_HISTORY) for r in chunk]), dtype=torch.long)
        hsc = torch.tensor(np.stack([pad(r["hist_score"], K_HISTORY) for r in chunk]), dtype=torch.long)
        gid = torch.tensor([r["gender_id"] for r in chunk], dtype=torch.long)
        jb = torch.tensor([r["joined_bucket"] for r in chunk], dtype=torch.long)
        U = enc.encode(hid, hid != 0, hsc, gid, jb)
        scores = (U @ enc.item_cache.t()).numpy()
        for ci, r in enumerate(chunk):
            srow = scores[ci]
            masked = srow.copy(); masked[r["hist_idx"]] = -np.inf; masked[:2] = -np.inf
            cand = np.argpartition(masked, -K_RETRIEVE)[-K_RETRIEVE:]
            hv = V[r["hist_idx"]]; sims = V[cand] @ hv.T
            ga, ta, go = itemfeat.affinity(cand, r["g_pref"], r["t_pref"])
            nc = len(cand)
            cross = {
                "cos_uv": srow[cand], "hist_cos_max": sims.max(1), "hist_cos_mean": sims.mean(1),
                "genre_aff": ga, "theme_aff": ta, "genre_overlap": go,
                "u_n_rated": np.full(nc, r["u_n_rated"]), "u_mean_score": np.full(nc, r["u_mean_score"]),
                "u_std_score": np.full(nc, r["u_std_score"]),
                "u_account_age": np.full(nc, r["u_age"] if not np.isnan(r["u_age"]) else age_med),
            }
            X = build_frame(itemfeat, cand, cross)
            preds = {m: boosters[m].predict(X) for m in MODELS}
            out.append({"cos": srow[cand], "preds": preds, "cand": cand, "query": r["query"]})
            pool_hit.append(len(r["query"] & set(cand.tolist())) / len(r["query"]))
    return out, float(np.mean(pool_hit))


def score_config(rec: dict, model: str | None, alpha: float) -> np.ndarray:
    if model is None:
        return rec["cos"]
    return (1 - alpha) * rank_norm(rec["cos"]) + alpha * rank_norm(rec["preds"][model])


def eval_config(records: list, model, alpha) -> dict:
    acc = {m: 0.0 for m in METRIC_ORDER}
    for rec in records:
        ranked = rec["cand"][np.argsort(-score_config(rec, model, alpha))]
        mr = metrics_row(ranked, rec["query"])
        for m in METRIC_ORDER:
            acc[m] += mr[m]
    return {m: v / len(records) for m, v in acc.items()}


def report(split: str, records: list, ceiling: float) -> dict:
    base = eval_config(records, None, 0.0)
    print(f"\n=== {split} (n={len(records)}, pool@200 recall ceiling={ceiling:.3f}) ===")
    print(f"  {'config':<20}" + "".join(f"{m:>12}" for m in METRIC_ORDER))
    print(f"  {'cosine(baseline)':<20}" + "".join(f"{base[m]:>12.4f}" for m in METRIC_ORDER))
    results = {"cosine": base}
    for model in MODELS:
        for a in ALPHAS:
            r = eval_config(records, model, a)
            results[f"{model}|a{a}"] = r
            flags = "".join(f"{r[m]:>12.4f}" for m in METRIC_ORDER)
            print(f"  {model + '|a' + str(a):<20}{flags}")
    return results


def pareto_pick(val: dict) -> tuple[str, dict]:
    """Chọn config (≥ cosine mọi metric, ndcg@10 strict >) max ndcg@10; fallback max ndcg@10 s.t. recall@50≥cosine."""
    base = val["cosine"]
    cands = {k: v for k, v in val.items() if k != "cosine"}
    pareto = {k: v for k, v in cands.items()
              if all(v[m] >= base[m] - 1e-9 for m in METRIC_ORDER) and v["ndcg@10"] > base["ndcg@10"]}
    pool = pareto or {k: v for k, v in cands.items() if v["recall@50"] >= base["recall@50"] - 1e-9}
    pool = pool or cands
    best = max(pool, key=lambda k: pool[k]["ndcg@10"])
    return best, {"pareto": bool(pareto)}


def main() -> None:
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    enc, meta = load_user_encoder("cpu")
    V = enc.item_cache.numpy()
    itemfeat = ItemFeatures.load(ARTIFACTS, CLEANED)
    boosters = {m: lgb.Booster(model_file=str(DATA / f"ranker_{m}.txt")) for m in MODELS}

    val_rec, val_ceil = collect_records("val", enc, meta, itemfeat, boosters, V, rng)
    val_res = report("val", val_rec, val_ceil)
    test_rec, test_ceil = collect_records("test", enc, meta, itemfeat, boosters, V, rng)
    test_res = report("test", test_rec, test_ceil)

    best, info = pareto_pick(val_res)
    model, alpha = best.split("|a")
    alpha = float(alpha)
    print(f"\n>>> CHỌN (theo val): {best}  [{'Pareto-dominate cosine' if info['pareto'] else 'fallback: recall@50≥cosine'}]")
    print(f"    test {best}: " + "  ".join(f"{m}={test_res[best][m]:.4f}(Δ{test_res[best][m]-test_res['cosine'][m]:+.4f})" for m in METRIC_ORDER))

    shutil.copy(DATA / f"ranker_{model}.txt", ARTIFACTS / "ranker.txt")
    meta_out = {
        "feature_names": FEATURE_NAMES, "categorical_features": CAT_COLS,
        "k_retrieve": K_RETRIEVE, "objective": model, "blend_alpha": alpha,
        "blend": "score = (1-alpha)*rank_norm(cos_uv) + alpha*rank_norm(ranker_pred)",
        "val_metrics": val_res[best], "test_metrics": test_res[best],
        "baseline_test": test_res["cosine"], "pool_recall_ceiling_test": test_ceil,
        "source_checkpoint": next((l for l in (ARTIFACTS / "CONTRACT.md").read_text().splitlines()
                                   if "Source checkpoint" in l), "unknown").strip("- "),
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (ARTIFACTS / "ranker_meta.json").write_text(json.dumps(meta_out, indent=2, ensure_ascii=False))
    print(f"saved artifacts/ranker.txt ({model}, α={alpha}) + ranker_meta.json  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
