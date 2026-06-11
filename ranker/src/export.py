"""export.py (ranker) — chốt winner → artifacts/ranker.txt + ranker_meta.json (CHỈ 2 file này).

Đọc ranker/data/eval_selection.json (eval.py ghi sau blend sweep + Pareto select). Winner phải
là LightGBM (.txt) — contract serve; neural thắng thì vẫn chỉ ghi nhận trên leaderboard
(đổi serving contract = quyết định riêng, không export ở đây).

    venv/bin/python ranker/src/export.py
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import torch  # noqa: F401  (trước lightgbm)
import lightgbm as lgb

import config
from features import CAT_COLS, FEATURE_NAMES


def main() -> None:
    sel = json.loads((config.DATA / "eval_selection.json").read_text())
    model_path = Path(sel["model"])
    if model_path.suffix != ".txt":
        raise SystemExit(
            f"winner = {model_path.name} (không phải LightGBM .txt) — contract serve là "
            "LightGBM; nếu muốn ship neural phải đổi contract service trước.")

    booster = lgb.Booster(model_file=str(model_path))
    assert booster.feature_name() == FEATURE_NAMES, "model feature order lệch FEATURE_NAMES"
    shutil.copyfile(model_path, config.ARTIFACTS / "ranker.txt")

    bmeta = json.loads((config.DATASETS / "build_meta.json").read_text())
    meta = {
        "model_type": "lightgbm", "model_run": sel["model_name"],
        "best_iteration": booster.num_trees(),
        "feature_names": FEATURE_NAMES, "categorical_features": CAT_COLS,
        "grading": config.GRADE_MAP,
        "k_retrieve": sel["k_retrieve"], "blend_alpha": sel["blend_alpha"],
        "blend": "score = (1-alpha)*rank_norm(cos_uv) + alpha*rank_norm(ranker_pred)",
        "cold_feature_policy": "is_cold -> mal_score/scored_by/members/favorites/popularity/"
                               "rank impute-as-missing + flag (features.py)",
        "hist_feat_cap": config.HIST_FEAT_CAP, "eval_history_cap": 1024,
        "pareto_vs_cosine": sel["pareto"],
        "val_metrics": sel["val_metrics"], "baseline_val": sel["baseline_val"],
        "test_metrics": sel["test_metrics"], "baseline_test": sel["baseline_test"],
        "val_cold_metrics": sel["val_cold_metrics"],
        "baseline_val_cold": sel["baseline_val_cold"],
        "pool_ceiling": sel["pool_ceiling"],
        "feature_importance_gain": dict(zip(
            FEATURE_NAMES, booster.feature_importance("gain").round(1).tolist())),
        "train_provenance": {k: bmeta[k] for k in
                             ("n_groups_kept", "rows", "k_pool", "pos_rate", "seed",
                              "git_rev", "source_checkpoint")},
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (config.ARTIFACTS / "ranker_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"exported artifacts/ranker.txt ({sel['model_name']}, α={sel['blend_alpha']}, "
          f"K={sel['k_retrieve']}) + ranker_meta.json")
    print(f"  val  ndcg@10 {sel['baseline_val']['ndcg@10']:.4f} -> "
          f"{sel['val_metrics']['ndcg@10']:.4f}")
    print(f"  test ndcg@10 {sel['baseline_test']['ndcg@10']:.4f} -> "
          f"{sel['test_metrics']['ndcg@10']:.4f}")


if __name__ == "__main__":
    main()
