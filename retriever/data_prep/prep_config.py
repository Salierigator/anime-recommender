"""Config chung cho data_prep v2 — nguồn DUY NHẤT cho seed/caps/labels/cold-item.

02/03/05/06 import từ đây (hết drift hardcode rải 3 script như v1); 06 echo vào
feature_spec.json để train/serve đọc lại đúng định nghĩa đã build.

Đổi labeling để ablate: sửa POS_STATUSES / is_hardneg_expr ở đây rồi re-run 02->06
(02 đổi n_pos -> split membership đổi theo, chấp nhận).
"""
import polars as pl

SEED = 42
EVAL_MIN_POS = 11        # eligibility eval (03): n_pos >= 11 mới được vào val/test
EVAL_QUERY_FRAC = 0.2    # tỉ lệ warm positive của eval user tách làm query
HARD_NEG_CAP = 64        # cap hard_neg_ids lưu mỗi user
COLD_FRAC = 0.05         # tỉ lệ anime mới nhất (theo start_date) vào tập cold H (01)
EVAL_HISTORY_CAP = 1024  # default cap prefix history lúc eval (echo vào spec; src override được)

# --- labels v2 (positive/hard-neg RỜI nhau by construction: pos cần score ∉ [1,4]) ---
POS_STATUSES = ["completed", "watching"]


def _score():
    return pl.col("score").cast(pl.Int64, strict=False)


def is_pos_expr():
    """positive = status ∈ POS_STATUSES & score ∉ [1,4] (giữ score==0 và 5..10)."""
    return pl.col("status").is_in(POS_STATUSES) & ~_score().is_between(1, 4)


def is_hardneg_expr():
    """hard-neg = dropped ∪ (score ∈ [1,4] mọi status) — 'bỏ dở' + 'xem và ghét'."""
    return (pl.col("status") == "dropped") | _score().is_between(1, 4)


# echo vào feature_spec.json (06) — mô tả human-readable, đổi expr thì đổi cả đây
LABELS_SPEC = {
    "positive": "status in {completed, watching} & score not in [1,4]",
    "pos_statuses": POS_STATUSES,
    "hard_negative": "status == dropped | score in [1,4] (any status)",
}
