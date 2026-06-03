"""Audit users có MAX score <= 5 trong ratings.csv (bỏ qua score==0).

Câu hỏi:
  A. Có bao nhiêu user chỉ chấm điểm từ 1-5 (không bao giờ chấm >= 6)?
  B. Phân bố số lượng rating (score >= 1) của nhóm này vs toàn bộ user.
  C. Phân bố score của nhóm low-rater.
  D. Status breakdown của nhóm này.

ratings.csv ~3.2GB => polars lazy + streaming, KHÔNG collect full.

Usage:
    python scripts/ratings_audit/audit_low_raters.py
"""
import pathlib

import polars as pl

ROOT = pathlib.Path(__file__).parent.parent.parent
SRC = ROOT / "cleaned-data" / "ratings.csv"

LOW_MAX = 5  # ngưỡng: max_score <= này => "low rater"

lf = pl.scan_csv(SRC).with_columns(pl.col("score").cast(pl.Int64, strict=False))

# Chỉ xét các dòng có score >= 1 (bỏ score == 0)
lf_scored = lf.filter(pl.col("score") >= 1)

# ----------------------------------------------------------------------
# PASS 1 — per-user max score + tổng số rating đã chấm
# ----------------------------------------------------------------------
per_user = (
    lf_scored.group_by("username")
    .agg(
        pl.col("score").max().alias("max_score"),
        pl.len().alias("n_rated"),
    )
    .collect(engine="streaming")
)

n_users = len(per_user)
low_users = per_user.filter(pl.col("max_score") <= LOW_MAX)
n_low = len(low_users)

print(f"Tổng user có ít nhất 1 rating (score>=1): {n_users:,}")
print(f"User chỉ chấm <= {LOW_MAX}:                {n_low:,}  ({n_low / n_users * 100:.3f}%)\n")

# ---- A. Low-rater count -----------------------------------------------
print("=" * 60)
print(f"A. USER CHỈ CHẤM <= {LOW_MAX} (không bao giờ score > {LOW_MAX})")
print("=" * 60)
print(f"  {n_low:,} / {n_users:,} user  ({n_low / n_users * 100:.3f}%)")

# ---- B. Phân bố n_rated: low vs toàn bộ ------------------------------
print("\n" + "=" * 60)
print("B. N_RATED DISTRIBUTION — low-rater vs toàn bộ user")
print("=" * 60)
QS = [0.10, 0.25, 0.50, 0.75, 0.90]
print(f"{'':>20}{'mean':>8}  " + "  ".join(f"p{int(q*100):>2}" for q in QS))
for label, df in [("low-rater", low_users), ("tất cả", per_user)]:
    col = df["n_rated"]
    qv = [col.quantile(q) for q in QS]
    print(f"  {label:<18}{col.mean():>8.1f}  " + "  ".join(f"{v:>5.0f}" for v in qv))

# ---- C. Score distribution của low-rater ------------------------------
print("\n" + "=" * 60)
print(f"C. SCORE DISTRIBUTION — chỉ trong nhóm low-rater (score 1-{LOW_MAX})")
print("=" * 60)

low_usernames = low_users["username"]

# Lọc theo danh sách username low-rater, lấy joint distribution score
low_joint = (
    lf_scored.filter(pl.col("username").is_in(low_usernames))
    .group_by("score")
    .agg(pl.len().alias("n"))
    .collect(engine="streaming")
    .sort("score")
)

total_low_ratings = int(low_joint["n"].sum())
print(f"  Tổng số rating của nhóm low-rater: {total_low_ratings:,}")
for row in low_joint.iter_rows(named=True):
    print(f"  score {row['score']:>3}: {row['n']:>10,}  ({row['n'] / total_low_ratings * 100:6.2f}%)")

# ---- D. Status breakdown của low-rater --------------------------------
print("\n" + "=" * 60)
print("D. STATUS BREAKDOWN — nhóm low-rater")
print("=" * 60)

status_dist = (
    lf_scored.filter(pl.col("username").is_in(low_usernames))
    .group_by("status")
    .agg(pl.len().alias("n"))
    .collect(engine="streaming")
    .sort("n", descending=True)
)

for row in status_dist.iter_rows(named=True):
    st = row["status"] if row["status"] is not None else "<null>"
    print(f"  {row['n']:>10,}  ({row['n'] / total_low_ratings * 100:6.2f}%)  {st!r}")
