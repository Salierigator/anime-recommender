"""Audit `status` + `score` (ratings.csv) — phục vụ quyết định positive pool.

Câu hỏi cần trả lời bằng số (không đoán):
  A. status: count + % mỗi giá trị trên tổng số dòng.
  B. score: phân bố từng giá trị; score==0 chiếm bao nhiêu (giả thuyết: "unrated").
  C. status × score: score có tồn tại đầy đủ ở các status NGOÀI completed,
     hay phần lớn chỉ completed mới có điểm? Trong completed có bao nhiêu % bị
     score==0? => quyết định score==0 có thực sự là "chưa chấm" không.
  D. positive pool / user: nếu positive = score>=7, mỗi user thực sự có bao nhiêu
     positive khi (i) chỉ tính completed vs (ii) tính mọi status. Chênh lệch =
     mức lợi ích của việc mở rộng status / score-weighting.

ratings.csv ~3.2GB => polars lazy + streaming, KHÔNG collect full.

Usage:
    python scripts/ratings_audit/audit_status_score.py
"""
import pathlib

import polars as pl

ROOT = pathlib.Path(__file__).parent.parent.parent.parent.parent
SRC = ROOT / "cleaned-data" / "ratings.csv"

POSITIVE = 7  # score>=7 coi là "positive"

# ----------------------------------------------------------------------
# PASS 1 — joint distribution (status × score). Kết quả nhỏ => suy ra A,B,C.
# ----------------------------------------------------------------------
lf = pl.scan_csv(SRC).with_columns(pl.col("score").cast(pl.Int64, strict=False))

joint = (
    lf.group_by("status", "score")
    .agg(pl.len().alias("n"))
    .collect(engine="streaming")
)

total = int(joint["n"].sum())
print(f"Total ratings rows: {total:,}\n")

# ---- A. STATUS distribution -----------------------------------------
print("=" * 60)
print("A. STATUS — count + % trên tổng số dòng")
print("=" * 60)
status_dist = (
    joint.group_by("status").agg(pl.col("n").sum().alias("n")).sort("n", descending=True)
)
for row in status_dist.iter_rows(named=True):
    st = row["status"] if row["status"] is not None else "<null>"
    print(f"  {row['n']:>13,}  ({row['n'] / total * 100:6.3f}%)  {st!r}")

# ---- B. SCORE distribution ------------------------------------------
print("\n" + "=" * 60)
print("B. SCORE — phân bố từng giá trị (score==0 = 'unrated'?)")
print("=" * 60)
score_dist = (
    joint.group_by("score").agg(pl.col("n").sum().alias("n")).sort("score", nulls_last=True)
)
for row in score_dist.iter_rows(named=True):
    sc = row["score"] if row["score"] is not None else "<null>"
    print(f"  score {str(sc):>6}:  {row['n']:>13,}  ({row['n'] / total * 100:6.3f}%)")

n_zero = int(joint.filter(pl.col("score") == 0)["n"].sum())
n_null = int(joint.filter(pl.col("score").is_null())["n"].sum())
n_scored = total - n_zero - n_null  # score>=1
n_pos = int(joint.filter(pl.col("score") >= POSITIVE)["n"].sum())
print(f"\n  score==0      : {n_zero:>13,}  ({n_zero / total * 100:6.3f}%)")
print(f"  score==null   : {n_null:>13,}  ({n_null / total * 100:6.3f}%)")
print(f"  score>=1 (rated): {n_scored:>11,}  ({n_scored / total * 100:6.3f}%)")
print(f"  score>={POSITIVE} (positive): {n_pos:>9,}  "
      f"({n_pos / total * 100:6.3f}% tổng | {n_pos / max(n_scored, 1) * 100:6.3f}% trong số đã chấm)")

# ---- C. STATUS × SCORE ----------------------------------------------
# Với mỗi status: %score==0, %đã-chấm, %positive => score có ngoài completed?
print("\n" + "=" * 60)
print("C. STATUS × SCORE — score có đầy đủ ngoài completed không?")
print("=" * 60)
print(f"{'status':<16}{'n':>13}{'%zero':>9}{'%rated':>9}{'%pos≥'+str(POSITIVE):>9}"
      f"{'mean(rated)':>13}")
zero = pl.col("score") == 0
rated = pl.col("score") >= 1
posit = pl.col("score") >= POSITIVE
per_status = (
    joint.group_by("status")
    .agg(
        pl.col("n").sum().alias("n"),
        pl.col("n").filter(zero).sum().alias("n_zero"),
        pl.col("n").filter(rated).sum().alias("n_rated"),
        pl.col("n").filter(posit).sum().alias("n_pos"),
        # mean score trên các dòng đã chấm (score>=1)
        (pl.col("score") * pl.col("n")).filter(rated).sum().alias("score_sum_rated"),
    )
    .sort("n", descending=True)
)
for row in per_status.iter_rows(named=True):
    st = row["status"] if row["status"] is not None else "<null>"
    n = row["n"]
    n_rated_s = row["n_rated"] or 0
    mean_rated = (row["score_sum_rated"] / n_rated_s) if n_rated_s else float("nan")
    print(f"{st:<16}{n:>13,}{(row['n_zero'] or 0) / n * 100:>8.2f}%"
          f"{n_rated_s / n * 100:>8.2f}%{(row['n_pos'] or 0) / n * 100:>8.2f}%"
          f"{mean_rated:>13.2f}")
print("\n  => Nếu %rated của các status non-completed RẤT thấp còn completed cao")
print("     => score chủ yếu sống ở completed; score==0 ở các status khác = 'chưa chấm'.")
print("     Nếu completed cũng có %zero đáng kể => pool positive thưa hơn tưởng.")

# ----------------------------------------------------------------------
# PASS 2 — positive pool / user: completed-only vs mọi status.
# ----------------------------------------------------------------------
print("\n" + "=" * 60)
print("D. POSITIVE POOL / USER  (positive = score>=" + str(POSITIVE) + ")")
print("=" * 60)
is_completed = pl.col("status") == "completed"
per_user = (
    lf.group_by("username")
    .agg(
        posit.sum().alias("pos_all"),
        (posit & is_completed).sum().alias("pos_completed"),
    )
    .collect(engine="streaming")
)
n_users = len(per_user)
print(f"Tổng user: {n_users:,}\n")

QS = [0.10, 0.25, 0.50, 0.75, 0.90]
def describe(col):
    s = per_user[col]
    qv = {q: s.quantile(q) for q in QS}
    return s.mean(), qv, int((s == 0).sum())

for label, col in [("completed-only", "pos_completed"), ("mọi status", "pos_all")]:
    mean, qv, n0 = describe(col)
    print(f"[{label}]  mean={mean:6.2f}   "
          + "  ".join(f"p{int(q*100)}={qv[q]:.0f}" for q in QS))
    print(f"   user có 0 positive: {n0:>9,}  ({n0 / n_users * 100:5.2f}%)")
    for thr in (1, 3, 5, 10, 20):
        c = int((per_user[col] >= thr).sum())
        print(f"   user có >= {thr:<2} positive: {c:>9,}  ({c / n_users * 100:5.2f}%)")
    print()

# Lợi ích mở rộng: positive thêm được khi tính mọi status thay vì chỉ completed.
extra = per_user["pos_all"] - per_user["pos_completed"]
tot_completed = int(per_user["pos_completed"].sum())
tot_all = int(per_user["pos_all"].sum())
print(f"Tổng positive (completed-only): {tot_completed:,}")
print(f"Tổng positive (mọi status):     {tot_all:,}")
print(f"  => mở rộng status thêm {tot_all - tot_completed:,} positive "
      f"(+{(tot_all - tot_completed) / max(tot_completed, 1) * 100:.2f}%)")
print(f"  user được THÊM >=1 positive nhờ non-completed: "
      f"{int((extra >= 1).sum()):,}  ({int((extra >= 1).sum()) / n_users * 100:.2f}%)")
