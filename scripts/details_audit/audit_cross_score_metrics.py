import pathlib

import pandas as pd

ROOT = pathlib.Path(__file__).parent.parent.parent
SRC = ROOT / "cleaned-data" / "details.csv"

df = pd.read_csv(SRC, usecols=["score", "scored_by", "rank", "title", "type", "members", "popularity"])
n = len(df)
print(f"Total anime: {n:,}\n")

print("=" * 50)
print("CROSS-CONSISTENCY (null patterns)")
print("=" * 50)
nan_score = df["score"].isna()
nan_sb = df["scored_by"].isna()
nan_rank = df["rank"].isna()

print(f"score NaN == scored_by NaN:   {bool((nan_score == nan_sb).all())}")
print(f"score NaN == rank NaN:        {bool((nan_score == nan_rank).all())}")

print(f"\nNaN combination breakdown (S=score, B=scored_by, R=rank):")
combos = (
    nan_score.astype(int).astype(str)
    + nan_sb.astype(int).astype(str)
    + nan_rank.astype(int).astype(str)
)
for combo, cnt in combos.value_counts().items():
    label = f"S={combo[0]} B={combo[1]} R={combo[2]}"
    print(f"  {label}: {cnt:>6,}  ({cnt / n * 100:5.2f}%)")

mask_score_no_rank = (~nan_score) & nan_rank
n_inconsistent = int(mask_score_no_rank.sum())
print(f"\nHas score but no rank: {n_inconsistent:,}")
if 0 < n_inconsistent <= 20:
    print("  scores of those:")
    for v in df.loc[mask_score_no_rank, "score"].head(10).tolist():
        print(f"    {v}")

print("\n" + "=" * 50)
print("INVESTIGATE: no score+scored_by nhưng có rank")
print("=" * 50)
mask_no_score_has_rank = nan_score & nan_sb & (~nan_rank)
n_anomaly = int(mask_no_score_has_rank.sum())
sub = df.loc[mask_no_score_has_rank].copy()
print(f"Total rows: {n_anomaly:,}  ({n_anomaly / n * 100:5.2f}%)")

print(f"\nBreakdown by type:")
for t, c in sub["type"].value_counts(dropna=False).items():
    print(f"  {str(t):<12}: {c:>5,}  ({c / n_anomaly * 100:5.2f}%)")

sub_sorted = sub.sort_values("rank", ascending=True).head(10)
print(f"\nSample 10 rows (sorted by rank ascending):")
print(f"  {'rank':>7}  {'members':>9}  {'pop':>6}  {'type':<10}  title")
for _, row in sub_sorted.iterrows():
    title = str(row["title"])[:60]
    print(
        f"  {int(row['rank']):>7,}  {int(row['members']):>9,}  "
        f"{int(row['popularity']):>6,}  {str(row['type']):<10}  {title!r}"
    )

print(f"\nRank distribution của nhóm này:")
sub_rank = sub["rank"].dropna()
for p in [5, 25, 50, 75, 95]:
    print(f"  p{p:>2}: {sub_rank.quantile(p / 100):>8,.0f}")
print(f"  min:  {int(sub_rank.min()):>8,}")
print(f"  max:  {int(sub_rank.max()):>8,}")
