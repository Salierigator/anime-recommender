import pathlib

import pandas as pd

ROOT = pathlib.Path(__file__).parent.parent.parent
SRC = ROOT / "cleaned-data" / "details.csv"

df = pd.read_csv(SRC, usecols=["title"])
n = len(df)
print(f"Total anime: {n:,}\n")

print("=" * 50)
print("TITLE")
print("=" * 50)
t = df["title"]
n_nan = int(t.isna().sum())
present = t.dropna().astype(str)
present_stripped = present.str.strip()
n_empty = int(present_stripped.eq("").sum())
n_present = len(present) - n_empty
print(f"NaN:     {n_nan:>7,}  ({n_nan / n * 100:5.2f}%)")
print(f"Empty:   {n_empty:>7,}  ({n_empty / n * 100:5.2f}%)")
print(f"Present: {n_present:>7,}  ({n_present / n * 100:5.2f}%)")

present_nonempty = present_stripped[present_stripped != ""]

lens = present_nonempty.str.len()
print(f"\nTitle length (chars):")
for p in [50, 75, 90, 95, 99, 99.9]:
    print(f"  p{p:>5}: {int(lens.quantile(p / 100)):>4}")
print(f"  min  : {int(lens.min()):>4}")
print(f"  max  : {int(lens.max()):>4}")

dup_counts = present_nonempty.value_counts()
n_dup_title = int((dup_counts > 1).sum())
n_anime_in_dups = int(dup_counts[dup_counts > 1].sum())
print(f"\nDuplicate titles (>=2 anime share same title): {n_dup_title:,} groups, {n_anime_in_dups:,} anime")
if n_dup_title > 0:
    print("  Top 10 most-shared titles:")
    for v, c in dup_counts.head(10).items():
        print(f"    {c:>3}x  {v!r}")

PLACEHOLDERS = ["unknown", "untitled", "tba", "tbd", "n/a", "na", "none", "???", "??", "test"]
mask_placeholder = present_nonempty.str.lower().isin(PLACEHOLDERS)
n_placeholder = int(mask_placeholder.sum())
print(f"\nPlaceholder-suspect titles (unknown/untitled/tba/...): {n_placeholder}")
if n_placeholder > 0:
    for v, c in present_nonempty[mask_placeholder].value_counts().head(10).items():
        print(f"  {c:>3}x  {v!r}")

mask_alldigit = present_nonempty.str.fullmatch(r"\d+")
n_alldigit = int(mask_alldigit.sum())
print(f"\nAll-digit titles: {n_alldigit}")
if n_alldigit > 0 and n_alldigit < 30:
    for v in present_nonempty[mask_alldigit].head(10).tolist():
        print(f"  {v!r}")

mask_short = lens <= 2
n_short = int(mask_short.sum())
print(f"\nVery short titles (<=2 chars): {n_short}")
if 0 < n_short <= 30:
    for v in present_nonempty[mask_short].head(20).tolist():
        print(f"  {v!r}")
