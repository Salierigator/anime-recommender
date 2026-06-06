import pathlib

import pandas as pd

ROOT = pathlib.Path(__file__).parent.parent.parent.parent.parent
SRC = ROOT / "cleaned-data" / "details.csv"

COL = "rating"
DEFAULT_SUSPECT = {"unknown", "none", "n/a", "na", "not yet aired", "tba", "tbd"}

df = pd.read_csv(SRC, usecols=[COL])
n = len(df)
print(f"Total anime: {n:,}\n")

print("=" * 50)
print(f"{COL.upper()}")
print("=" * 50)
s = df[COL]
n_nan = int(s.isna().sum())
print(f"NaN:         {n_nan:>6,}  ({n_nan / n * 100:5.2f}%)")
print(f"Cardinality: {s.nunique():>6,}")

vc = s.value_counts(dropna=False)
print(f"\nFull value distribution:")
for v, c in vc.items():
    marker = ""
    if isinstance(v, str) and v.strip().lower() in DEFAULT_SUSPECT:
        marker = "  ← default-suspect"
    print(f"  {c:>6,}  ({c / n * 100:5.2f}%)  {v!r}{marker}")
print()
