import pathlib

import pandas as pd

ROOT = pathlib.Path(__file__).parent.parent.parent.parent.parent
SRC = ROOT / "cleaned-data" / "details.csv"

df = pd.read_csv(SRC, usecols=["type", "status", "source", "rating"])
n = len(df)
print(f"Total anime: {n:,}\n")

print("=" * 50)
print("CROSS-TAB: type vs (status, source, rating)")
print("=" * 50)
for other in ["status", "source", "rating"]:
    print(f"\n  type × {other}:")
    ct = pd.crosstab(df["type"], df[other], dropna=False)
    print(ct.to_string())
    print()
