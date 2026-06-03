import pathlib

import pandas as pd

ROOT = pathlib.Path(__file__).parent.parent.parent
SRC = ROOT / "cleaned-data" / "details.csv"

df = pd.read_csv(SRC, usecols=["popularity", "members"])
n = len(df)
print(f"Total anime: {n:,}\n")

both = df[["popularity", "members"]].dropna()
corr = both["popularity"].corr(both["members"])
print(f"Correlation popularity vs members: {corr:+.4f}")
print("  (nếu popularity là rank: strong NEGATIVE; nếu là count: strong POSITIVE)")
