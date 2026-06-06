import pathlib

import pandas as pd

ROOT = pathlib.Path(__file__).parent.parent.parent.parent.parent
SRC = ROOT / "cleaned-data" / "details.csv"

df = pd.read_csv(SRC, usecols=["mal_id"])
n = len(df)
print(f"Total anime: {n:,}\n")

print("=" * 50)
print("MAL_ID (primary key)")
print("=" * 50)
mid = df["mal_id"]
print(f"Null:        {int(mid.isna().sum()):>7,}  (expect 0)")
print(f"Unique:      {mid.nunique():>7,}  (expect = total)")
print(f"Duplicates:  {n - mid.nunique():>7,}")
print(f"min:         {int(mid.min()):>7,}")
print(f"max:         {int(mid.max()):>7,}")
print(f"dtype:       {mid.dtype}")
