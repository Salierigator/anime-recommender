import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).parent.parent.parent.parent.parent
SRC = ROOT / "cleaned-data" / "details.csv"

df = pd.read_csv(SRC, usecols=["scored_by"])
n = len(df)
print(f"Total anime: {n:,}\n")


def print_numeric(name, s, percentiles=(1, 5, 25, 50, 75, 95, 99), log_too=False):
    print("=" * 50)
    print(name.upper())
    print("=" * 50)
    n_nan = int(s.isna().sum())
    present = s.dropna()
    print(f"NaN:     {n_nan:>7,}  ({n_nan / n * 100:5.2f}%)")
    print(f"Present: {len(present):>7,}  ({len(present) / n * 100:5.2f}%)")
    if len(present) == 0:
        return
    print(f"\nDistribution:")
    for p in percentiles:
        print(f"  p{p:>3}: {present.quantile(p / 100):>12,.2f}")
    print(f"  min:  {present.min():>12,.2f}")
    print(f"  max:  {present.max():>12,.2f}")
    print(f"  mean: {present.mean():>12,.2f}")
    print(f"  std:  {present.std():>12,.2f}")
    if log_too and (present > 0).all():
        log_s = np.log10(present)
        print(f"\nLog10 distribution:")
        for p in [50, 90, 99]:
            print(f"  p{p:>3} (log10): {log_s.quantile(p / 100):>6.2f}")


print_numeric("scored_by", df["scored_by"], log_too=True)

sb = df["scored_by"].dropna()
n_zero = int((sb == 0).sum())
print(f"\nscored_by == 0: {n_zero:,}  (anime present nhưng 0 user rate)")
