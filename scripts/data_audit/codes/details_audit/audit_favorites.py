import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).parent.parent.parent.parent.parent
SRC = ROOT / "cleaned-data" / "details.csv"

df = pd.read_csv(SRC, usecols=["favorites"])
n = len(df)
print(f"Total anime: {n:,}\n")


def print_engagement(name, s):
    print("=" * 50)
    print(name.upper())
    print("=" * 50)
    n_nan = int(s.isna().sum())
    present = s.dropna()
    print(f"NaN:     {n_nan:>7,}  ({n_nan / n * 100:5.2f}%)")
    print(f"== 0:    {int((present == 0).sum()):>7,}  ({(present == 0).sum() / n * 100:5.2f}%)")
    print(f"Present (non-zero): {int((present > 0).sum()):>7,}")

    if len(present) == 0:
        return

    print(f"\nDistribution:")
    for p in [1, 5, 25, 50, 75, 95, 99, 99.9]:
        print(f"  p{p:>5}: {present.quantile(p / 100):>14,.0f}")
    print(f"  min:   {present.min():>14,.0f}")
    print(f"  max:   {present.max():>14,.0f}")
    print(f"  mean:  {present.mean():>14,.0f}")
    print(f"  median:{present.median():>14,.0f}")

    pos = present[present > 0]
    if len(pos) > 0:
        log_s = np.log10(pos)
        print(f"\nLog10 distribution (positive only, n={len(pos):,}):")
        for p in [50, 75, 90, 99]:
            v = 10 ** log_s.quantile(p / 100)
            print(f"  p{p:>3} ~= 10^{log_s.quantile(p / 100):.2f}  ({v:,.0f})")

    print(f"\n10 sample values (random):")
    for v in present.sample(min(10, len(present)), random_state=42).tolist():
        print(f"  {v:,.0f}")


print_engagement("favorites", df["favorites"])
print()
