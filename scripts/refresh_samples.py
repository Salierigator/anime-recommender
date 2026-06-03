"""Refresh data-sample/ với 5 dòng đầu từ cleaned-data/*.csv.

Chạy lại sau mỗi lần re-run cleaning.ipynb để sample reflect schema mới.

Usage:
    python scripts/refresh_samples.py
"""
import pathlib

import pandas as pd

ROOT = pathlib.Path(__file__).parent.parent
SRC = ROOT / "cleaned-data"
DST = ROOT / "data-sample"
DST.mkdir(exist_ok=True)

for src in sorted(SRC.glob("*.csv")):
    df = pd.read_csv(src, nrows=5)
    dst = DST / f"{src.stem}_sample.csv"
    df.to_csv(dst, index=False)
    print(f"  {src.name:>15} → {dst.name:>25}  ({df.shape[1]} cols × 5 rows)")
