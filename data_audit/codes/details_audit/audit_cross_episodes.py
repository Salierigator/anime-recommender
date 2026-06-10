import pathlib
import pandas as pd

ROOT = pathlib.Path(__file__).parent.parent.parent.parent
SRC = ROOT / "cleaned-data" / "details.csv"

df = pd.read_csv(SRC, usecols=["episodes", "type", "status"])
n = len(df)
print(f"Total anime: {n:,}\n")

e = df["episodes"]
nan_mask = e.isna()

BUCKET_LABELS = [
    "1 (one-shot/movie)",
    "2 (pair)",
    "3-6 (short/OVA)",
    "7-13 (1-cua)",
    "14-26 (2-cua)",
    "27-52 (long/4-cua)",
    "53+ (long-running)",
]
BUCKET_BINS = [0, 1, 2, 6, 13, 26, 52, float("inf")]


def bucketize(series):
    return pd.cut(
        series,
        bins=BUCKET_BINS,
        labels=BUCKET_LABELS,
        right=True,
        include_lowest=True,
    )


df["ep_bucket"] = bucketize(df["episodes"])

print("=" * 50)
print("NaN vs STATUS")
print("=" * 50)
print(f"Episodes NaN breakdown by status:")
for status, cnt in df.loc[nan_mask, "status"].value_counts(dropna=False).items():
    print(f"  {str(status):<25} {cnt:>5,}")

print("\n" + "=" * 50)
print("EPISODES theo TYPE")
print("=" * 50)
by_type = df.groupby("type", dropna=False)["episodes"].agg(
    n="count",
    n_nan=lambda s: int(s.isna().sum()),
    median="median",
    p95=lambda s: s.quantile(0.95),
    max="max",
)
print(by_type.to_string(float_format=lambda x: f"{x:,.1f}"))

print("\n" + "=" * 50)
print("BUCKET × TYPE  (row% trong mỗi type)")
print("=" * 50)
ct = pd.crosstab(df["type"], df["ep_bucket"])
ct = ct.reindex(columns=BUCKET_LABELS, fill_value=0)
ct_pct = ct.div(ct.sum(axis=1).replace(0, 1), axis=0) * 100
type_n = df.groupby("type", dropna=False).size()
type_nan = df[nan_mask].groupby("type", dropna=False).size()
print(f"{'type':<12}{'n':>7}{'%NaN':>7}  | " + "".join(f"{l.split()[0]:>8}" for l in BUCKET_LABELS))
for t in ct_pct.index:
    nn = int(type_n.get(t, 0))
    pn = type_nan.get(t, 0) / nn * 100 if nn else 0
    row = "".join(f"{ct_pct.loc[t, l]:>8.1f}" for l in BUCKET_LABELS)
    print(f"{str(t):<12}{nn:>7,}{pn:>6.1f}%  | {row}")
print("\n(Đọc: hàng nào dồn ~100% vào 1 cột => bucket dư thừa với type ở type đó;")
print(" hàng trải đều nhiều cột => episodes mang signal độc lập, đáng feed tower.)")
