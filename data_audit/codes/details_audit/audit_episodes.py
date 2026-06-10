import pathlib
import pandas as pd

ROOT = pathlib.Path(__file__).parent.parent.parent.parent
SRC = ROOT / "cleaned-data" / "details.csv"

df = pd.read_csv(SRC, usecols=["episodes", "type", "status"])
n = len(df)
print(f"Total anime: {n:,}\n")

print("=" * 50)
print("EPISODES")
print("=" * 50)
e = df["episodes"]
n_nan = int(e.isna().sum())
nan_mask = e.isna()
present = e.dropna()
print(f"NaN:     {n_nan:>7,}  ({n_nan / n * 100:5.2f}%)")
print(f"Present: {len(present):>7,}  ({len(present) / n * 100:5.2f}%)")
print(f"== 0:    {int((present == 0).sum()):>7,}")

if len(present) > 0:
    print(f"\nDistribution:")
    for p in [1, 5, 25, 50, 75, 95, 99, 99.9]:
        print(f"  p{p:>5}: {present.quantile(p / 100):>8,.1f}")
    print(f"  min:   {present.min():>8,.1f}")
    print(f"  max:   {present.max():>8,.1f}")
    print(f"  mean:  {present.mean():>8,.1f}")

    print(f"\nTop 20 most common episode counts:")
    top = present.value_counts().head(20)
    for v, c in top.items():
        print(f"  {int(v):>4} ep: {c:>5,}  ({c / len(present) * 100:5.2f}%)")

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

print("\n" + "=" * 50)
print("BUCKET DISTRIBUTION (đề xuất layout)")
print("=" * 50)
present_buckets = df.loc[~nan_mask, "ep_bucket"]
print("Trên non-NaN rows (n={:,}):".format(len(present_buckets)))
bcount = present_buckets.value_counts().reindex(BUCKET_LABELS)
for label in BUCKET_LABELS:
    c = int(bcount.get(label, 0))
    pct = c / len(present_buckets) * 100 if len(present_buckets) else 0
    print(f"  {label:<22} {c:>6,}  ({pct:5.2f}%)")

print("\nTrên toàn catalog (gồm NaN = Null slot, n={:,}):".format(n))
for label in BUCKET_LABELS:
    c = int(bcount.get(label, 0))
    print(f"  {label:<22} {c:>6,}  ({c / n * 100:5.2f}%)")
print(f"  {'NULL (NaN)':<22} {n_nan:>6,}  ({n_nan / n * 100:5.2f}%)")

print("\n" + "=" * 50)
print("KIỂM TRA RANH GIỚI (count quanh mỗi đường cắt)")
print("=" * 50)
edges = [2, 6, 13, 26, 52]
vc = present.value_counts()
for edge in edges:
    around = {k: int(vc.get(k, 0)) for k in range(edge - 1, edge + 3)}
    cells = "  ".join(f"{k}ep:{v:,}" for k, v in around.items())
    print(f"  cắt sau {edge:>2} ->  {cells}")
print("\n(Nếu ngay cạnh đường cắt có đỉnh lớn bị tách đôi (vd 12 và 13 cùng cao")
print(" mà rơi 2 bucket) thì cân nhắc dịch ranh giới cho đỉnh nằm trọn 1 bucket.)")

print("\n" + "=" * 50)
print("ANOMALIES")
print("=" * 50)
n_zero = int((present == 0).sum())
n_huge = int((present > 500).sum())
print(f"Episodes == 0:  {n_zero:>5,}  (đáng nghi — anime tồn tại nhưng 0 ep)")
print(f"Episodes > 500: {n_huge:>5,}  (long-running shounen?)")
if n_huge > 0:
    print(f"\n  Top entries với episodes > 500:")
    big = df[df["episodes"] > 500].sort_values("episodes", ascending=False).head(10)
    for _, row in big.iterrows():
        print(f"    {int(row['episodes']):>5} ep  type={str(row['type']):<10}  status={row['status']}")
