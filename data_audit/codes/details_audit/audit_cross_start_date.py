import pathlib
import pandas as pd

ROOT = pathlib.Path(__file__).parent.parent.parent.parent
SRC = ROOT / "cleaned-data" / "details.csv"

df = pd.read_csv(SRC, usecols=["start_date", "type", "source"])
n = len(df)
print(f"Total anime: {n:,}\n")

parsed = pd.to_datetime(df["start_date"], errors="coerce", utc=True)
if int(parsed.notna().sum()) == 0:
    raise SystemExit("(no parseable dates)")
df["year"] = parsed.dt.year

ERA_LABELS = ["<=1989", "1990-99", "2000-09", "2010-17", "2018+"]
ERA_BINS = [-float("inf"), 1989, 1999, 2009, 2017, float("inf")]


def to_era(year_series):
    return pd.cut(year_series, bins=ERA_BINS, labels=ERA_LABELS, right=True)


df["era"] = to_era(df["year"])

print("=" * 50)
print("ERA × TYPE  (row% trong mỗi era)")
print("=" * 50)
ct_t = pd.crosstab(df["era"], df["type"])
ct_t_pct = ct_t.div(ct_t.sum(axis=1).replace(0, 1), axis=0) * 100
type_cols = list(ct_t_pct.columns)
print(f"{'era':<10}{'n':>7}  | " + "".join(f"{str(c):>8}" for c in type_cols))
for era in ERA_LABELS:
    if era not in ct_t_pct.index:
        continue
    nn = int(ct_t.loc[era].sum())
    row = "".join(f"{ct_t_pct.loc[era, c]:>8.1f}" for c in type_cols)
    print(f"{era:<10}{nn:>7,}  | {row}")

print("\n" + "=" * 50)
print("ERA × SOURCE  (row% trong mỗi era — đọc theo HÀNG)")
print("=" * 50)
ct_s = pd.crosstab(df["era"], df["source"])
ct_s_pct = ct_s.div(ct_s.sum(axis=1).replace(0, 1), axis=0) * 100
src_cols = list(ct_s_pct.columns)
print(f"{'era':<10}{'n':>7}  | " + "".join(f"{str(c)[:10]:>11}" for c in src_cols))
for era in ERA_LABELS:
    if era not in ct_s_pct.index:
        continue
    nn = int(ct_s.loc[era].sum())
    row = "".join(f"{ct_s_pct.loc[era, c]:>11.1f}" for c in src_cols)
    print(f"{era:<10}{nn:>7,}  | {row}")

print("\n" + "-" * 50)
print("SOURCE × ERA  (row% trong mỗi source — đọc theo HÀNG)")
print("-" * 50)
ct_s2 = pd.crosstab(df["source"], df["era"])
ct_s2_pct = ct_s2.div(ct_s2.sum(axis=1).replace(0, 1), axis=0) * 100
era_cols = [e for e in ERA_LABELS if e in ct_s2_pct.columns]
print(f"{'source':<16}{'n':>7}  | " + "".join(f"{e:>10}" for e in era_cols))
for src in ct_s2_pct.index:
    nn = int(ct_s2.loc[src].sum())
    row = "".join(f"{ct_s2_pct.loc[src, e]:>10.1f}" for e in era_cols)
    print(f"{str(src):<16}{nn:>7,}  | {row}")

print("\n(Đọc: source nào dồn ~hết vào 1-2 era => source ĐÃ mã hóa era đó")
print(" => era dư thừa, cân nhắc nhường ranker. Source trải đều mọi era")
print(" => era mang signal thời kỳ độc lập => đáng feed tower.)")
