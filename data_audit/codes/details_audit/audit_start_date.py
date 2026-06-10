import pathlib
import pandas as pd

ROOT = pathlib.Path(__file__).parent.parent.parent.parent
SRC = ROOT / "cleaned-data" / "details.csv"

df = pd.read_csv(SRC, usecols=["start_date"])
n = len(df)
print(f"Total anime: {n:,}\n")

print("=" * 50)
print("START_DATE")
print("=" * 50)
sd = df["start_date"]
n_nan = int(sd.isna().sum())
present = sd.dropna().astype(str)
print(f"NaN:     {n_nan:>7,}  ({n_nan / n * 100:5.2f}%)")
print(f"Present: {len(present):>7,}  ({len(present) / n * 100:5.2f}%)")

print(f"\n10 sample values (random):")
for v in present.sample(min(10, len(present)), random_state=42).tolist():
    print(f"  {v!r}")

parsed = pd.to_datetime(sd, errors="coerce", utc=True)
n_parsed = int(parsed.notna().sum())
n_unparsed = len(present) - n_parsed
print(f"\nParseable: {n_parsed:>7,}  ({n_parsed / n * 100:5.2f}% of total)")
print(f"Unparseable (present but invalid): {n_unparsed:,}")
if n_unparsed > 0:
    bad = sd[sd.notna() & parsed.isna()].sample(min(10, n_unparsed), random_state=42)
    print(f"  Sample unparseable:")
    for v in bad.tolist():
        print(f"    {v!r}")

if n_parsed == 0:
    raise SystemExit("(no parseable dates)")

df["year"] = parsed.dt.year
years = df["year"].dropna().astype(int)

print(f"\nYear distribution:")
for p in [1, 5, 25, 50, 75, 95, 99]:
    print(f"  p{p:>3}: {int(years.quantile(p / 100)):,}")
print(f"  min:  {int(years.min()):,}")
print(f"  max:  {int(years.max()):,}")

print(f"\nFull year count:")
yc = years.value_counts().sort_index()
for y, c in yc.items():
    print(f"  {y}: {c:>5,}  ({c / n_parsed * 100:5.2f}%)")

CURRENT_YEAR = pd.Timestamp.today().year
n_pre1917 = int((years < 1917).sum())
n_future = int((years > CURRENT_YEAR + 2).sum())
print(f"\nYear anomalies:")
print(f"  Year < 1917 (trước Namakura Gatana): {n_pre1917:>4,}")
print(f"  Year > {CURRENT_YEAR + 2}:                       {n_future:>4,}")

months = parsed.dt.month.dropna().astype(int)
days = parsed.dt.day.dropna().astype(int)
print(f"\nMonth distribution:")
for m, c in months.value_counts().sort_index().items():
    marker = "  ← default-suspect" if m == 1 else ""
    print(f"  {m:>2}: {c:>5,}  ({c / n_parsed * 100:5.2f}%){marker}")
print(f"\nDay-of-month — top 10 (day=1 dominant -> scraper default):")
for d, c in days.value_counts().head(10).items():
    marker = "  ← default-suspect" if d == 1 else ""
    print(f"  day {d:>2}: {c:>5,}  ({c / n_parsed * 100:5.2f}%){marker}")
jan1 = ((months == 1) & (days == 1)).sum()
print(f"\nDate = Jan 1 (year-only fallback): {jan1:,}  ({jan1 / n_parsed * 100:5.2f}%)")

ERA_LABELS = ["<=1989", "1990-99", "2000-09", "2010-17", "2018+"]
ERA_BINS = [-float("inf"), 1989, 1999, 2009, 2017, float("inf")]


def to_era(year_series):
    return pd.cut(year_series, bins=ERA_BINS, labels=ERA_LABELS, right=True)


df["era"] = to_era(df["year"])

print("\n" + "=" * 50)
print("ERA BUCKET DISTRIBUTION")
print("=" * 50)
era_present = df["era"].dropna()
ec = era_present.value_counts().reindex(ERA_LABELS)
print("Trên toàn catalog (gồm NaN = NULL slot, n={:,}):".format(n))
for label in ERA_LABELS:
    c = int(ec.get(label, 0))
    print(f"  {label:<10} {c:>6,}  ({c / n * 100:5.2f}%)")
print(f"  {'NULL(NaN)':<10} {n_nan:>6,}  ({n_nan / n * 100:5.2f}%)")
print("\n(Bucket nào < ~3-5% => cân nhắc gộp; <=1989 mỏng thì gộp vào 1990-99.)")
