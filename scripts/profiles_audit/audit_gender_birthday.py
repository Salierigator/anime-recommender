"""Audit nhanh `gender` + `birthday` trong cleaned-data/profiles.csv.

In stdout: null rate, value distribution, year-only parse-ability, anomalies
(year < 1950, MAL default 1900, year > current, format lạ — không có year).

Birthday chỉ parse YEAR (4-digit), không quan tâm ngày/tháng.

Usage:
    python scripts/profiles_audit/audit_gender_birthday.py
"""
import pathlib
import re

import pandas as pd

ROOT = pathlib.Path(__file__).parent.parent.parent
SRC = ROOT / "cleaned-data" / "profiles.csv"

df = pd.read_csv(SRC, usecols=["gender"])
n = len(df)
print(f"Total profiles: {n:,}\n")

# ===== GENDER =====
print("=" * 50)
print("GENDER")
print("=" * 50)
g = df["gender"]
n_nan = int(g.isna().sum())
print(f"NaN:     {n_nan:>7,}  ({n_nan / n * 100:5.2f}%)")
print(f"Present: {n - n_nan:>7,}  ({(n - n_nan) / n * 100:5.2f}%)")
print(f"\nValue distribution:")
print(g.value_counts(dropna=False).to_string())

# # ===== BIRTHDAY =====
# print("\n" + "=" * 50)
# print("BIRTHDAY")
# print("=" * 50)
# b = df["birthday"]
# n_nan = int(b.isna().sum())
# present = b.dropna().astype(str).str.strip()
# present = present[present != ""]
# n_present = len(present)
# print(f"NaN/empty: {n - n_present:>7,}  ({(n - n_present) / n * 100:5.2f}%)")
# print(f"Present:   {n_present:>7,}  ({n_present / n * 100:5.2f}%)")

# print(f"\n10 sample values (random):")
# for v in present.sample(10, random_state=42).tolist():
#     print(f"  {v!r}")

# # ===== YEAR EXTRACTION =====
# # Bắt 4-digit year (1000-2999). Dùng \b boundary để tránh nuốt "1995x".
# YEAR_RE = re.compile(r"\b(1\d{3}|2\d{3})\b")

# def extract_year(s):
#     m = YEAR_RE.search(s)
#     return int(m.group(1)) if m else None

# years_series = present.map(extract_year)
# has_year = years_series.notna()
# no_year = ~has_year

# n_with_year = int(has_year.sum())
# n_no_year = int(no_year.sum())
# print(f"\nExtracted 4-digit year: {n_with_year:>7,}  ({n_with_year / n_present * 100:5.2f}% of present)")
# print(f"NO year found:          {n_no_year:>7,}  ({n_no_year / n_present * 100:5.2f}% of present)")

# # Dump distribution of no-year formats (format lạ — chỉ có month/day, hoặc weird text)
# if n_no_year > 0:
#     print(f"\nTop 30 'no-year' raw values (likely month-day only hoặc format lạ):")
#     no_year_vals = present[no_year].value_counts().head(30)
#     for v, c in no_year_vals.items():
#         print(f"  {c:>6,}  {v!r}")

#     print(f"\n20 random 'no-year' samples:")
#     for v in present[no_year].sample(min(20, n_no_year), random_state=42).tolist():
#         print(f"  {v!r}")

# # Multiple years in 1 string → suspicious
# def count_years(s):
#     return len(YEAR_RE.findall(s))

# multi_year_mask = present.map(count_years) > 1
# n_multi = int(multi_year_mask.sum())
# print(f"\nStrings with >1 year (suspicious): {n_multi:,}")
# if n_multi > 0:
#     for v in present[multi_year_mask].head(10).tolist():
#         print(f"  {v!r}")

# # ===== YEAR DISTRIBUTION =====
# years = years_series.dropna().astype(int)
# if len(years) == 0:
#     print("\n(no parseable years — skip distribution)")
# else:
#     print(f"\nYear of birth — distribution:")
#     for p in [1, 5, 25, 50, 75, 95, 99]:
#         print(f"  p{p:>3}: {int(years.quantile(p / 100)):,}")
#     print(f"  min:  {int(years.min()):,}")
#     print(f"  max:  {int(years.max()):,}")

#     CURRENT_YEAR = pd.Timestamp.today().year
#     n_pre1900 = int((years < 1900).sum())
#     n_1900 = int((years == 1900).sum())
#     n_pre1950 = int(((years >= 1900) & (years < 1950)).sum())
#     n_future = int((years > CURRENT_YEAR).sum())
#     n_too_young = int(((years > 2015) & (years <= CURRENT_YEAR)).sum())

#     print(f"\nAnomalies (year-based):")
#     print(f"  Year < 1900:                {n_pre1900:>6,}  (impossible)")
#     print(f"  Year == 1900:               {n_1900:>6,}  (MAL default suspect)")
#     print(f"  1900 < Year < 1950:         {n_pre1950:>6,}  (likely fake/default, non-1900)")
#     print(f"  2015 < Year <= {CURRENT_YEAR}:       {n_too_young:>6,}  (likely fake/typo — quá trẻ)")
#     print(f"  Year > {CURRENT_YEAR} (future):     {n_future:>6,}  (invalid)")

#     print(f"\nFull year count (sorted by year):")
#     yc = years.value_counts().sort_index()
#     for y, c in yc.items():
#         print(f"  {y}: {c:>6,}  ({c / len(years) * 100:5.2f}%)")

# # Top raw values (detect default strings)
# print(f"\nTop 20 most common raw birthday values (detect defaults):")
# top = present.value_counts().head(20)
# for v, c in top.items():
#     print(f"  {c:>6,}  {v!r}")
