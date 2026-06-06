import ast
import pathlib
from collections import Counter

import pandas as pd

ROOT = pathlib.Path(__file__).parent.parent.parent.parent.parent
SRC = ROOT / "cleaned-data" / "details.csv"

COL = "studios"
df = pd.read_csv(SRC, usecols=[COL])
n = len(df)
print(f"Total anime: {n:,}\n")


def parse_list(v):
    if pd.isna(v):
        return None
    try:
        out = ast.literal_eval(v)
        return out if isinstance(out, list) else None
    except (ValueError, SyntaxError):
        return None


print("=" * 60)
print(f"{COL.upper()}")
print("=" * 60)
s = df[COL]

n_nan = int(s.isna().sum())
n_emptyrepr = int((s == "[]").sum())
n_missing = n_nan + n_emptyrepr
print(f"NaN:           {n_nan:>6,}  ({n_nan / n * 100:5.2f}%)")
print(f"'[]' (empty):  {n_emptyrepr:>6,}  ({n_emptyrepr / n * 100:5.2f}%)")
print(f"Total missing: {n_missing:>6,}  ({n_missing / n * 100:5.2f}%)")

parsed = s.map(parse_list)
n_parse_fail = int(((~s.isna()) & parsed.isna()).sum())
print(f"Parse failure: {n_parse_fail:>6,}  (non-NaN nhưng literal_eval lỗi)")
if n_parse_fail > 0:
    bad = s[(~s.isna()) & parsed.isna()].head(5)
    print(f"  Sample bad:")
    for v in bad.tolist():
        print(f"    {v!r}")

nonempty = parsed[parsed.map(lambda x: isinstance(x, list) and len(x) > 0)]
if len(nonempty) == 0:
    print("\n(no non-empty lists)\n")
    raise SystemExit

all_tags = [tag for lst in nonempty for tag in lst]
tag_counter = Counter(all_tags)
print(f"\nNon-empty rows: {len(nonempty):,}")
print(f"Unique tags:    {len(tag_counter):,}")
print(f"Total tag occurrences: {len(all_tags):,}")

print(f"\nTop 30 tags:")
for tag, c in tag_counter.most_common(30):
    print(f"  {c:>5,}  {tag!r}")

lens = nonempty.map(len)
print(f"\nList length distribution (non-empty rows only):")
for p in [25, 50, 75, 90, 95, 99]:
    print(f"  p{p:>3}: {int(lens.quantile(p / 100)):>3}")
print(f"  min:  {int(lens.min()):>3}")
print(f"  max:  {int(lens.max()):>3}")

single = nonempty[lens == 1]
print(f"\nSingle-tag lists: {len(single):,}  ({len(single) / len(nonempty) * 100:5.2f}% of non-empty)")
if len(single) > 0:
    single_tags = Counter(lst[0] for lst in single)
    print(f"  Top 10 single-tag values (default-suspect):")
    for tag, c in single_tags.most_common(10):
        print(f"    {c:>5,}  {tag!r}")

if len(tag_counter) > 30:
    print(f"\nCoverage CDF (top-N tags) cho long-tail column:")
    print(f"  {'N':>4}  {'mass%':>7}  {'row%':>7}  {'bucket_other (1-mass%)':>22}")
    total_occ = len(all_tags)
    for top_n in [10, 25, 50, 100, 200, 500]:
        if top_n > len(tag_counter):
            break
        top_tags_set = {t for t, _ in tag_counter.most_common(top_n)}
        mass = sum(c for _, c in tag_counter.most_common(top_n)) / total_occ
        row_cov = nonempty.map(lambda lst: any(t in top_tags_set for t in lst)).mean()
        print(f"  {top_n:>4}  {mass * 100:>6.2f}%  {row_cov * 100:>6.2f}%  {(1 - mass) * 100:>21.2f}%")

if len(tag_counter) > 100:
    print(f"\nNgưỡng cắt embed_id — occurrence count tại từng rank:")
    ranked = tag_counter.most_common()
    for r in [50, 100, 150, 200, 250, 300, 500]:
        if r <= len(ranked):
            print(f"  rank {r:>4}: count = {ranked[r - 1][1]:>4}  ({ranked[r - 1][0]!r})")

    print(f"\nPhân phối tag theo bin tần suất:")
    bins = [(1, 1), (2, 2), (3, 4), (5, 9), (10, 10**9)]
    labels = ["=1", "=2", "3-4", "5-9", ">=10"]
    for (lo, hi), lab in zip(bins, labels):
        tags_in_bin = {t for t, c in tag_counter.items() if lo <= c <= hi}
        n_tags = len(tags_in_bin)
        rows_cov = nonempty.map(lambda lst: any(t in tags_in_bin for t in lst)).mean()
        print(f"  count {lab:>4}: {n_tags:>5,} tag(s)  |  cover {rows_cov * 100:5.2f}% non-empty rows")

    print(f"\nBảng quyết định ngưỡng (giữ tag count >= min_count, còn lại → OOV):")
    print(f"  {'min_count':>9}  {'vocab(kept)':>11}  {'row%kept':>9}  {'row%OOV':>8}")
    for min_count in [2, 3, 5, 10]:
        kept = {t for t, c in tag_counter.items() if c >= min_count}
        vocab = len(kept)
        row_kept = nonempty.map(lambda lst: any(t in kept for t in lst)).mean()
        row_oov = nonempty.map(lambda lst: all(t not in kept for t in lst)).mean()
        print(f"  {min_count:>9}  {vocab:>11,}  {row_kept * 100:>8.2f}%  {row_oov * 100:>7.2f}%")

print()
