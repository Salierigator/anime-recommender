import pathlib
import re

import pandas as pd

ROOT = pathlib.Path(__file__).parent.parent.parent
SRC = ROOT / "cleaned-data" / "details.csv"

df = pd.read_csv(SRC, usecols=["synopsis"])
n = len(df)
print(f"Total anime: {n:,}\n")

print("=" * 50)
print("SYNOPSIS")
print("=" * 50)
s = df["synopsis"]
n_nan = int(s.isna().sum())
present = s.dropna().astype(str)
stripped = present.str.strip()
n_empty = int(stripped.eq("").sum())
nonempty = stripped[stripped != ""]
n_nonempty = len(nonempty)
print(f"NaN:       {n_nan:>7,}  ({n_nan / n * 100:5.2f}%)")
print(f"Empty:     {n_empty:>7,}  ({n_empty / n * 100:5.2f}%)")
print(f"Non-empty: {n_nonempty:>7,}  ({n_nonempty / n * 100:5.2f}%)")

if n_nonempty == 0:
    raise SystemExit("(no non-empty synopsis)")

char_lens = nonempty.str.len()
word_lens = nonempty.str.split().str.len()

print(f"\nLength (chars):")
for p in [50, 75, 90, 95, 99, 99.9]:
    print(f"  p{p:>5}: {int(char_lens.quantile(p / 100)):>6,}")
print(f"  min:   {int(char_lens.min()):>6,}")
print(f"  max:   {int(char_lens.max()):>6,}")

print(f"\nLength (words):")
for p in [50, 75, 90, 95, 99]:
    print(f"  p{p:>3}: {int(word_lens.quantile(p / 100)):>5,}")
print(f"  min:  {int(word_lens.min()):>5,}")
print(f"  max:  {int(word_lens.max()):>5,}")

SHORT_THRESHOLD = 50
n_short = int((char_lens < SHORT_THRESHOLD).sum())
print(f"\nShort synopsis (< {SHORT_THRESHOLD} chars): {n_short:,}  ({n_short / n_nonempty * 100:5.2f}% of non-empty)")
if n_short > 0:
    print("  5 sample shortest:")
    shortest = nonempty[char_lens < SHORT_THRESHOLD].sample(min(5, n_short), random_state=42)
    for v in shortest.tolist():
        print(f"    [{len(v):>3}] {v!r}")

PLACEHOLDER_PATTERNS = [
    r"^no synopsis",
    r"^no description",
    r"^tba\b",
    r"^to be announced",
    r"^coming soon",
    r"^n/?a$",
    r"has been added",
]
ph_regex = re.compile("|".join(PLACEHOLDER_PATTERNS), re.IGNORECASE)
n_placeholder = int(nonempty.str.contains(ph_regex, na=False).sum())
print(f"\nPlaceholder-suspect (matches 'no synopsis'/'tba'/'coming soon'/...): {n_placeholder:,}  ({n_placeholder / n_nonempty * 100:5.2f}%)")
if n_placeholder > 0:
    print("  5 sample placeholder:")
    for v in nonempty[nonempty.str.contains(ph_regex, na=False)].head(5).tolist():
        print(f"    {v[:150]!r}")

print(f"\nTop 20 most common synopsis values (default detect):")
top = nonempty.value_counts().head(20)
for v, c in top.items():
    snippet = v[:100] + ("..." if len(v) > 100 else "")
    print(f"  {c:>4}x  {snippet!r}")

src_pattern = re.compile(r"\(\s*source\s*:", re.IGNORECASE)
n_source = int(nonempty.str.contains(src_pattern, na=False).sum())
print(f"\nContains '(Source: ...)' attribution: {n_source:,}  ({n_source / n_nonempty * 100:5.2f}%)")
print("  (cần strip trước khi feed vào text encoder)")

p50 = int(char_lens.quantile(0.5))
band = nonempty[(char_lens >= p50 - 50) & (char_lens <= p50 + 50)]
print(f"\n5 random samples ~p50 length ({p50} chars):")
for v in band.sample(min(5, len(band)), random_state=42).tolist():
    print(f"  [{len(v):>4}] {v[:200]!r}{'...' if len(v) > 200 else ''}")
