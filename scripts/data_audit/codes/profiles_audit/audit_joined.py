"""Audit `joined` (profiles.csv) phục vụ quyết định bucket cho user tower.

Phần gốc: null/parse/year-month/anomalies.
Phần thêm (quyết định bucket):
  B. Cohort bucket distribution — chắc không bucket nào mỏng.
  C. joined × activity (số rating/user từ ratings.csv) — joined có là proxy
     cho mức độ tích cực không?
  D. joined × năm-TB-anime-user-đã-xem (ratings + details.start_date) —
     joined có DƯ THỪA với history không (late-joiner chỉ xem đồ mới?).

Usage:
    python scripts/profiles_audit/audit_joined.py
"""
import pathlib
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).parent.parent.parent.parent.parent
PROFILES = ROOT / "cleaned-data" / "profiles.csv"
RATINGS  = ROOT / "cleaned-data" / "ratings.csv"
DETAILS  = ROOT / "cleaned-data" / "details.csv"

# --- tên cột: CHỈNH nếu auto-detect sai ---------------------------------
PROFILE_KEY_CANDS = ["username", "user", "user_id"]
RATING_USER_CANDS = ["username", "user", "user_id"]
RATING_ITEM_CANDS = ["anime_id", "mal_id", "anime", "item_id"]
RATING_SCORE_CANDS = ["score", "rating", "my_score"]
DETAIL_ITEM_CANDS = ["mal_id", "anime_id", "id", "anime"]
DETAIL_DATE_CANDS = ["start_date", "aired_from", "start"]


def pick(cols, cands, label):
    for c in cands:
        if c in cols:
            return c
    raise SystemExit(f"[!] Không tìm thấy cột {label} trong {list(cols)}; "
                     f"sửa *_CANDS ở đầu file.")


# ======================================================================
# PHẦN GỐC — joined audit (profiles.csv)
# ======================================================================
df = pd.read_csv(PROFILES)
key_col = pick(df.columns, PROFILE_KEY_CANDS, "username")
n = len(df)
print(f"Total profiles: {n:,}\n")
print("=" * 50)
print("JOINED")
print("=" * 50)
j = df["joined"]
n_nan = int(j.isna().sum())
present = j.dropna()
print(f"NaN:     {n_nan:>7,}  ({n_nan / n * 100:5.2f}%)")
print(f"Present: {len(present):>7,}  ({len(present) / n * 100:5.2f}%)")

print(f"\n10 sample values (random):")
for v in present.sample(min(10, len(present)), random_state=42).tolist():
    print(f"  {v}")

parsed = pd.to_datetime(j, errors="coerce")
n_parsed = int(parsed.notna().sum())
n_unparsed = len(present) - n_parsed
print(f"\nParseable as date: {n_parsed:>7,}  ({n_parsed / n * 100:5.2f}% of total)")
print(f"Unparseable (present but invalid): {n_unparsed:,}")
if n_unparsed > 0:
    bad = j[j.notna() & parsed.isna()].sample(min(10, n_unparsed), random_state=42)
    print(f"  Sample unparseable:")
    for v in bad.tolist():
        print(f"    {v}")

df["join_year"] = parsed.dt.year
years = df["join_year"].dropna()
print(f"\nYear joined — distribution:")
for p in [1, 5, 25, 50, 75, 95, 99]:
    print(f"  p{p:>3}: {int(years.quantile(p / 100)):,}")
print(f"  min:  {int(years.min()):,}")
print(f"  max:  {int(years.max()):,}")

print(f"\nYear counts (full):")
for y, c in years.astype(int).value_counts().sort_index().items():
    print(f"  {y}: {c:>7,}  ({c / n_parsed * 100:5.2f}%)")

MAL_LAUNCH = pd.Timestamp("2004-11-04")
TODAY = pd.Timestamp.today().normalize()
print(f"\nAnomalies:")
print(f"  Before MAL launch (< 2004-11-04): {int((parsed < MAL_LAUNCH).sum()):>6,}")
print(f"  After today ({TODAY.date()}):     {int((parsed > TODAY).sum()):>6,}")

months = parsed.dt.month.dropna().astype(int)
print(f"\nMonth distribution:")
for m, c in months.value_counts().sort_index().items():
    print(f"  {m:>2}: {c:>7,}  ({c / n_parsed * 100:5.2f}%)")
days = parsed.dt.day.dropna().astype(int)
print(f"\nDay-of-month — top 10:")
for d, c in days.value_counts().head(10).items():
    print(f"  day {d:>2}: {c:>7,}  ({c / n_parsed * 100:5.2f}%)")
print(f"\nTop 10 most common joined values:")
for v, c in j.value_counts().head(10).items():
    print(f"  {c:>6,}  {v}")

# ======================================================================
# B. COHORT BUCKET DISTRIBUTION
# ======================================================================
COHORT_LABELS = ["<=2012", "2013-16", "2017-19", "2020-21", "2022+"]
COHORT_BINS = [-np.inf, 2012, 2016, 2019, 2021, np.inf]

def to_cohort(year_series):
    return pd.cut(year_series, bins=COHORT_BINS, labels=COHORT_LABELS, right=True)

df["cohort"] = to_cohort(df["join_year"])
print("\n" + "=" * 50)
print("B. COHORT BUCKET DISTRIBUTION")
print("=" * 50)
cc = df["cohort"].value_counts().reindex(COHORT_LABELS)
for label in COHORT_LABELS:
    c = int(cc.get(label, 0))
    print(f"  {label:<10} {c:>8,}  ({c / n * 100:5.2f}%)")
print(f"  {'NULL':<10} {n_nan:>8,}  ({n_nan / n * 100:5.2f}%)")
print("\n(Bucket nào < ~5% => cân nhắc gộp.)")

# ======================================================================
# C + D. NỐI RATINGS + DETAILS — activity & redundancy với history
# ======================================================================
print("\n" + "=" * 50)
print("LOAD ratings.csv (chunked) + details.csv ...")
print("=" * 50)
RATINGS_CHUNK = 5_000_000  # số dòng mỗi chunk; ratings.csv ~3.5GB nên KHÔNG load 1 lần
try:
    r_head = pd.read_csv(RATINGS, nrows=5)
    ru = pick(r_head.columns, RATING_USER_CANDS, "rating.user")
    ri = pick(r_head.columns, RATING_ITEM_CANDS, "rating.item")

    d_head = pd.read_csv(DETAILS, nrows=5)
    di = pick(d_head.columns, DETAIL_ITEM_CANDS, "details.item")
    dd = pick(d_head.columns, DETAIL_DATE_CANDS, "details.date")
    details = pd.read_csv(DETAILS, usecols=[di, dd]).rename(columns={di: "item", dd: "start_date"})
    details["anime_year"] = pd.to_datetime(details["start_date"], errors="coerce", utc=True).dt.year
    print(f"  details rows: {len(details):,}  (item={di}, date={dd})")
    year_map = details.set_index("item")["anime_year"]
except FileNotFoundError as e:
    raise SystemExit(f"[!] Không mở được file: {e}. Phần C/D bỏ qua.")

# --- per-user aggregates: stream ratings theo chunk để tránh tràn RAM ---
# Chỉ cần count + mean(anime_year) mỗi user => cộng dồn được qua từng chunk:
#   n_ratings  = tổng số dòng / user
#   mean_year  = (tổng anime_year non-null) / (số dòng có anime_year)  -- bỏ NaN, khớp .mean()
n_ratings_acc = year_sum_acc = year_cnt_acc = None
total_rows = 0
for chunk in pd.read_csv(RATINGS, usecols=[ru, ri], chunksize=RATINGS_CHUNK):
    chunk = chunk.rename(columns={ru: "user", ri: "item"})
    total_rows += len(chunk)
    chunk["anime_year"] = chunk["item"].map(year_map)
    cnt = chunk.groupby("user").size()
    yg = chunk.groupby("user")["anime_year"]
    ysum = yg.sum()    # NaN coi như 0 => cộng dồn được
    ycnt = yg.count()  # chỉ đếm non-null
    if n_ratings_acc is None:
        n_ratings_acc, year_sum_acc, year_cnt_acc = cnt, ysum, ycnt
    else:
        n_ratings_acc = n_ratings_acc.add(cnt, fill_value=0)
        year_sum_acc = year_sum_acc.add(ysum, fill_value=0)
        year_cnt_acc = year_cnt_acc.add(ycnt, fill_value=0)
print(f"  ratings rows: {total_rows:,}  (user={ru}, item={ri})")

n_ratings = n_ratings_acc.astype(int).rename("n_ratings")
mean_anime_year = (year_sum_acc / year_cnt_acc.replace(0, np.nan)).rename("mean_anime_year")

user_agg = pd.concat([n_ratings, mean_anime_year], axis=1).reset_index()
# nối cohort của user
prof = df[[key_col, "join_year", "cohort"]].rename(columns={key_col: "user"})
user_agg = user_agg.merge(prof, on="user", how="inner")
print(f"  users khớp profiles∩ratings: {len(user_agg):,}")

# ----------------------------------------------------------------------
# C. joined cohort × ACTIVITY (n_ratings)
# ----------------------------------------------------------------------
print("\n" + "=" * 50)
print("C. COHORT × ACTIVITY (số rating mỗi user)")
print("=" * 50)
print(f"{'cohort':<10}{'n_user':>9}{'median':>9}{'mean':>10}{'p90':>9}")
for label in COHORT_LABELS:
    sub = user_agg.loc[user_agg["cohort"] == label, "n_ratings"]
    if len(sub) == 0:
        continue
    print(f"{label:<10}{len(sub):>9,}{sub.median():>9.0f}{sub.mean():>10.1f}{sub.quantile(.9):>9.0f}")
print("\n(Nếu cohort cũ có n_ratings cao hơn hẳn => joined là proxy cho activity")
print(" => activity nên xử riêng, không nhầm thành 'sở thích'.)")

# ----------------------------------------------------------------------
# D. joined cohort × NĂM-TB-ANIME-ĐÃ-XEM  (redundancy với history)
# ----------------------------------------------------------------------
print("\n" + "=" * 50)
print("D. COHORT × NĂM-TB-ANIME-ĐÃ-XEM  (dư thừa với history?)")
print("=" * 50)
print(f"{'cohort':<10}{'n_user':>9}{'median':>9}{'mean':>9}{'p25':>9}{'p75':>9}")
for label in COHORT_LABELS:
    sub = user_agg.loc[user_agg["cohort"] == label, "mean_anime_year"].dropna()
    if len(sub) == 0:
        continue
    print(f"{label:<10}{len(sub):>9,}{sub.median():>9.1f}{sub.mean():>9.1f}"
          f"{sub.quantile(.25):>9.1f}{sub.quantile(.75):>9.1f}")

# correlation join_year vs mean_anime_year (per user)
valid = user_agg.dropna(subset=["join_year", "mean_anime_year"])
corr = valid["join_year"].corr(valid["mean_anime_year"])
print(f"\ncorr(join_year, mean_anime_year) = {corr:+.3f}   (n={len(valid):,})")
print("(|corr| cao & cohort cũ xem anime cũ rõ rệt => joined DƯ THỪA với history")
print(" => bỏ joined, history bắt hộ. |corr| thấp & các cohort xem trải đều mọi")
print(" era => joined mang cohort-signal độc lập => giữ làm cold-start prior.)")