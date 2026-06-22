# Cleaning Pipeline

Doc tổng hợp cho `cleaning.ipynb` — giải thích **clean cái gì, set up thế nào, vì sao**.

---

## 1. Overview

**Mục tiêu**: chuẩn bị 3 file CSV ở `cleaned-data/` làm input cho training pipeline (two-tower retriever + ranker).

**Philosophy**: principled aggressive — mọi rule đều có lý do định lượng (percentile, physical argument, evidence từ specific case), không có magic number tùy hứng.

**Kết quả**: 124.3M ratings raw → 120.03M ratings sạch (−3.43%); 28,955 anime → 22,821 (−21.18%); 337K user profiles → 292,591 (−13.22%). Sparsity 98.20%.

---

## 2. Input — raw data

| File | Size | Rows | Cols |
|---|---|---|---|
| `data/details.csv` | 19 MB | 28,955 | 29 |
| `data/profiles.csv` | 17 MB | 337,155 | 10 |
| `data/ratings.csv` | **4.2 GB** | 124,298,357 | 6 |

### Schema highlights

**`details.csv`** — anime metadata, scrape từ MAL. Notable:
- 5 cột list-string (`genres`, `studios`, `themes`, `demographics`, `producers`) lưu Python repr `"['a','b']"`. **Empty là `"[]"` không phải NaN** → `pandas.isna()` báo 0% là false-zero (xem §11 audit).
- Numeric: `score`, `members`, `favorites`, `scored_by`, `rank`, `popularity`.
- `mal_id` (PK), `title`, `synopsis`, `type`, `episodes`, `status`, `rating`, `source`, `start_date`.

> Phân phối chi tiết từng cột (sau làm sạch) — null-rate, range/percentile, top category, đặc
> điểm long-tail — ở [DATA_DISTRIBUTIONS.md](DATA_DISTRIBUTIONS.md). Doc này chỉ giữ phần liên
> quan trực tiếp tới *lý do làm sạch*.

**`profiles.csv`** — user metadata:
- `username` (PK), `gender`, `birthday`, `location`, `joined`.
- 5 cột counter aggregate (`watching`, `completed`, `on_hold`, `dropped`, `plan_to_watch`) — redundant với `ratings.csv`.

**`ratings.csv`** — interaction (4.2 GB, 124.3M rows):
- `username`, `anime_id`, `status` ∈ {completed, plan_to_watch, watching, dropped, on_hold, unknown}, `score` ∈ [0..10] (0 = chưa rate), `is_rewatching`, `num_watched_episodes`.
- Status raw distribution:

| status | count | % |
|---|---|---|
| completed | 79,138,385 | 63.7% |
| plan_to_watch | 31,631,312 | 25.4% |
| watching | 5,678,167 | 4.6% |
| dropped | 4,580,651 | 3.7% |
| on_hold | 3,253,152 | 2.6% |
| unknown | 16,690 | 0.013% (scraper artifact) |

---

## 3. Output — cleaned data

| File | Size | Rows | Cols |
|---|---|---|---|
| `cleaned-data/details.csv` | 12.5 MB | 22,821 | 19 |
| `cleaned-data/profiles.csv` | 10.9 MB | 292,591 | 3 |
| `cleaned-data/ratings.csv` | **3.2 GB** | 120,032,917 | 4 |

### Schema final

- `details.csv`: `mal_id, title, type, status, score, scored_by, start_date, synopsis, rank, popularity, members, favorites, genres, studios, themes, demographics, source, rating, episodes`
- `profiles.csv`: `username, gender, joined`
- `ratings.csv`: `username, anime_id, status, score`

---

## 4. Pipeline overview

| § | Tên | Input | Output | Key metric |
|---|---|---|---|---|
| §0 | Setup | — | 8 constants | params tunable |
| §1 | Load details/profiles + drop weak cols | raw CSV | DataFrame in-memory | details 29→19, profiles 10→5 |
| §3 | Streaming clean ratings → parquet | 4.2 GB CSV | 302 MB parquet | drop unknown + orphan + dedup; −324K (−0.26%) |
| §4 | Bot detection (4 rules) | parquet | `bad_users` set | **936 bad users** catch được |
| §5 | Iterative k-core | parquet − bad_users | `(users_final, anime_final)` | converge 2 iter; 292,591 × 22,821 |
| §6 | Write final CSV | filtered | 3 file `cleaned-data/` | 120,032,917 ratings |
| §7 | Before/after stats | both CSV | bảng + 3 plots | sparsity 98.20% |
| §9 | Audit (empty-list, cross-tab, sanity) | cleaned CSV | diagnostic | confirm rule áp đúng |

---

## 5. §0 Parameters & rationale

| Param | Value | Rationale |
|---|---|---|
| `USER_MIN` | 10 | Tối thiểu để embedding user ổn định. Dưới 10 ratings, noise dominate signal. |
| `ANIME_MIN` | 20 | Standard cho collab signal stable (industry 10–50). Anime <20 không học embedding tin cậy. |
| `KCORE_MAX_ITER` | 20 | Safety cap; thực tế converge 2 iter. |
| `SPAM_COUNT` | **2,000** | Data-driven: trong subset users với `mean_score > 9`, p99.87 của `rated_count` ≈ 2,138. Cắt 2,000 catch 36 extreme outlier. Base là `rated_count` (rows có `score > 0`) để khớp với `mean_score`. |
| `SPAM_MEAN` | 9.0 | Anime đạt mean > 9 theo global MAL consensus chỉ vài trăm bộ. User rate 2,000+ với mean > 9 = không organic. |
| `CONST_COUNT` | 500 | Dưới 500 entries rated thì std<0.3 có thể organic (fan rate vài chục anime tủ). Trên 500 + std<0.3 = mass-add. Base là `rated_count`. |
| `CONST_STD` | 0.3 | Strict hơn 0.5 để giảm FP. Real user rate 9-10 cũng có std ~0.7. std<0.3 với 500+ rated ≈ identical scores. |
| `WATCHED_MAX` | **5,000** | Empirical p99.99 của `watched_count` post-cleaning = 4,615, max = 4,947. Cắt 5,000 buffer ~8%. Anime ~5h × 5000 = 25K giờ ≈ 11 năm 6h/ngày → bất khả thi. |
| `NAN_RATER_MIN_WATCHED` | 1,000 | User mark >1000 anime "đã xem" nhưng không rate gì = không organic. Evidence: case `dumnorix98` (MAL thật 250 rated, data sai 5,928 watched NaN — scraper bug). |
| `DROP_UNKNOWN_STATUS` | True | 16,690 rows scraper error, không signal. |
| `DEDUP_KEEP` | "last" | Chỉ ~6 cặp duplicate, impact ~0. |

---

## 6. §1 Column drops

### `details` — drop 10 cols, giữ 19

| Drop | Lý do |
|---|---|
| `title_japanese` | Metadata-only, không phải feature |
| `url`, `image_url` | Pointer raw data |
| `explicit_genres` | Adult content đã lọc khỏi catalog, cột này gần empty |
| `licensors` | Noise (entity phân phối, không liên quan content) |
| `streaming` | Noise (platform list, không stable) |
| `end_date` | 61% null, redundant với `start_date + episodes×7d` cho TV |
| `year` | 78% null, derivable từ `start_date.year` |
| `season` | 78% null, derivable từ `start_date.month` |
| `producers` | 53% empty. Producers ≠ studio (production committee, sponsor). Signal đã có gián tiếp qua `members`/`score`/`studios`. High cardinality + multi-label + noisy |

### `profiles` — drop 7 cols, giữ 3

`watching, completed, on_hold, dropped, plan_to_watch` — counter aggregate per-status. Đã có raw trong `ratings.csv`, không cần aggregate sẵn.

`birthday, location` cũng bị drop — không dùng làm feature (user tower chỉ ăn `gender` + `joined`).

### `ratings` — drop 2 cols, giữ 4

`is_rewatching` (flag không stable), `num_watched_episodes` (chỉ có cho status=watching, sparse).

---

## 7. §3 Ratings filters (streaming)

1 pass streaming trên 4.2 GB `ratings.csv` (polars lazy + `engine="streaming"`):

1. **Drop `status=unknown`** (16,690 rows): scraper artifact.
2. **Drop orphan**: user/anime không trong profiles/details (giữ relational consistency).
3. **Dedup** `(username, anime_id)` keep last: chỉ ~6 cặp.
4. **Sink → parquet intermediate** (`/tmp/ratings_intermediate.parquet`, 302 MB): §4/§5 re-read parquet nhanh hơn CSV.

**Result**: 124,298,357 → 123,973,931 (**Δ −324,426, chỉ −0.26%**).

> ⚠️ §3 là filter rẻ — drop nhỏ. Bulk ratings reduction xảy ra ở **§4 bot drop** (−3.8M, −3.08%) và §5 k-core (−120K). Xem §10 Key results cho cascade đầy đủ.

---

## 8. §4 Bot/spam detection — 4 rules

Aggregate per-user stats từ parquet:
```python
.group_by("username").agg(
    pl.len().alias("count"),
    (pl.col("status") != "plan_to_watch").sum().alias("watched_count"),
    (pl.col("score") > 0).sum().alias("rated_count"),
    pl.col("score").filter(pl.col("score") > 0).mean().alias("mean_score"),
    pl.col("score").filter(pl.col("score") > 0).std().alias("std_score"),
)
```

Apply 4 mask, union vào `bad_mask`.

### Rule 1 — Bot dương (36 users, exclusive 11)

`rated_count > 2000 AND mean_score > 9.0`

**Catch**: bot mass-rate-10. Vd `ReMightyRon` (28,250 rated, mean 9.99, std 0.03).

**Vì sao**: real user có mean cao nhất ~8.5 dù toàn fan cuồng. mean > 9 với 2,000+ rated = không organic.

**Vì sao base là `rated_count` không phải `count`**: `mean_score` chỉ tính trên `score > 0`. Dùng `count` (incl. PTW) → mismatch base → false-positive cho user nhiều PTW.

### Rule 2 — Constant rater (175 users, exclusive 150)

`rated_count > 500 AND std_score < 0.3`

**Catch**: mass-1/5/10 raters. Bot chấm cùng 1 điểm hoặc dao động cực nhỏ.

**Vì sao**: std < 0.3 với 500+ rated yêu cầu rating gần identical. Fan "chỉ rate 9-10" cũng có std ~0.7.

**Note**: count tụt mạnh so với version cũ (1,084 → 175) vì base đổi từ `count` (incl. PTW) → `rated_count`. Rule cũ catch oan user nhiều PTW + ít rated điểm thấp std.

### Rule 3 — Physical impossibility (238 users, exclusive 211)

`watched_count > 5000`

`watched_count` = số records có `status != plan_to_watch` (đã thực sự touch anime).

**Catch**: bot "khôn" — mean/std realistic nhưng count physical impossible. Vd `DepaOhDepa` (28,468 watched, mean 5.48, std 1.39 — looks human).

**Vì sao 5,000**: empirical p99.99 = 4,615, max post-cleaning = 4,947. Buffer ~8%. 5000 anime × 5h ≈ 25K giờ ≈ 11 năm 6h/ngày.

### Rule 4 — NaN-rater (538 users, exclusive 521)

`watched_count > 1000 AND mean_score is null`

**Catch**: user mark >1000 anime là "đã xem" nhưng không chấm điểm bất kỳ → `mean_score` ra null.

**Evidence cụ thể** — case `dumnorix98`: MAL profile thật 250 rated + 600 PTW, dataset sai 5,928 watched + NaN mean → scraper phình ~10x. Drop.

### Tổng kết — rule contribution

| Rule | Total | Exclusive | % exclusive |
|---|---|---|---|
| Bot dương | 36 | 11 | 31% |
| Constant rater | 175 | 150 | 86% |
| Physical | 238 | 211 | 89% |
| NaN-rater | 538 | 521 | 97% |
| **Bad union** | **936** | — | — |

→ Mỗi rule có exclusive contribution rõ rệt — **không rule nào redundant**. Overlap chỉ 43 user catch bởi ≥2 rule. `spam_pos` post-fix vẫn catch 11 user mới mà 3 rule khác bỏ lọt → SPAM_COUNT=2,000 đã giải quyết được vấn đề redundancy với physical cap ở version cũ.

---

## 9. §5 K-core algorithm

```
Loop:
    drop user có count(ratings) < USER_MIN (=10)
    drop anime có count(ratings) < ANIME_MIN (=20)
    nếu không drop thêm gì → converge, break
```

**Converge**: 2 iter.

**Kết quả**:
- Initial sau drop bad_users: 120,152,637 pairs
- Iter 1: 292,591 users × 22,821 anime × 120,032,917 pairs
- Iter 2: identical → converge

**Vì sao 2 iter đủ?** §4 đã catch nhóm power-user có ratings rải khắp catalog. Sau khi loại họ, một số anime niche tụt dưới ANIME_MIN ngay iter 1; iter 2 verify không có cascade.

**Materialization**: chỉ load 2 cột `(username, anime_id)` — vài trăm MB, fit RAM dễ dàng.

---

## 10. §7 Key results — before/after

```
metric                           before            after        delta
---------------------------------------------------------------------
n_users (profiles)              337,155          292,591      -13.22%
n_users (in ratings)            309,314          292,591       -5.41%
n_anime (details)                28,955           22,821      -21.18%
n_anime (in ratings)             29,271           22,821      -22.04%
n_ratings                   124,298,357      120,032,917       -3.43%
sparsity                       98.6271%         98.2024%       -0.43pp
mean ratings/user                401.85           410.24       +2.09%
mean ratings/anime             4,246.47         5,259.76      +23.86%
```

### Ratings cascade — đóng góp từng stage

| Stage | Rows after | Δ rows | % drop |
|---|---|---|---|
| Raw | 124,298,357 | — | — |
| Sau §3 (unknown + orphan + dedup) | 123,973,931 | −324,426 | −0.26% |
| Sau §4 (drop 936 bad_users + ratings của họ) | 120,152,637 | −3,821,294 | −3.08% |
| Sau §5 k-core (drop niche anime) | 120,032,917 | −119,720 | −0.10% |
| **Final** | **120,032,917** | **−4,265,440** | **−3.43%** |

→ Bulk reduction là **§4 bot drop**, không phải §3 unknown/orphan filter.

### Catalog cut 21% (loại 6,134 anime)

Anime mất chủ yếu **niche** (members < 100, ratings < 20). Sau khi loại bot, niche anime mất 1–3 ratings → tụt dưới `ANIME_MIN=20` → drop.


### So với industry

Netflix Prize cut 5–20% noise, MovieLens 1–5% — 3.43% là **bình thường, không khắt khe**.

---

## 11. §9 Audit findings

> Phần này là audit **để xác nhận rule cleaning áp đúng** (không phải bản đồ phân phối đầy
> đủ — cái đó ở [DATA_DISTRIBUTIONS.md](DATA_DISTRIBUTIONS.md)).

### §9a Empty-list false-zero (cleaned details, 22,821 rows)

```
column          % NaN  % '[]'  % missing (NaN+[])
genres           0.00   14.03               14.03
studios          0.00   28.44               28.44
themes           0.00   37.99               37.99
demographics     0.00   71.12               71.12
```

**Gotcha**: 4 cột list-string lưu Python repr `"['a','b']"`. Empty là `"[]"` không phải NaN → `pandas.isna()` báo 0% là false-zero.

Notebook đã có helper `missing_pct()` ở §0:
```python
def missing_pct(df):
    for col in df.columns:
        if col in LIST_COLS_AS_NULL:
            out[col] = (s.isna() | s.eq("[]")).mean() * 100
        else:
            out[col] = s.isna().mean() * 100
```

### §9b Cross-tab — empty có correlate obscurity?

```
=== members trung vị: empty vs non-empty ===
  genres        empty=     453  non-empty=   3,860  ratio= 8.5x
  studios       empty=     470  non-empty=   6,495  ratio=13.8x
  themes        empty=   1,792  non-empty=   3,412  ratio= 1.9x
  demographics  empty=   2,375  non-empty=   3,874  ratio= 1.6x
```

**Insights**:
- **`studios` empty 13.8x members**: empty = anime niche không có studio metadata.
- **`demographics` ratio 1.6x**: KHÔNG correlate obscurity. Demographics là **structural property của manga gốc** (Shounen/Shoujo/Seinen/Josei/Kids — từ tạp chí Nhật). Anime original / LN / web novel / game không có demographic. 71% null là **expected**, không phải data quality issue.

```
=== % empty theo anime type ===
            genres  studios  themes  demographics
TV             4.7     10.3    29.7          58.8
Music         72.7     75.0     0.0          83.7  ← themes 0%!
```

**Music + themes redundancy**: MAL auto-tag `theme="Music"` cho mọi anime type=Music. Khi feature-engineer multi-hot themes, `theme=Music` sẽ collinear 1:1 với `type=Music` → drop sau multi-hot.

### §9c Post-cleaning sanity

**watched_count** percentiles:
```
  p50    →    203
  p99    →  1,670
  p99.9  →  3,271
  p99.99 →  4,615
  max    →  4,947  (≤ 5,000 ✓)
```

**rated_count** percentiles:
```
  p50    →    141
  p99    →  1,435
  p99.9  →  2,834
  p99.99 →  4,294
  max    →  6,523
```

Note: `rated_count max = 6,523 > SPAM_COUNT=2,000` không vi phạm Rule 1 vì user đó có `mean_score ≤ 9` (rule là AND).

**4-rule verification post-cleaning** (expect 0 each):
```
  spam_pos    (rated>2000 & mean>9.0): 0
  const_rater (rated>500 & std<0.3):   0
  physical    (watched>5000):          0
  nan_lister  (watched>1000 & no rate): 0
```

→ Tất cả 4 rule không còn user nào hold post-cleaning. Cleaning áp đúng.

---

## 12. Known limitations

### Không có timestamp trong ratings

Schema cuối `(username, anime_id, status, score)` — không có `rated_at`. Hệ quả:
- Không temporal-split được (train/val/test theo thời gian)
- Bắt buộc random split hoặc leave-one-out per user
- Chấp nhận leakage nhẹ; ở scale 120M ratings, impact nhỏ

### Catalog cut 21%

Anime mất chủ yếu niche. Với <20 interactions, two-tower không học được embedding tin cậy. Acceptable cho training; cold-start nên giải quyết ở serving:
- Index content embedding (synopsis + genres + studios) cho cold anime
