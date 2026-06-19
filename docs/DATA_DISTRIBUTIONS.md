# DATA_DISTRIBUTIONS.md — Schema & phân phối dữ liệu

> **Phạm vi:** mô tả schema + phân phối thống kê của dữ liệu **sau làm sạch** (tầng
> `cleaned-data/`) — đây là input thực tế của mọi model. Pipeline raw→cleaned & lý do bỏ
> cột/lọc bot: [CLEANING.md](CLEANING.md). Cách các cột này được *encode thành feature*
> (bucket, vocab, OOV): [TRAIN_DATA.md](TRAIN_DATA.md).
>
> **Nguồn số:** toàn bộ bảng dưới trích từ `data_audit/output/**` (script audit chạy trực
> tiếp trên `cleaned-data/`, xác nhận trong `data_audit/codes/**`). Không đọc trực tiếp data
> thô (luật repo §0). Snapshot 2026-06-19.

## 1. Ba bảng & schema (sau làm sạch)

| Bảng | #dòng | Khoá | Cột (cleaned) |
|---|---|---|---|
| `details.csv` | **22.821** anime | `mal_id` | mal_id, title, type, status, score, scored_by, start_date, synopsis, rank, popularity, members, favorites, genres, studios, themes, demographics, source, rating, episodes (**19 cột**) |
| `profiles.csv` | **292.591** user | `username` | username, gender, joined (**3 cột**) |
| `ratings.csv` | **120.032.917** tương tác | (username, anime_id) | username, anime_id, status, score (**4 cột**) |

Raw có nhiều cột hơn (details 29→19, profiles 10→3, ratings 6→4); danh sách cột bị bỏ +
lý do ở [CLEANING.md](CLEANING.md). `genres/studios/themes/demographics` lưu dạng
list-as-string (`"['Action','Fantasy']"`), `[]` = rỗng (KHÔNG phải NaN — xem cảnh báo §2.4).

---

## 2. `details.csv` — phân phối từng cột
(nguồn: `data_audit/output/details_audit/audit_<col>.txt`; n=22.821)

### 2.1 Numeric / điểm số
| Cột | Null | Phân phối chính |
|---|---|---|
| `score` | 4.000 (17.53%) | range [1.89, 9.29], mean 6.39 ±0.89; p25 5.77 / p50 6.37 / p75 7.03 / p95 7.87. Không có giá trị ngoài [1,10]. |
| `scored_by` | (kèm score) | đếm số người chấm; long-tail (dùng làm feature `log_scored_by`). |
| `members` | 0 | range [43, 4.230.312], median **2.592**, mean 49.138 (lognormal nặng đuôi): p75 16.618, p95 237.489, p99 873.642. log10 p50≈3.41. |
| `popularity` | 0 | rank toàn cục MAL, range [1, 28.925], 17.726 giá trị unique (≈rank-based, gần đều), median 11.427. |
| `favorites` | 0 | long-tail (feature `log_favorites`). |
| `rank` | có null đáng kể | hạng chất lượng MAL; null khi chưa đủ điểm (feature `rank` + `rank_missing`). |
| `episodes` | 652 (2.86%) | range [1, 3.000], mean 11.3, p50 **2**, p75 12, p95 49, p99 104. **48.98% là 1-tập** (movie/one-shot/music), 10.11% là 12-tập (1 cua). 19 anime >500 tập (long-running: max 3.000). |

### 2.2 Categorical đơn trị
| Cột | Null | Cardinality | Top giá trị |
|---|---|---|---|
| `type` | 65 (0.28%) | 9 | TV 27.10% · OVA 16.94% · Movie 16.43% · ONA 14.12% · Music 11.75% · Special 7.27% · TV Special 3.12% · CM 1.93% · PV 1.05% |
| `source` | 0 | 17 | Original 34.69% · Manga 23.72% · Unknown 8.43% · Game 5.96% · Light novel 5.25% · Visual novel 5.08% · … |
| `status` | 0 | 3 | Finished Airing **96.39%** · Not yet aired 2.34% · Currently Airing 1.26% |
| `rating` | 543 (2.38%) | 6 | PG-13 42.31% · G-All Ages 29.25% · PG-Children 7.07% · Rx-Hentai 6.94% · R-17+ 6.80% · R+ 5.25% |

### 2.3 Multi-label (list tag)
| Cột | Rỗng `[]` | #tag unique | Top tag | Đặc điểm long-tail |
|---|---|---|---|---|
| `genres` | 3.201 (14.03%) | 21 | Comedy 6.845 · Action 5.373 · Fantasy 4.713 · Adventure 3.717 · Sci-Fi 3.145 | list p50=2, max 7; 42.7% non-empty là single-tag |
| `themes` | 8.670 (37.99%) | 52 | Music 3.727 · School 2.158 · Historical 1.493 · Mecha 1.195 | top-10 phủ 60.27% mass; list p50=1, max 6 |
| `demographics` | **16.231 (71.12%)** | 5 | Kids 2.723 · Shounen 2.143 · Seinen 1.106 · Shoujo 512 · Josei 160 | 99.18% single-tag; 71% rỗng là **cấu trúc** (chỉ manga-derived có) — KHÔNG phải lỗi |
| `studios` | 6.490 (28.44%) | **1.237** | Toei 899 · Sunrise 580 · J.C.Staff 439 · Madhouse 381 | đuôi rất dài: 300 studio có count≥10 phủ **86.53%** non-empty rows (cơ sở chọn vocab — xem §5) |

### 2.4 `start_date` & `synopsis`
- `start_date`: null 291 (1.28%); năm [1917, 2027], p50 **2013**, p95 2024. Phân phối lệch
  hiện đại (2010+ chiếm ~61%). **Cảnh báo scraper-default:** tháng 1 chiếm 15.39%, ngày 1
  chiếm 12.54%, Jan-1 (year-only fallback) 6.88% → ngày/tháng KHÔNG đáng tin, chỉ dùng **năm**.
  Era bucket (toàn catalog): ≤1989 10.03% · 1990–99 9.23% · 2000–09 18.26% · 2010–17 28.15%
  · 2018+ 33.05% · NULL 1.28%.
- `synopsis`: null 1.384 (6.06%), non-empty 21.437; độ dài p50 294 ký tự / 50 từ, max 3.750
  ký tự. 32.92% có chuỗi `(Source: …)` (cần strip trước khi feed text encoder), ~0.12%
  placeholder "No synopsis…". (Synopsis từng thử làm feature embedding nhưng **bị bác** —
  [SYNOPSIS_EMB.md](SYNOPSIS_EMB.md).)

> ⚠️ **`[]` ≠ NaN.** Với genres/themes/demographics/studios, ô rỗng lưu literal `[]`
> (NaN=0). Audit đếm `[]` là "missing"; khi load phải parse list, không nhầm `[]` thành 0.

---

## 3. `profiles.csv` — phân phối
(nguồn: `data_audit/output/profiles_audit/`; n=292.591)

- `gender`: **NaN 48.15%** (140.870). Có giá trị: Male 110.503 (37.8%) · Female 37.438
  (12.8%) · Non-Binary 3.780 (1.3%). → encode kèm slot "unknown".
- `joined`: null 0.54%; năm [2004, 2025], p50 2020. Cohort: ≤2012 6.95% · 2013–16 17.10% ·
  2017–19 19.89% · 2020–21 26.22% · 2022+ 29.30% · NULL 0.54%.
  **Phát hiện audit (quan trọng cho thiết kế feature):** cohort cũ hoạt động mạnh hơn hẳn
  (median #rating: ≤2012 = 406 vs 2022+ = 145) và `corr(join_year, mean_anime_year)=+0.473`
  → `joined` **dư thừa một phần với history** (history bắt hộ tín hiệu seniority/era). Vẫn
  giữ làm cold-start prior (user mới ít history). Ngày-trong-tháng phân phối phẳng (peak nhẹ
  day 23, 3.46%) ⇒ chỉ dùng năm/cohort, không dùng ngày.

---

## 4. `ratings.csv` — phân phối
(nguồn: `data_audit/output/ratings_audit/`; n=120.032.917)

### 4.1 `status` (toàn bộ dòng)
completed **63.56%** (76.29M) · plan_to_watch 25.78% (30.94M) · watching 4.50% (5.40M) ·
dropped 3.54% (4.24M) · on_hold 2.62% (3.15M).

### 4.2 `score` — `score=0` là "chưa chấm", KHÔNG phải điểm 0
- `score=0` (unrated): **43.45%** (52.15M); đã chấm (≥1): 56.55% (67.88M); null: 0.
- Trong số đã chấm: score 7 (13.33%), 8 (14.03%), 9 (8.54%), 10 (5.71%); 1–5 ~10.7% tổng.
- `score≥7` (định nghĩa positive): 49.95M = 41.61% tổng / **73.58% trong số đã chấm**.
- **status × score** (vì sao score sống chủ yếu ở `completed`):

| status | %score=0 | %đã chấm | %pos(≥7) | mean(đã chấm) |
|---|---|---|---|---|
| completed | 16.23% | 83.77% | 63.08% | 7.40 |
| plan_to_watch | 99.40% | 0.60% | 0.45% | 7.49 |
| watching | 78.73% | 21.27% | 17.07% | 7.68 |
| dropped | 54.80% | 45.20% | 6.21% | **4.55** |
| on_hold | 77.13% | 22.87% | 15.81% | 7.06 |

### 4.3 Positive pool / user (positive = score≥7, mọi status)
mean 170.7/user, p50 108, p90 403. User có ≥1 positive: 275.504 (**94.16%**); ≥10: 87.09%;
≥20: 82.00%. Mở rộng positive ra mọi status (không chỉ completed) thêm +3.79% positive và
cấp ≥1 positive cho thêm 67.98% user. (Định nghĩa positive/hard-neg đầy đủ:
[DATA_SPLIT.md](DATA_SPLIT.md).)

### 4.4 Low-raters (kiểm tra "dislike" thật vs nhiễu)
600/276.369 user (0.217%) chỉ chấm ≤5 — median 1 rating (mean 15.4 vs toàn cục 245.6).
Score nhóm này: 1 (28%), 2 (36%), 3 (12%), 4 (12%), 5 (12%); status 79.4% completed, 13.7%
dropped → tín hiệu **không thích rõ ràng**, củng cố việc dùng score≤4 làm hard-negative.

---

## 5. Từ phân phối → quyết định feature (tóm tắt — chi tiết [TRAIN_DATA.md](TRAIN_DATA.md))
- **Era bucket** (≤1989/1990–99/2000–09/2010–17/2018+) lấy từ phân phối `start_date`, chỉ
  dùng năm vì ngày/tháng là scraper-default.
- **Episodes bucket** (1 / 2 / 3–6 / 7–13 / 14–26 / 27–52 / 53+) bám đỉnh phân phối (48.98%
  một-tập, đỉnh 12–13 tập).
- **Studios vocab**: giữ tag count≥10 → **300 studio** (phủ 86.53% rows), còn lại → OOV
  (bảng quyết định ngưỡng ở `audit_studios.txt`).
- **`members/scored_by/favorites`** lognormal → dùng `log1p` (feature `log_*`).
- **`gender/joined`** có slot "unknown" do null cao / cold-start.
