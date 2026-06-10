# Ranker (GBDT Reranking)

> **⚠ STALE** — toàn bộ số đo trong file này theo **protocol v1 + artifacts v1** (best.pt cũ). Retriever đang rebuild (data v2 + protocol v2, xem `PROGRESS.md`); khi retriever v2 chốt phải export lại artifacts + retrain ranker + đo lại. Nội dung thiết kế (label/candidate/feature/blend) vẫn đúng để tham khảo.

Doc thiết kế cho `ranker/src/` — **build cái gì, set up thế nào, vì sao**. Stage 2 của recommender: rerank top-K cosine của retriever bằng feature giàu hơn (popularity, quality, recency, affinity nội dung) mà embedding khó biểu diễn rẻ. Ăn artifacts firewall `artifacts/` + feature thô `cleaned-data/`, đẻ ra `artifacts/ranker.txt`.

---

## 1. Overview

**Mục tiêu**: stage Ranking — cho 1 user + tập candidate (top-N của retriever), học hàm điểm rerank sao cho item user thật sự thích nổi lên đầu. Headline metric = **two-stage cold-by-user** `ndcg@10` / `recall@10`, đo trên user `val`/`test` (xem §7). Baseline phải vượt = **chỉ-retriever** (thứ tự cosine), val `ndcg@10` ≈ 0.251 (xem `artifacts/CONTRACT.md`).

**Vì sao GBDT (LightGBM LambdaRank)**: rẻ mà đủ tốt — train CPU vài phút, xử lý **categorical native** (không one-hot phình to), model text nhỏ gọn portable (`ranker.txt`), mạnh trên tabular ranking ít feature. Slot contract `artifacts/ranker.txt` chính là format dump text của LightGBM, `requirements.txt` đã có lightgbm → model đã chốt sẵn. Không dùng deep reranker / DCN ở v1: thừa so với quy mô feature và phá mục tiêu "rẻ".

**Philosophy** (khớp firewall, `PROJECT_STRUCTURE.md §4`): ranker chỉ **ĐỌC** `artifacts/` + `cleaned-data/`, chỉ **import định nghĩa model** (`UserTower`) — KHÔNG import code train retriever, KHÔNG đọc `retriever/train-data/`. User history dựng lại từ `cleaned-data/` **cùng code path lúc serve** → không lệch train/serve. Feature set gọn (~30–40) xoay quanh **1 tín hiệu chủ đạo = cosine retriever**; phần còn lại là content/popularity/affinity rẻ.

---

## 2. Files — `ranker/src/`

```
ranker/
├── CLAUDE.md                 # guide mảng (firewall, layout, thứ tự chạy)
└── src/
    ├── user_encode.py        # dựng U (UserEncoder mirror UserTower) — DÙNG CHUNG với service
    ├── features.py           # bảng feature item (details.csv) + lắp ma trận — DÙNG CHUNG build/eval
    ├── build_dataset.py      # users → candidate → ma trận feature (parquet, group theo user)
    ├── train.py              # LightGBM LambdaRank → artifacts/ranker.txt + ranker_meta.json
    └── eval.py               # two-stage eval vs baseline chỉ-retriever
```

Convention path: `.py` trong `src/` dùng `ROOT = Path(__file__).resolve().parent.parent` (= `ranker/`); artifacts firewall ở `ROOT.parent/"artifacts"`, data thô ở `ROOT.parent/"cleaned-data"`. `UserTower` import từ `retriever/src/model.py` (thêm `sys.path`, chỉ định nghĩa class — không chạm train code).

---

## 3. Artifacts firewall ranker đụng tới

| File | Chiều | Nội dung dùng cho |
|---|---|---|
| `artifacts/item_vectors.npy` | ĐỌC | `[22823,128]` L2-norm, row==`anime_idx`. Tính cosine `U·V`, history-similarity, pool history cho UserTower. **Lọc `anime_idx≥2`** khi tạo candidate (0=PAD, 1=OOV). |
| `artifacts/item_index.parquet` | ĐỌC | `anime_idx:int32 ↔ mal_id:int64` (-1 cho PAD/OOV). Join item feature thô (keyed `mal_id`) ↔ vector row (keyed `anime_idx`). |
| `artifacts/user_tower.pt` | ĐỌC | state_dict user-side + metadata (`d`, `history_pool`, `score_pool`, `user_features` vocab/dim, `special_idx`). Load `UserTower` → encode U. Vocab/special-idx lấy từ đây, **không hard-code**. |
| `artifacts/user_split.parquet` | ĐỌC | `username, user_idx, split∈{train,val,test}`. **Bắt buộc** dùng chung để chia train/eval đúng cold-by-user (chống leak). |
| `artifacts/ranker.txt` | **GHI** | model LightGBM (text dump). |
| `artifacts/ranker_meta.json` | **GHI** | sidecar: list tên feature theo thứ tự, index categorical, `K_RETRIEVE`, timestamp + source-checkpoint (track drift; service đọc để dựng feature đúng thứ tự). |

> `artifacts/CONTRACT.md` do `retriever/export.py` tự sinh — **không sửa tay**. Slot `ranker.txt` đã có sẵn trong bảng contract; `ranker_meta.json` là phần mở rộng của riêng ranker, chỉ document ở đây.

Nguồn data thô (`cleaned-data/`, đọc read-only):
- `details.csv` (~22.8k dòng) — metadata anime. **pandas** (nhỏ). Cột: `mal_id, title, type, status, score, scored_by, start_date, synopsis, rank, popularity, members, favorites, genres, studios, themes, demographics, source, rating, episodes`.
- `ratings.csv` (~120M dòng, cỡ GB) — `(username, anime_id, status, score)`. **BẮT BUỘC polars lazy + streaming**, filter user trước rồi mới `.collect(engine="streaming")`.
- `profiles.csv` — `(username, gender, joined)`. pandas (nhỏ).

---

## 4. Label & candidate (lúc train)

### 4.1 Profile/target split per user (chống leak)

Mỗi user, lấy **positives** = tương tác dương từ `ratings.csv` theo đúng định nghĩa retriever: `status=completed AND score∉[1,4]` (giữ score 0 và 5..10). Chia rời 2 phần (chọn ngẫu nhiên theo tie hash — **cùng cơ chế support/query split của eval** ở §7, để train↔eval khớp):
- **profile set** — dựng U + history-similarity + affinity + user-stats.
- **target set** — held-out **TẤT CẢ positive** (không chỉ score≥7), chính là **label** (graded ở §4.2). Cap `TARGET_MAX=8`, giữ history ≥1.

Item đã vào target **không** được nằm trong history sinh ra feature của nó (mô phỏng query/profile split cold-by-user của retriever — xem `TWO_TOWER_MODEL.md §7`). Đây là điểm chống leak cốt lõi.

> **Vì sao target = tất cả positive, không chỉ score≥7**: eval coi *relevant* = mọi positive (completed & score∉[1,4]). Nếu chỉ train trên score≥7, ranker không học surface positive score 0/5/6 mà eval thưởng → recall@10 tụt. Sửa lại (align train↔eval) đẩy ndcg@10 từ +0.007 lên +0.038 và recall@10 từ âm sang +0.012 (xem §8).

### 4.2 Graded relevance (cho LambdaRank)

Relevance chia bậc theo score user chấm trên target dương (phủ TẤT CẢ positive, khớp *relevant* của eval):

| score | relevance |
|---|---|
| 0 / 5 / 6 (completed, relevant thấp) | 1 |
| 7–8 | 2 |
| 9 | 3 |
| 10 | 4 |
| negative (sample) | 0 |

LambdaRank tối ưu trực tiếp NDCG nên graded label > binary: tách "thích vừa" vs "tủ".

### 4.3 Candidate per user

`target positives` ∪ `hard-negative retriever` ∪ `random negative`:
- **hard-neg** (`HARD_NEG=40`) = top `U·item_vectors` **không** nằm trong positives (cả profile + target) + dropped của user. Buộc ranker học phân biệt *trong phân phối candidate của retriever* — đúng việc lúc serve. **Bỏ `HARD_SKIP=5` item top nhất**: top retriever ngoài positives thường là *false-neg* (item user sẽ thích nhưng chưa xem) → label 0 cho chúng dạy ranker chống lại recall.
- **random neg** (`RANDOM_NEG=40`) = sample từ pool item (lọc `anime_idx≥2`, bỏ item user đã seen) → phủ phần đuôi, chống overfit hard-neg.

**Subsample `N_TRAIN_USERS=30k` train + `N_VAL_USERS=6k` val** (GBDT cần ít hơn nhiều so với 65M example của retriever; thêm user → diminishing returns). Mỗi user = **1 query group** cho LambdaRank; val groups dùng early-stopping.

---

## 5. User vector U + history (dựng lại, sạch firewall)

`user_encode.py` — helper inference **dùng chung với `service/`** (cùng path → không lệch train/serve):

1. **History từ `ratings.csv`** (polars streaming, filter các user đã chọn): mỗi user lấy positives theo **đúng định nghĩa của retriever** = `status=="completed" AND score∉[1,4]` (giữ score 0 và 5..10 — **không** phải score≥7; score≥7 chỉ dùng cho *label* ở §4), sort `(score desc, tie asc)` với `tie=hash(struct(user_idx,anime_idx),seed=42)`, **top-30** → `history_ids` (map `mal_id→anime_idx` qua `item_index.parquet`) + `history_scores`. (Khớp `02_user_counts` / `05_history_examples` của retriever — xem `docs/TRAIN_DATA.md`.) Lúc train: bỏ item thuộc target set khỏi history.
2. **gender / joined từ `profiles.csv`**: encode theo map gender + cutoff bucket joined trong `docs/TRAIN_DATA.md`; vocab size + special-idx đọc từ metadata `user_tower.pt` (không hard-code).
3. **U = UserTower(pool_history(history_ids), gender, joined)**: load `UserTower` (import định nghĩa) với weight `user_tower.pt`, pool history qua `item_vectors.npy` (cùng `history_pool` đã train: `mean`/`attn`) → `U[128]` L2-norm.

User rỗng history (cold thật) → UserTower tự fallback `h_empty`. Service lúc serve cũng chạy đúng 3 bước này từ profile sống của user → ranker và service tiêu thụ cùng một U.

---

## 6. Feature (25 cột, gọn, xoay quanh tín hiệu retriever)

Mỗi row `(user, candidate-item)`. Bảng dưới = nguồn + xử lý null (theo `data_audit/output/`):

| Nhóm | Feature | Nguồn | Ghi chú |
|---|---|---|---|
| **Retriever** | `cos_uv = U·V_item` | `item_vectors` + U | **Feature số 1.** U,V đã L2-norm → dot = cosine. |
| **History-content** | `hist_cos_max`, `hist_cos_mean` | `item_vectors` (V candidate vs V history) | max/mean cosine candidate với item trong profile-history. Tín hiệu content-KNN, không cần tower. |
| **Quality/Popularity** | `mal_score`, `log1p(scored_by)`, `log1p(members)`, `log1p(favorites)`, `popularity`, `rank` | `details.csv` | `score`/`scored_by` ~17.5% null → impute (median) + **cờ `score_missing`**. `members` 0% null. Log-transform vì lệch nặng. |
| **Recency / format** | `episodes`, `recency_days`/`era_bucket` | `details.csv` `start_date` | `start_date` 1.28% null + ~12% ngày scraper-default (Jan-1) → **chỉ dùng year/era**, không dùng ngày thô. `episodes` 2.86% null → bucket/impute. |
| **Categorical (native)** | `type`, `source`, `rating`, `demographics`, `era_bucket` | `details.csv` | LightGBM `categorical_feature` — **không one-hot**. `demographics` 71% null → coi null là 1 category. |
| **Affinity nội dung** | `genre_affinity`, `theme_affinity`, `genre_overlap_cnt` | `details.csv` genres/themes + history | vector sở thích genre/theme của user (gộp multi-hot trên profile-history) · multi-hot candidate → 2 scalar (+ số tag trùng). Gói tín hiệu genre/theme **không** cần 73 cột multi-hot. |
| **User stats** | `u_n_rated`, `u_mean_score`, `u_std_score`, `u_account_age` | `ratings`/`profiles` (profile set) | Chỉ tính trên profile set (không đụng target → không leak). |

**Loại bỏ có chủ đích** (theo audit + mục tiêu "rẻ"):
- full multi-hot genres(21)/themes(52) — affinity scalar đã đủ, tránh phình cột.
- one-hot top-300 studio — 28% null, đuôi dài, lợi biên nhỏ với GBDT.
- `gender` thô làm feature mạnh — 48% null, tín hiệu yếu cho gu xem; đã gián tiếp qua U.

Tất cả tên feature + thứ tự + index categorical ghi vào `ranker_meta.json` để service dựng lại đúng.

---

## 7. Eval — two-stage cold-by-user (`eval.py`)

Trên user `val` (và `test` để báo cáo cuối) từ `user_split.parquet`:

1. Mỗi user: hold-out 1 **query set** positives, dựng U từ phần còn lại (§5).
2. **Retrieve**: score U vs toàn `item_cache`, mask item đã seen, lấy **top-N=200** (candidate pool).
3. **Rerank + blend**: tính feature (§6) cho 200 candidate → LightGBM `pred` → **blend sàn cosine**:
   `score = (1-α)·rank_norm(cos_uv) + α·rank_norm(pred)` (α∈{.3,.5,.7,1}; α=0 ⇒ cosine) → sort.
4. Đo `recall@{10,50,100}` / `ndcg@{10,50,100}` trên query set + **trần pool** (tỉ lệ query lọt top-200).

`eval.py` **sweep objective × α**, chọn cấu hình **Pareto-dominate cosine TRÊN VAL** (≥ cosine mọi K, tốt hơn ở head) rồi ghi `artifacts/ranker.txt` + `ranker_meta.json` (lưu `objective`, `blend_alpha`).

**Baseline = chỉ-retriever** (α=0). Candidate pool top-200 cố định → recall@200 đặt trần chung; baseline tái hiện đúng `ndcg@10`≈0.25, `recall@10`≈0.10 trong CONTRACT (so sánh công bằng).

### 7.1 Kết quả (best.pt provisional, n=3000/split)

**Production = `xendcg`, blend α=0.5, 100k train user.** Two-stage TEST:

| metric | baseline (cosine) | ranker v2 | Δ |
|---|---|---|---|
| **recall@10** | 0.102 | **0.125** | **+0.023** |
| **recall@50** | 0.279 | **0.291** | **+0.012** |
| recall@100 | 0.396 | 0.403 | +0.007 |
| **ndcg@10** | 0.246 | **0.295** | **+0.049** |
| ndcg@50 | 0.268 | 0.293 | +0.025 |
| **pool@200 recall ceiling** | — | **0.534** | (trần cứng của retriever) |

v2 **Pareto-dominate cosine** — tốt hơn ở MỌI K (head + tail), không còn đánh đổi của v1.

**Bài học**: chìa khoá là **blend sàn cosine (α)**, không phải objective. Ranker thuần (α=1) vẫn regress recall@50 ở *mọi* objective (giống v1 `lambdarank30`: recall@50 −0.041); blend lấy cosine làm sàn mới tiến đều cả list (α=0.5 sweet spot). Objective lambdarank30/200/xendcg gần ngang ở α=0.5, xendcg nhỉnh ở α cao. Chi tiết tình trạng + đòn bẩy tiếp: `ranker/CLAUDE.md §5`.

---

## 8. Thứ tự chạy

```
ranker/src/ (CWD)
1. build_dataset.py   # cleaned-data + artifacts → ma trận feature train/val (parquet, group theo user)
2. train.py           # LightGBM LambdaRank, early-stop trên group val → artifacts/ranker.txt + ranker_meta.json
3. eval.py            # two-stage eval val/test vs baseline chỉ-retriever
```

Retriever chốt artifacts mới → chạy lại cả 3 (CPU, vài phút). `user_encode.py` không chạy độc lập — `build_dataset.py`, `eval.py`, và `service/` import.
