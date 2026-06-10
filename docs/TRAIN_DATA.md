# Train-Data Pipeline (Two-Tower Retrieval)

Doc tổng hợp cho `retriever/data_prep/` — **build cái gì, set up thế nào, vì sao**. Biến `cleaned-data/` → artifacts sẵn-sàng-train ở `retriever/train-data/`. Split + định nghĩa support/query: `docs/DATA_SPLIT.md`.

---

## 1. Overview

**Mục tiêu**: input cho stage Retrieval (Two-Tower): (a) labels positive `completed∪watching` + hard-neg score≤4, (b) split 2 trục (cold-user + cold-item H), (c) history lưu FULL, (d) seen-set đầy đủ cho eval mask. Mọi transform tất định đóng băng ra file; chỉ embedding được học nằm trong model.

**Kết quả** (chạy 2026-06-10, `99_verify` ALL PASS):

| | |
|---|---|
| Positives | 77.96M (`completed∪watching` & score∉[1,4]) |
| Hard-neg | 8.03M (dropped 4.24M ∪ score≤4 mọi status) |
| Users giữ (n_pos≥1) | 291,001 — train 262,676 / val 14,058 / test 14,267 |
| Items | 22,823 = 22,821 anime + PAD + OOV |
| **Cold H** | 1,142 anime (5% mới nhất, cutoff **2024-09-30**, 291 null-date loại) |
| Examples warm | train 67,456,284 / val 751,026 / test 748,751 |
| **Examples cold** | val_cold 150,335 (8,388 users) / test_cold 145,691 (8,510 users) |
| History full | p50 168 / max 4,893 / 316 user rỗng |
| eval_seen | 28,325 eval users / 11.9M pairs (mọi status, kể cả PTW) |

---

## 2. Labels (nguồn duy nhất: `data_prep/prep_config.py`)

- **Positive** = `status ∈ {completed, watching}` & `score ∉ [1,4]` (giữ score 0 = chưa chấm và 5..10). *watching* là engagement thật; *plan_to_watch* KHÔNG vào positive (intent/hype, không phải experience).
- **Hard-neg** = `dropped ∪ (score ∈ [1,4] mọi status)` — "bỏ dở" + "xem và ghét". Sàn tuyệt đối ở 4: KHÔNG dùng quantile per-user (score 7 = "Good" không phải negative).
- Hai tập **rời nhau by construction** (positive đòi score∉[1,4]).
- Đổi labels để ablate: sửa `prep_config.py` → re-run 02→06 (n_pos đổi → split membership đổi theo, chấp nhận). Định nghĩa được echo vào `feature_spec.json["labels"]`.

## 3. Split — 2 trục overlay

### 3.1 Cold-user
Hold-out **trọn user** 90/5/5, tất định `hash(username, SEED=42) % 100`; eval cần `n_pos ≥ 11`; user 1..10 positive về train; n_pos=0 drop.

### 3.2 Cold-item H
**H = 5% anime mới nhất theo `start_date`** (chọn ở `01`; null date → loại khỏi candidacy; tie cùng ngày phá bằng mal_id — tất định). *Newest chứ không random*: mô phỏng đúng "anime vừa ra, chưa có interaction lúc train"; random holdout đo kịch bản không tồn tại.

**H cách ly 4 chỗ** (làm ở `05`, assert ở `05`+`99`): (1) train examples, (2) history MỌI user, (3) hard_neg pools, (4) support eval. **Positive-H của eval user → toàn bộ vào cold query** (`examples/split={val,test}_cold`); positive-H của train user vứt hẳn.

Cùng tập eval user phục vụ 2 bộ đo: **warm** (tuning, headline recall@200) và **cold** (final exam — test_cold chấm 1 lần lúc cuối; val_cold để debug). Eval-user có thể 0 warm query (mọi positive đều H) — history rỗng → h_empty, vẫn có cold query.

### 3.3 Support/query warm (trên warm pool)
Eval user: warm positives chia query (random tie-hash, `n_query = clip(round(0.2·n_warm), 1, n_warm−1)`; `n_warm<2` → 0 query) và support (→ history). Leak assert = 0.

## 4. Output — artifacts `retriever/train-data/`

```
train-data/
├── feature_spec.json          # single source of truth (vocab/dim/special + labels + cold meta)
├── anime_id_map.parquet       # mal_id ↔ anime_idx (0=PAD, 1=OOV, real 2..N+1)
├── user_id_map.parquet        # username ↔ user_idx
├── cold_items.parquet         # ★ anime_idx + mal_id + start_date của H
├── item_features.parquet      # 1 row/anime_idx — 9 feature encoded
├── users.parquet              # gender/joined + history_ids/scores FULL + hard_neg_ids(≤64)
├── eval_seen.parquet          # ★ eval user → seen_ids (MỌI status, nguồn seen-mask)
├── examples/split={train,val,test}/            # warm positives
├── examples/split={val_cold,test_cold}/        # ★ cold queries (positive-H của eval user)
├── logq.parquet (+ logq.npy)  # logQ từ TRAIN warm; floor max(count,1) → H finite, vẫn candidate
└── _user_stats/_user_split/_user_feats/_spec_*.{parquet,json}   # intermediates
```

**`users.parquet`**: `user_idx(Int32)`, `split(str)`, `gender_id(Int8)`, `joined_bucket(Int8)`, `history_ids(List[Int32], FULL — sort score desc, tie hash asc → prefix = top-by-score)`, `history_scores(List[Int8], cùng len/order)`, `hard_neg_ids(List[Int32], ≤64, −H)`. Eval user: history = **support warm** (đã trừ query + H).

**`eval_seen.parquet`**: `user_idx`, `seen_ids(List[Int32], unique sorted)` — mọi interaction mọi status. Protocol eval: **mask = seen − query_đang_chấm** (query ⊆ seen nên không được mask thẳng seen).

`item_features` schema + encoding 9 feature (vocab 10/18/7/6/6/8/22/53/302, bucket edges từ audit): nguồn sống là `data_prep/01_item_features.py` + `data_audit/output/`.

## 5. Pipeline — 6 script + verify

`retriever/data_prep/`, chạy tuần tự; constants + labels TẬP TRUNG ở **`prep_config.py`** (02/03/05/06 import — hết drift). 2 pass streaming `ratings.csv` (02, 05).

| # | Script | Vai trò |
|---|---|---|
| 01 | `01_item_features.py` | encode 9 feature item + **chọn cold H** → `cold_items.parquet` |
| 02 | `02_user_counts.py` | đếm n_pos/n_hardneg theo labels (từ prep_config) |
| 03 | `03_split.py` | cold-by-user 90/5/5 (constants từ prep_config) |
| 04 | `04_user_features.py` | gender/joined |
| 05 | `05_history_examples.py` | 1 pass: (pos∪hardneg) ∪ (mọi row eval-user) → **eval_seen** + **cold queries** + H-isolation + **history FULL** + hard_neg−H |
| 06 | `06_logq_and_spec.py` | logQ (từ TRAIN warm) + feature_spec (labels echo, cold meta, history_store=full) |
| 99 | `99_verify.py` | id/align/leak checks + sorted-desc history + **H-isolation 4 chỗ** + cold examples ⊆ H đúng split + seen ⊇ history∪queries + cold logq count=0 |

```bash
for s in retriever/data_prep/0[1-6]_*.py; do venv/bin/python "$s"; done
venv/bin/python retriever/data_prep/99_verify.py
```

## 6. Design decisions (vì sao)

- **Mask seen đầy đủ**: nếu chỉ mask một phần history thì hàng chục–trăm item user ĐÃ xem vẫn chiếm slot top-K và đếm là miss → mọi số tuyệt đối bị đè thấp, lệch khỏi serving (service filter cả list MAL). Mask = seen − query_đang_chấm.
- **History FULL**: user trung vị có ~170 positives — cap tĩnh lúc prep (vd top-30) bỏ đói user vector ở cả train (không augmentation) lẫn eval/serve. Prep lưu hết; **cap nằm ở src** (`train_hist_len` sample/anchor lúc train — augmentation; `eval_history_cap` prefix lúc eval). User cực đoan (>2000): ragged storage lo memory, eval cắt prefix 1024 (~p99).
- **Cold-item H + encode OOV**: model dùng `use_item_id` → metric cold-user thuần KHÔNG đo được khả năng gợi ý anime mới; id-dropout/OOV backoff phải được kiểm chứng bằng slice riêng. Eval cold encode H bằng **id→OOV** (serve-path thật của item ngoài vocab) — encode id thật (random chưa train) = đo noise.
- **logQ**: floor `max(count,1)` tự cho H (count=0) finite → H vẫn là candidate; correction có hệ số `logq_alpha` ở loss (tune được).

## 7. Runtime contract (src đọc gì)

- `feature_spec.json`: vocab/dim/special_idx (dựng tower) + `hard_neg_cap` + `eval_history_cap_default` + `cold_items`/`cold_examples` meta.
- History: ragged (UserTable đọc list cols → values+offsets). Train: sample `train_hist_len`/anchor, gỡ anchor, dropout. Eval: prefix `eval_history_cap`.
- `eval_seen.parquet` → `metrics.build_masks(seen, queries)` = mask per slice.
- `cold_items.parquet` → `data.load_cold_mask` → `model.refresh_item_cache(cold_mask=...)` cho cold eval.
- `logq.npy` → InfoNCE correction (`logq_alpha`) + candidate mask (`isfinite`).
