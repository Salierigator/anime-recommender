# Train-Data Pipeline (Two-Tower Retrieval)

Doc tổng hợp cho `scripts/build_train_data/` — giải thích **build cái gì, set up thế nào, vì sao**. Biến `cleaned-data/` → artifacts sẵn-sàng-train ở `train-data/`.

---

## 1. Overview

**Mục tiêu**: chuẩn bị input cho giai đoạn **Retrieval (Two-Tower)**. Output là data thuần + 1 file config; **chưa đụng model** (PyTorch towers/embedding/loss là task riêng, cần `pip install torch`).

**Philosophy**: tách bạch **data artifact** (cố định, dùng chung train+serve) vs **model parameter** (được học). Mọi transform tất định + thống kê đóng băng ra file; chỉ ma trận embedding được học nằm trong model.

**Kết quả** (chạy thật trên 120.03M ratings):

| | |
|---|---|
| User giữ | 285,388 / 292,591 (drop 7,203 = 2.46% zero-positive) |
| Positive | 72.6M (completed & score ∉ [1,4]) |
| Hard-neg (dropped) | 4.24M |
| Split (cold-by-user) | train 257,970 / val 13,628 / test 13,790 (90.4 / 4.8 / 4.8%) |
| Examples | train 65,334,420 / val 728,759 / test 724,991 (66.8M) |
| Items | 22,823 = 22,821 anime + PAD + OOV |

---

## 2. Câu hỏi kiến trúc: pre-compute file vs nằm trong model

Nguyên tắc phân định: thành phần **không đổi khi trọng số model đổi** → data artifact; thành phần **được học** → model.

| Thành phần | Thuộc về | Lý do |
|---|---|---|
| Re-index `anime_id`/`user_id` | **Data artifact** | Song ánh cố định, tính 1 lần, dùng chung mọi bảng + serve. Là *input* của embedding. Nhúng trong code train → train/serve skew. |
| Encoding feature (category→id, bucket, multi-hot, bảng đã encode) | **Data artifact** | Transform tất định, frozen vocab. → `item_features.parquet`, `users.parquet`, `feature_spec.json`. |
| Ma trận embedding (`nn.Embedding`), `h_empty`, MLP, τ | **Model** (task sau) | Được học → trong checkpoint. Pipeline chỉ pre-compute *vocab size* + *dim* để model dựng layer. **KHÔNG** lưu embedding vector ra file. |
| Bảng `logQ` | **Data artifact** | Thống kê tất định của TRAIN split (tần suất item-as-positive). Groupby 1 lần, kiểm soát leak. |

**Cầu nối** = `feature_spec.json`: vocab sizes + dims + special idx. Metadata để model dựng layer khớp encoding. *Điểm dễ nhầm:* "trích xuất embedding" thực ra là trích xuất **chỉ số đã encode + spec vocab** (data), KHÔNG phải vector (vector của model).

---

## 3. Output — artifacts `train-data/`

```
train-data/
├── feature_spec.json          # single source of truth: vocab map + size + dim + special idx + params
├── anime_id_map.parquet       # mal_id ↔ anime_idx (real)
├── user_id_map.parquet        # username ↔ user_idx
├── item_features.parquet      # 1 row / anime_idx (gồm PAD=0, OOV=1)
├── users.parquet              # 1 row / user (feature + history + hard_neg)
├── examples/split={train,val,test}/part-0.parquet   # (user_idx, anime_idx), toàn positive
├── logq.parquet  (+ logq.npy) # dense theo anime_idx
└── _user_stats/_user_split/_user_feats.parquet, _spec_item/_spec_user.json   # intermediates (giữ để re-run khỏi stream lại 120M)
```

### Schema

**`item_features.parquet`** (mọi gather idx 0..N+1 hợp lệ):

| col | dtype | ghi chú |
|---|---|---|
| `anime_idx` | Int32 | 0=PAD, 1=OOV, real 2..22822 |
| `type_id` | Int8 | vocab 10 |
| `source_id` | Int8 | vocab 18 |
| `rating_id` | Int8 | vocab 7 |
| `demographics_id` | Int8 | vocab 6 (single-tag) |
| `startyear_bucket` | Int8 | 6 bucket |
| `episodes_bucket` | Int8 | 8 bucket |
| `genres_multihot` | List[Int8] | width 22 (21 tag + present) |
| `themes_multihot` | List[Int8] | width 53 (52 tag + present) |
| `studio_ids` | List[Int32] | vocab 302, avg-pool ở model |

**`users.parquet`**: `user_idx(Int32)`, `split(str)`, `gender_id(Int8, vocab 4)`, `joined_bucket(Int8, 6)`, `history_ids(List[Int32], ≤30)`, `history_scores(List[Int8], cùng len/order)`, `hard_neg_ids(List[Int32], ≤64)`.

**`examples/`**: `user_idx(Int32)`, `anime_idx(Int32, luôn ≥2)`. Toàn positive → không có cột label. `split` là partition key.

**`logq.parquet`**: `anime_idx(Int32)`, `count(Int64)`, `is_candidate(Bool)`, `log_q(Float32)`. PAD/OOV → `-inf`, `is_candidate=False`. `logq.npy` = vector float32 length 22823 align anime_idx (load thẳng torch).

---

## 4. Pipeline — 6 script + verify

`scripts/build_train_data/`, chạy theo thứ tự, mỗi script standalone & re-runnable. Convention khớp `scripts/`: `ROOT = Path(__file__).parent.parent.parent`, pandas cho file nhỏ, polars `scan_csv(...).collect(engine="streaming")` cho `ratings.csv`. **2 streaming pass** trên 120M rows (02, 05); còn lại đọc artifact nhỏ.

| # | Script | Input | Output | Key |
|---|---|---|---|---|
| 01 | `01_item_features.py` | `details.csv` (pandas) | `item_features`, `anime_id_map`, `_spec_item.json` | encode 9 item feature; vocab 10/18/7/6/6/8/22/53/302 |
| 02 | `02_user_counts.py` | `ratings.csv` (Pass-1) | `_user_stats` | per-user n_pos/n_dropped |
| 03 | `03_split.py` | `_user_stats` | `_user_split`, `user_id_map` | cold-by-user 90/5/5, stable-hash |
| 04 | `04_user_features.py` | `profiles.csv`, `_user_split` | `_user_feats`, `_spec_user.json` | gender_id, joined_bucket |
| 05 | `05_history_examples.py` | `ratings.csv` (Pass-2), maps | `users.parquet`, `examples/` | history/hard_neg + support/query (chống leak) |
| 06 | `06_logq_and_spec.py` | `examples/split=train`, specs | `logq`, `feature_spec.json` | merge spec + logQ |
| 99 | `99_verify.py` | `train-data/` | — | check schema/range/leak end-to-end |

Chạy:
```bash
for s in scripts/build_train_data/0[1-6]_*.py; do venv/bin/python "$s"; done
venv/bin/python scripts/build_train_data/99_verify.py
```

---

## 5. Design decisions

### 5.1 Labeling (§ plan.md)
- **Positive** = `status=="completed"` & `score ∉ [1,4]` (giữ `score==0` = chưa chấm, và 5..10). Chỉ vứt completed bị chấm 1–4.
- **Hard-neg** = toàn bộ `status=="dropped"` (kể cả dropped score cao — bỏ dở là tín hiệu mạnh hơn điểm).
- Bỏ qua v1: `plan_to_watch`, `watching`, `on_hold`.
- **Vì sao**: định nghĩa không phụ thuộc hành vi chấm điểm → unbiased với user "chỉ chấm bộ thích" lẫn "chỉ chấm bộ ghét".

### 5.2 Re-index
- **anime_idx**: `0=PAD` (pad tensor history/hard_neg), `1=OOV/MASK` (id-dropout lúc train + cold serve), real `2..N+1` sort theo `mal_id`. OOV là vector **học được**.
- **user_idx**: `0..U-1` trên user giữ (sort username). User-id embedding **drop ở v1** (cold-by-user hold-out trọn user → id user lạ vô dụng) → user_idx chỉ là join key, không special slot.

### 5.3 Split — cold-by-user (chống leak)
- Hold out **trọn user** 90/5/5. Eval (val/test) chỉ nhận user `n_pos ≥ 11` (đủ chia support/query); user 1–10 positive luôn về train. Drop `n_pos==0` (1-positive thì GIỮ → supervise `h_empty`).
- Split tất định bằng `hash(username, SEED) % 100` (reproducible, không RNG; eval-ineligible rớt val/test → train).

### 5.4 History + support/query (build trong 05)
- **Train user**: `history_ids` = top-30 theo score trên TOÀN positive pool; mọi positive cũng là example. Runtime gỡ anchor: `history = history_ids − {anchor}` → rỗng thì `h_empty`. 1-positive → history rỗng sau gỡ.
- **Eval user**: positives chia 1 lần thành **query** (random theo tie hash, `n_query = clip(round(0.2·n_pos), 1, n_pos−1)`) = examples, và **support** (phần còn lại) → `history_ids` = top-30 trên support. Query KHÔNG bao giờ nằm trong history → **không leak** (đã assert `leak == 0`).
- `history_scores` lấy từ **cùng các dòng đã sort** với `history_ids` → 2 list song song, không re-join. Tie giữa `score==0` phá bằng stable hash (reproducible).
- `hard_neg_ids` = dropped item của user, dedup + sort + cap 64 (runtime sample m≈1–5).

### 5.5 Feature encoding (note.txt)

| Feature | Kind | Vocab/Width | Dim | Missing/OOV |
|---|---|---|---|---|
| type | cat | 10 | 4 | nan → OOV(0) |
| source | cat | 18 | 8 | `Unknown` là tag **thật**; unseen → OOV(0) |
| rating | cat | 7 | 4 | nan → OOV(0) |
| demographics | cat (single-tag) | 6 | 4 | empty → `none`(0); single-tag lấy tag đầu (54 row có 2 tag); closed set, không OOV |
| start_year | bucket | 6 | 4 | NULL(0) + 5 era; year tương lai luôn vào bucket → không cần OOV |
| episodes | bucket | 8 | 4 | NULL/OOV(0) + 7 range |
| genres | multi-hot | 22 | — | 21 tag + 1 present (anime rỗng genres) |
| themes | multi-hot | 53 | — | 52 tag + 1 present |
| studios | multi-value | 302 | 16 | 0=empty, 1=OOV, 2..=300 studio (occ≥10); avg-pool list |
| gender | cat | 4 | 4 | nan → OOV(0) |
| joined | bucket | 6 | 4 | NULL(0) + 5 cohort |

Bucket edges copy **verbatim** từ `scripts/details_audit/` (`start_date` ERA_BINS, `episodes` BUCKET_BINS) và `scripts/profiles_audit/audit_joined.py` (COHORT_BINS) để khớp audit. PAD/OOV item row: mọi feature = neutral 0 (id OOV/NULL, multi-hot toàn 0, `studio_ids=[0]`).

### 5.6 logQ correction
- Từ **TRAIN examples only** (chống leak): `Q(item) = count/total`, `log_q = log(max(count,1)/total)`. Floor `max(·,1)` để 871 anime chỉ xuất hiện ở val/test support vẫn finite.
- PAD/OOV → `log_q = -inf`, `is_candidate=False`. Verify `sum(exp(log_q))` trên count>0 = **1.00000000**.
- **Vì sao static**: in-batch negative lấy đúng từ phân phối positive này → Q empirical là chuẩn; groupby 1 lần rẻ, training loop đơn giản & reproducible.

---

## 6. Runtime contract (model task sẽ dùng)

- Anime embedding: `nn.Embedding(num_items=22823, dim, padding_idx=0)`. Vocab+dim mỗi feature đọc từ `feature_spec.json`.
- `logq.npy` → tensor, trừ vào in-batch logits (InfoNCE + logQ correction + temperature).
- Mỗi anchor: `history = history_ids − {pos_item}` rồi pool (mean v1); rỗng → `h_empty`. Pad history/hard_neg bằng idx 0, mask `!= 0`.
- Hard-neg per anchor: sample m từ `hard_neg_ids` của *chính user*; pad + mask −∞; user không có dropped → loss thuần in-batch.
- Mask false-negative cho in-batch (2 anchor trùng pos_item).
- History dropout ~10–15%: random bỏ history dùng `h_empty` (supervise + cold-by-user).

---

## 7. Verify (`99_verify.py`, chỉ đọc `train-data/`)

1. `item_features` rows == `num_items`; mọi `*_id ∈ [0, vocab)`; multihot width 22/53, giá trị 0/1; studio id ∈ [0,302).
2. `users` rows == `num_users`; `len(history_ids)==len(history_scores) ≤ 30`; gender/joined/hard_neg in-range.
3. `examples` mọi `anime_idx ≥ 2`, `user_idx ∈ [0,num_users)`.
4. `logq` length == `num_items`; real item finite; `sum(exp(log_q))` count>0 = 1.0.
5. (trong 05) leak check: eval example ∩ history = ∅.
