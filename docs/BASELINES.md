# BASELINES — các mốc so sánh của stage Retrieval

Doc tổng hợp cho `retriever/baselines/` — **mỗi baseline hoạt động thế nào, vì sao chọn, đọc số ra sao**. Baselines phía ranker (linear logistic, NN DIN) nằm ở `docs/RANKER.md §7` — không lặp ở đây.

> ⚠️ Số liệu = snapshot **2026-06-17** (đo trên TEST, protocol v2). Cập nhật 2026-06-17: (1) thêm
> **liked-metric** (liked-recall/liked-ndcg, report-only — `docs/LIKED_METRIC.md`) cho mọi baseline;
> (2) **HP-tune trên VAL** cho 3 baseline personalized (MF / itemknn / content) — chọn config theo val
> rồi report test (§3). Baselines không personalized (random/popular/meta_popular) tất định theo data,
> số binary **không đổi** so với 2026-06-11 (đã dùng làm sanity gate cho liked plumbing). File kết quả
> gốc: `retriever/baselines/*.txt`. Bảng so sánh mới nhất: `PROGRESS.md` + `docs/RESULTS.md`.

---

## 1. Vai trò

Two-tower chỉ có ý nghĩa nếu thắng được các phương pháp đơn giản hơn trên **cùng một protocol**. Bộ baseline phủ 3 nấc:

1. **Sàn** — random (mọi method phải >> nó; còn là sanity check harness).
2. **Non-personalized** — popular / meta_popular (bài kiểm tra "popularity bias": metric recall thưởng item phổ biến, model cá nhân hoá phải thắng cả khi không cá nhân hoá gì).
3. **Personalized cổ điển** — content-based, itemknn, **MF ALS (bar chính warm)** — trả lời "two-tower có đáng so với CF/content truyền thống không?".

Ràng buộc chung do **cold-user split** (hold trọn user khỏi train): mọi baseline KHÔNG được học tham số per-user — lúc eval chỉ được nhìn support history của user lạ, giống hệt user-tower. Vì vậy MF phải dùng fold-in, itemknn dùng similarity item-item (không cần user vector).

## 2. Harness chung — `_eval.py` (protocol v2, khớp y hệt `metrics.evaluate`)

Mỗi baseline chỉ cần cấp 1 hàm `score_fn(u, hist) -> scores [E, N]`; toàn bộ vòng eval dùng chung:

- **Cùng eval users + queries + seen** với two-tower (`metrics.load_eval_protocol`): warm (val/test) và cold ({val,test}_cold).
- **Cùng history**: prefix `eval_history_cap=1024` của list full sort score desc (pad 0) — apples-to-apples với user-tower.
- **Cùng mask**: candidate mask `isfinite(logq)` (loại PAD/OOV) + seen-mask `seen − query_đang_chấm` set −inf.
- **Cùng metrics**: recall@K / ndcg@K mean-per-user (K ∈ {10,50,100,200,500}, IDCG chuẩn hoá `min(R,K)`); cold thêm pooled hitrate@K.

→ Số baseline và số two-tower so được 1:1, không có chuyện mỗi method một protocol. Output mỗi script là 1 file text trong `retriever/baselines/` (header ghi rõ protocol + ngày + device).

## 3. Từng method

### 3.1 `rand.py` — Random (sàn, warm + cold)

Điểm uniform ngẫu nhiên (seed cố định) cho mọi candidate. Có công thức giải tích `recall@K ≈ K/N_cand` in kèm trong output — sai lệch giữa số đo và giải tích là sanity check cho harness.

### 3.2 `popular.py` — MostPopular (warm only)

`score(i) = count(i trong TRAIN examples)` — giống nhau cho mọi user. Đếm trên train (không leak test). Là bar "không cá nhân hoá": recall của nó cao bất ngờ (r@200 .4516) vì phân phối xem anime rất đầu nặng — model nào thua nó là chưa học được gì ngoài popularity.

### 3.3 `meta_popular.py` — Meta-Popular (warm + cold)

`score(i) = log1p(members)` từ metadata `details.csv` (số người add vào list trên MAL tại thời điểm scrape) — KHÔNG đụng interactions train nên định nghĩa được cho cả item H. Trên cold đây là baseline "cứ gợi ý anime mới đang hype cho mọi người" mà model phải vượt.

### 3.4 `content_based.py` — Content (mean + IDF, warm + cold)

- Mỗi item → 1 content-vector: concat các block multi-hot/one-hot từ feature đã encode (genres 21 + themes 52 + one-hot type/source/rating/demographics/start_year/episodes, bỏ id 0 = null; studios multi-hot, giữ OOV-studio). Cột được **IDF down-weight** (`log(n_real/(df+1))+1` — tag phổ biến nhẹ đi), rồi L2-normalize từng item (dim ~423).
- User profile = **mean** content-vector của history (masked, bỏ pad), L2-norm; score = cosine(profile, mọi item). Không train gì (IDF tất định).
- Vai trò kép: (a) warm — "two-tower có hơn pure content-similarity không?"; (b) **cold — comparator chính**: content vector của H tồn tại đầy đủ, đây đúng là kịch bản gợi ý anime mới bằng content.

### 3.5 `itemknn.py` — ItemKNN (item-item cosine, warm; bar CF "nhẹ")

- "Train" = dựng ma trận user×item binary từ TRAIN positives → similarity item-item cosine, giữ top-K neighbor mỗi item (`implicit.CosineRecommender` — đếm co-occurrence, không học per-user).
- Score user lạ: `score(i) = Σ_{j∈history} cosine(i, j)` — hợp lệ với cold-user vì chỉ cần history.
- **Tune (2026-06-17)**: sweep K ∈ {50,100,200,500,1000} trên val (recall@200) → **K=50 thắng** (val .6583, so với .5698 ở K=200 cũ). K nhỏ = neighborhood sạch hơn, đẩy mạnh recall sâu. Số test §5 là K=50.

### 3.6 `mf.py` — MF ALS + fold-in (**bar chính warm**)

- Train **implicit ALS** trên ma trận user×item binary từ TRAIN positives → `item_factors`.
- Cold-user: `user_factors` của eval user không tồn tại → **fold-in**: giải lại user-vector từ support history bằng đúng công thức ALS (`model.recalculate_user`) — không học tham số per-user nào của test. Score = `u · item_factorsᵀ`.
- **Tune (2026-06-17)**: TRAIN có ~67.5M interaction → 1 full fit ~4'(factors=64)–9'(factors=128), quá tốn để sweep nhiều combo. Quy trình: **coarse-sweep factors∈{64,128} × alpha∈{1,10,40}** (reg=.05, iters=15) trên **subset 15k user ngẫu nhiên** (fold-in chỉ cần item_factors → eval val đầy đủ vẫn hợp lệ), rồi **refit config thắng trên FULL train** report test. Hạ nhiệt CPU: `num_threads=4` + `OPENBLAS_NUM_THREADS=1` (tránh oversubscribe BLAS↔implicit). factors=256 bị bỏ (full refit ~17' — vượt budget; mở lại nếu cần capacity cao hơn).
- **Per-axis report** (chốt với user 2026-06-17): MF báo ở config tốt nhất MỖI trục — **ndcg-optimal** (factors=128, α=1) và **recall-optimal** (factors=128, α=10). Phát hiện chính: alpha (confidence weighting) đánh đổi head↔tail — α cao → recall sâu hơn nhưng ndcg@10 tụt; **config gốc cũ (factors=64, α=1 mặc định) KHÔNG ở mức tốt nhất** — chỉ cần factors 64→128 (α=1) đã vượt nó cả 2 trục.

- **Cơ chế đánh đổi recall↔ndcg (phân tích cho đồ án)**: ALS tối thiểu `Σ c_ui·(p_ui − xᵤ·yᵢ)²` với confidence `c_ui = 1 + α·r_ui`; data binary → item đã xem confidence `1+α`, chưa xem `1`. **α = tỉ lệ trọng số "fit positive đã quan sát" vs "tôn trọng các số 0"**:
  - α **thấp**: entry-0 còn nặng → factorization kéo về cấu trúc low-rank **toàn cục** (gradient *popularity × chất lượng*), đặt đúng title hay-phổ-biến lên đỉnh → **ndcg@10/r@10 cao** nhưng under-fit niche → recall sâu thấp.
  - α **cao**: positive thống trị loss → fit gắt sở thích riêng từng user → kéo nhiều item cụ thể vào top-200 (**recall@200 ↑**) nhưng top-10 mất prior toàn cục → **ndcg@10 tụt**. (Tương tự bias–variance.)
  - `factors` thì **không** đánh đổi — dung lượng thuần, mã hoá đồng thời global + niche → cải thiện cả 2 (f128 > f64 mọi α) tới khi diminishing returns.
- **Có config tối ưu cả 2 không?** Fine-sweep α tại f128 (val, subset 15k). Số fine-sweep:

  | α (f128, val) | r@10 | r@100 | r@200 | ndcg@10 |
  |---|---|---|---|---|
  | **1** | **.2115** | .6103 | .7109 | **.7093** |
  | 2 | .2079 | .5938 | .7284 | .7052 |
  | 3 | .2111 | .6155 | .7354 | .7031 |
  | 5 | .2075 | **.6179** | .7405 | .6846 |
  | 7 | .2029 | .6165 | .7412 | .6633 |
  | 10 | .1964 | .6124 | **.7418** | .6339 |
  | 15 | .1860 | .6040 | .7376 | .5867 |
  | 20 | .1769 | .5964 | .7330 | .5458 |

- Đây là CF matrix-factorization chuẩn, mạnh nhất trong bộ — two-stage (retriever+ranker) phải thắng nó thì kiến trúc mới đáng.

## 4. Cold slice — method nào đo được

| Method | Cold? | Vì sao |
|---|---|---|
| random, meta_popular, content | ✅ | không phụ thuộc interactions train — score của item H định nghĩa được |
| popular | ❌ N/A | popularity train của mọi item H = 0 by construction → không bao giờ lọt top-K |
| itemknn | ❌ N/A | H không có co-occurrence train → similarity = 0 |
| mf | ❌ N/A | item_factors của H không được học (H cách ly khỏi train) |

Đây không phải thiếu sót đo đạc mà là **kết quả cấu trúc**: CF thuần không thể gợi ý item chưa có interaction — chính là lý do two-tower (content path + id→OOV backoff) có lợi thế cold. Các file `.txt` ghi rõ "N/A by construction", không bịa số 0.

## 5. Kết quả hiện tại (TEST, snapshot 2026-06-17)

**Warm** (mask seen−query, history cap 1024, ~14.25k users). Cột `lr@k`/`lndcg@10` = liked-metric
(report-only; `n_users_liked` = 12,638 mọi hàng warm — user có ≥1 liked query):

| method | r@10 | r@100 | r@200 | r@500 | ndcg@10 | lr@100 | lr@200 | lndcg@10 |
|---|---|---|---|---|---|---|---|---|
| random | .0005 | .0044 | .0088 | .0219 | .0022 | .0046 | .0092 | .0011 |
| content (mean+IDF) | .0368 | .1577 | .2344 | .3779 | .0945 | .1580 | .2356 | .0495 |
| meta_popular | .0848 | .3198 | .4387 | .6156 | .3362 | .4085 | .5367 | .2456 |
| popular | .0865 | .3321 | .4516 | .6279 | .3527 | .4309 | .5568 | .2629 |
| itemknn (K=50) | .1177 | .4976 | .6592 | .8231 | .4638 | .5875 | .7403 | .3395 |
| **MF ndcg-opt** (f128/α1) | **.2087** | .5954 | .7136 | .8405 | **.7027** | .6960 | .7986 | **.5052** |
| **MF recall-opt** (f128/α10) | .1982 | **.6223** | **.7511** | **.8797** | .6374 | **.7213** | **.8328** | .4442 |

**Cold** (test_cold, full-catalog): content r@100 .1320 / r@200 .2177 / hit@500 .3784 (liked: lr@200 .2103 / lndcg@10 .0219) · meta_popular r@200 .0999 / hit@500 .1559 (lr@200 .1466) · random r@200 .0086 · popular/itemknn/mf = N/A.

Đọc số (so với two-tower hiện tại — chi tiết `PROGRESS.md` / `docs/RESULTS.md`):
- **Warm — MF là bar RẤT mạnh sau tune**: recall-opt MF r@200 **.7511**, ndcg-opt MF ndcg@10 **.7027**.
  Config gốc cũ (f64/α1: .6989 / .6771) bị f128/α1 vượt cả 2 trục → MF cũ chưa ở mức tốt nhất.
- **So với two-stage** (retriever + xendcg, test ndcg@10 **.7074**): vẫn **> MF ndcg-opt .7027** nhưng
  biên rất sát (+.0047, trước đây +.0303 vs MF cũ .6771). Ở **tail**, MF mạnh hơn two-stage rõ rệt
  (r@200 .7511 vs two-stage .6524, kẹt trần pool K=200) — tail recall là việc của retriever. ⇒ Lợi thế
  thật của 2-stage **không** nằm ở "thắng MF trên warm" (giờ sát/thua tail) mà ở **cold-start**:
- **Cold**: two-tower r@200 .3881 (val_cold) ≈ **1.8–2.2×** content (.2177 test_cold) trong khi MF/KNN/popular = **0 (N/A by construction)** — đây là claim cấu trúc mạnh nhất: không model CF cổ điển nào gợi ý được anime mới, kể cả MF đã tune.
- liked-metric: warm `lr@k` thường > `recall` binary (item user *thật sự thích* được surface tốt hơn item chỉ-tương-tác); MF ndcg-opt lndcg@10 .5052 cao nhất bộ.
- recall@K bị trần lý thuyết khi user có nhiều query hơn K — đọc `docs/DATA_SPLIT.md §8` trước khi so số tuyệt đối ở K nhỏ.

## 6. Chạy lại

```bash
venv/bin/python retriever/baselines/<rand|popular|meta_popular|content_based|itemknn|mf>.py
# mf/itemknn có --smoke (grid + subset rút gọn) để thử nhanh; output ghi đè *.txt cùng thư mục
# mf.py nặng (~17' full run, 2 refit per-axis) → nên chạy nền; đã cap num_threads=4 cho đỡ nóng máy
```

Mọi script đọc `retriever/train-data/` qua `src/` (import flat) — chạy sau khi prep xong; không cần GPU (MF/KNN dùng `implicit` trên CPU). 3 baseline personalized tune-on-val: lựa chọn HP ghi trong header `.txt` + section "val sweep".
