# BASELINES — các mốc so sánh của stage Retrieval

Doc tổng hợp cho `retriever/baselines/` — **mỗi baseline hoạt động thế nào, vì sao chọn, đọc số ra sao**. Baselines phía ranker (linear logistic, NN DIN) nằm ở `docs/RANKER.md §7` — không lặp ở đây.

> ⏳ **PENDING — baselines đang re-run (2026-06-17)**: `retriever/baselines/` đang tune/đo lại (itemknn đổi K từ 200 → đang thử K=50, content IDF, MF rerun, +liked-metric). **Mọi số trong file + RESULTS.md §4 là bản cũ 2026-06-11, KHÔNG trích cho báo cáo cho đến khi re-run xong.**
>
> ⚠️ Số liệu trong file = snapshot **2026-06-11** (đo trên TEST, protocol v2). Baselines tất định theo data — chỉ đổi nếu re-run prep; số two-tower để so thì còn đổi khi tune tiếp → bảng so sánh mới nhất: root `PROGRESS.md` + `docs/RESULTS.md` (bản đồ nguồn mọi con số). File kết quả gốc: `retriever/baselines/*.txt`.

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

- "Train" = dựng ma trận user×item binary từ TRAIN positives → similarity item-item cosine, giữ top-K=200 neighbor mỗi item (`implicit.CosineRecommender` — đếm co-occurrence, không học per-user).
- Score user lạ: `score(i) = Σ_{j∈history} cosine(i, j)` — hợp lệ với cold-user vì chỉ cần history.

### 3.6 `mf.py` — MF ALS + fold-in (**bar chính warm**)

- Train **implicit ALS** (factors=64, iterations=15, reg=.05) trên ma trận user×item binary từ TRAIN positives → `item_factors`.
- Cold-user: `user_factors` của eval user không tồn tại → **fold-in**: giải lại user-vector từ support history bằng đúng công thức ALS (`model.recalculate_user`) — không học tham số per-user nào của test. Score = `u · item_factorsᵀ`.
- Đây là CF matrix-factorization chuẩn, mạnh nhất trong bộ — two-stage (retriever+ranker) phải thắng nó thì kiến trúc mới đáng.

## 4. Cold slice — method nào đo được

| Method | Cold? | Vì sao |
|---|---|---|
| random, meta_popular, content | ✅ | không phụ thuộc interactions train — score của item H định nghĩa được |
| popular | ❌ N/A | popularity train của mọi item H = 0 by construction → không bao giờ lọt top-K |
| itemknn | ❌ N/A | H không có co-occurrence train → similarity = 0 |
| mf | ❌ N/A | item_factors của H không được học (H cách ly khỏi train) |

Đây không phải thiếu sót đo đạc mà là **kết quả cấu trúc**: CF thuần không thể gợi ý item chưa có interaction — chính là lý do two-tower (content path + id→OOV backoff) có lợi thế cold. Các file `.txt` ghi rõ "N/A by construction", không bịa số 0.

## 5. Kết quả hiện tại (TEST, snapshot 2026-06-11)

**Warm** (mask seen−query, history cap 1024, ~14.25k users):

| method | r@10 | r@100 | r@200 | r@500 | ndcg@10 |
|---|---|---|---|---|---|
| random | .0005 | .0045 | .0093 | .0230 | .0025 |
| content (mean+IDF) | .0368 | .1577 | .2344 | .3779 | .0945 |
| meta_popular | .0848 | .3198 | .4387 | .6156 | .3362 |
| popular | .0865 | .3321 | .4516 | .6279 | .3527 |
| itemknn (K=200) | .1211 | .4105 | .5722 | .7979 | .4685 |
| **MF ALS-64 fold-in** | **.1951** | **.5759** | **.6989** | **.8352** | **.6771** |

**Cold** (test_cold, full-catalog): content r@100 .1320 / r@200 .2177 / hit@500 .3784 · meta_popular r@200 .0999 · random r@200 .0086 · popular/itemknn/mf = N/A.

Đọc số (so với two-tower hiện tại — chi tiết `PROGRESS.md`):
- **Warm**: MF là bar thật (r@200 .6989, ndcg@10 .6771). Two-tower một mình (`v5_hist64_ep2`: r@200 .6608, ndcg@10 .5135 — số checkpoint-path; serve-path .6524/.5155, phân biệt 2 biến thể: `docs/RESULTS.md §2`) còn thua MF — nhưng head precision là việc của ranker: **two-stage** (retriever + xendcg) đạt test ndcg@10 **.7074 > .6771** → kiến trúc 2-stage vượt bar.
- **Cold**: two-tower r@200 .3881 (val_cold) ≈ **1.8–2.2×** content (.2177 test_cold) trong khi MF/KNN/popular = 0 — lợi thế cấu trúc của two-tower, không model cổ điển nào trong bộ đạt được cả hai mặt warm + cold.
- recall@K bị trần lý thuyết khi user có nhiều query hơn K — đọc `docs/DATA_SPLIT.md §8` trước khi so số tuyệt đối ở K nhỏ.

## 6. Chạy lại

```bash
venv/bin/python retriever/baselines/<rand|popular|meta_popular|content_based|itemknn|mf>.py
# mf/itemknn có --smoke (subset 200k user) để thử nhanh; output ghi đè *.txt cùng thư mục
```

Mọi script đọc `retriever/train-data/` qua `src/` (import flat) — chạy sau khi prep xong; không cần GPU (MF/KNN dùng `implicit` trên CPU).
