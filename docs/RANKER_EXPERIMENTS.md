# RANKER_EXPERIMENTS — thử nghiệm chọn config ranker (pool `final`, ưu tiên ndcg@10 + liked)

> Nguồn viết đồ án cho phần **thử nghiệm chọn LightGBM ranker cuối** trên pool retriever `final`
> (no synopsis, re-export 2026-06-17). Bổ sung cho `docs/RANKER.md` (kiến trúc + data flow + protocol)
> và `docs/LIKED_METRIC.md` (định nghĩa liked). Mirror vai trò `docs/EXPERIMENTS.md` của retriever.
>
> ⚠️ **Trạng thái**: đây là phiên **EXPLORE** (2026-06-17) — chưa chốt/export. `artifacts/ranker.txt`
> vẫn STALE (`v5_hist64_ep2`). Sau khi phân tích bảng dưới → chốt config + train full + `export.py`
> (loop `docs/RANKER.md §9`) ở phiên sau.

## 0. Bối cảnh & câu hỏi

Retriever chốt `final` → pool two-stage đổi (warm val pool ceiling r@200 **.6758**, cosine ndcg@10 **.5343**;
`docs/RESULTS.md §3b`). Ranker cũ train trên pool `v5_hist64_ep2` → phải **train lại + dò config**. Mục tiêu
phiên này: **ưu tiên ndcg@10 (head precision) + liked metric** (item user *thật sự thích*, score > u_mean —
`docs/LIKED_METRIC.md`), tìm xem có config nào vượt winner cũ (`xendcg`, α=1) không.

Khoảng trống nhận ra từ code (động cơ thử nghiệm):
1. **Train graded (10→4…) nhưng selection BINARY ndcg@10**; liked metric chỉ **report-only**, chưa từng
   được tối ưu → thử **liked-aware label** (grade +1 nếu score > u_mean) để gắn training với liked.
2. **label_gain** (lambdarank) chưa tune; **truncation** điều khiển độ dồn gradient vào đầu list.
3. Early-stop dùng ndcg nội bộ binary `[10,100]` → thử focus `[10]`.

## 1. Phương pháp (đo apples-to-apples với retriever)

- **Pool/protocol**: dùng `artifacts/` của `final` → `ranker/data_prep/build_eval.py` (pools depth 500) +
  `build_train.py` (100k train user, support 80%/target 20%, top-200 pool, label graded). Sanity gate
  (`eval.py --baseline-only`) PASS: cosine tái lập `eval_reference.json` < 2e-3.
- **Đo**: two-stage `metrics.py` trên `eval_val` (14,029 user) — KHÔNG tin ndcg nội bộ LightGBM (chỉ để
  early-stop). Mỗi model sweep blend α ∈ {.25,.4,.5,.6,.75,1} → lấy α tốt nhất theo **ndcg@10**.
- **Coarse → confirm** (như baselines MF): dò config trên **subset 25k train group** (nhanh ~4–5'/run,
  4 threads đỡ nóng), rồi **confirm top trên full 100k**. Số coarse hơi thấp hơn full (ít train data) nhưng
  **xếp hạng config** đáng tin.
- **Selection**: ndcg@10 chính (Pareto vs cosine) + **liked_ndcg@10 / liked_recall@100 tie-break** (ưu tiên
  của phiên này). Mọi run log `models/<run>/row.json` → `models/leaderboard.csv` (seed 42, reproducible).

**Trục đã quét** (chi tiết `ranker/src/train_lgbm.py::SWEEP`): objective `rank_xendcg` vs `lambdarank` ×
`lambdarank_truncation_level` {10,20,30,50}; `label_gain` (reshape grade→gain); `num_leaves`/`min_data_in_leaf`/
`learning_rate`/`feature_fraction`; `ndcg_eval_at` {[10,100] vs [10]}. **Relabel** (`--relabel`): default vs
`steep` (10→5,9→4,8→3,7→2,else→1) vs `liked` (grade default +1 nếu score>u_mean) — qua cột `target_score`
(build_train mới) nên không rebuild data.

## 2. Baseline trên pool `final` (cosine = retriever-only, α=0)

| split | ndcg@10 | recall@10 | recall@100 | liked_ndcg@10 | liked_recall@100 |
|---|---|---|---|---|---|
| **val** (14,029) | .5343 | .1678 | .5388 | .3894 | .6491 |
| **test** (14,250) | .5323 | .1681 | .5387 | .3903 | .6445 |
| val_cold (8,388) | .1398 | .0710 | .3373 | .0920 | .4074 |

Pool ceiling (trần rerank, r@200 trong pool): val .6758 / test .6758. recall@200 two-stage không vượt được mức này.

## 3. Kết quả coarse sweep ĐẦY ĐỦ (subset 25k train, eval FULL val 14,029) — 16 config, 2026-06-17

> α best = **1.0** mọi config (model đủ tin override hẳn cosine — khác ranker cũ cần α=.5). **TẤT CẢ
> vượt xa cosine** (ndcg@10 .5343 → ~.68–.71; liked_ndcg@10 .3894 → ~.554–.563). Số gốc: `models/leaderboard.csv`.

| run | objective / lever | ndcg@10 | ndcg@100 | **liked_ndcg@10** | liked_r@10 | liked_r@100 | r@10 | iter | sec |
|---|---|---|---|---|---|---|---|---|---|
| **lrank_t20_gainLin** | lambdarank t20, label_gain=[0,1,2,3,4] | **.7134** | .6078 | .5567 | .2993 | .7196 | .2116 | 1995 | 530 |
| lrank_t20_gainExp | lambdarank t20, gain=[0,3,7,15,31] | .6996 | .6013 | .5608 | .3019 | .7217 | .2084 | 2609 | 661 |
| **xendcg_lr03** | rank_xendcg, lr=0.03 | .6982 | .6012 | **.5632** | **.3057** | .7201 | .2102 | 3995 | 488 |
| xendcg_es10 | rank_xendcg, early-stop ndcg@10 only | .6981 | .6012 | .5627 | .3045 | .7195 | .2095 | 2964 | 362 |
| xendcg_l127 | rank_xendcg, num_leaves=127 | .6963 | .6000 | .5614 | .3044 | .7188 | .2095 | 1464 | 217 |
| xendcg | rank_xendcg (winner cũ) | .6956 | .6005 | .5609 | .3038 | .7200 | .2089 | 1983 | 243 |
| lrank_t50 | lambdarank truncation 50 | .6955 | .6007 | .5593 | .3021 | .7218 | .2082 | 1570 | 557 |
| xendcg_ff09 | rank_xendcg, feature/bagging .9 | .6955 | .6004 | .5610 | .3038 | .7195 | .2092 | 1917 | 264 |
| lrank_t30 | lambdarank truncation 30 | .6950 | .5999 | .5585 | .3012 | .7221 | .2078 | 1621 | 504 |
| lrank_t20_l127 | lambdarank t20, num_leaves=127 | .6948 | .5985 | .5604 | .3028 | .7213 | .2075 | 1484 | 440 |
| lrank_t20 | lambdarank truncation 20 | .6944 | .5989 | .5593 | .3011 | .7216 | .2070 | 2072 | 531 |
| xendcg_l255_mdl50 | rank_xendcg, leaves=255, min_data=50 | .6940 | .5984 | .5598 | .3046 | .7188 | .2088 | 546 | 118 |
| xendcg_lr10 | rank_xendcg, lr=0.1 | .6914 | .5970 | .5580 | .3024 | .7166 | .2073 | 685 | 108 |
| lrank_t10_es10 | lambdarank t10, early-stop ndcg@10 | .6879 | .5951 | .5549 | .2997 | .7204 | .2051 | 1362 | 294 |
| lrank_t10 | lambdarank truncation 10 | .6874 | .5953 | .5546 | .2992 | .7210 | .2047 | 1295 | 282 |
| lrank_t20_gainTop | lambdarank t20, gain=[0,1,3,7,31] | .6803 | .5926 | .5544 | .3009 | .7216 | .2033 | 1820 | 471 |

**Đọc (coarse — confirm full ở §5):**
- 🥇 **ndcg@10**: `lrank_t20_gainLin` (label_gain **tuyến tính** [0,1,2,3,4]) dẫn rõ rệt (.7134, +.018 vs xendcg).
  Cơ chế: eval ndcg@10 **binary** (bất kỳ query item) — gain tuyến tính bớt dồn vào grade-4 (điểm 10) nên model
  xếp *mọi* positive lên đầu tốt hơn. Ngược lại `gainTop` ([0,1,3,7,31], dồn grade-4) **kém nhất** (.6803).
- 🥇 **liked_ndcg@10**: `xendcg_lr03` (lr thấp, ~4000 cây) cao nhất (.5632) + ndcg@10 hạng 3 (.6982) → **all-rounder
  tốt nhất**. Kế đến `xendcg_es10` (.5627). Nhóm xendcg (.561–.563) liked đều **cao hơn** nhóm lambdarank gainLin (.5567).
- **liked_r@10** (recall top-10 riêng item user thật sự thích) gần như **phẳng** (.299–.306) nhưng kể cùng câu chuyện *gắt hơn*:
  `xendcg_lr03` cao nhất (**.3057**), `lrank_t20_gainLin` (max ndcg@10) **thấp nhất** (.2993) — tức head-10 của gainLin nhồi
  "positive bất kỳ" thay vì item ưa thích. → liked_r@10 là tín hiệu nhạy hơn liked_ndcg@10 cho mục tiêu "đẩy item user thích lên đầu".
- ⚖️ **Căng thẳng ndcg@10 ↔ liked CÓ THẬT**: config max-ndcg@10 (gainLin) **không** phải config max-liked.
  liked thưởng item *score > u_mean* (preference-weighted); gainLin tối ưu binary-relevance → kéo "positive bất kỳ"
  lên đầu nhưng không riêng "item user thật sự thích" → liked thấp hơn nhóm xendcg.
- `truncation` (10→50) gần như phẳng (.687–.696, t10 hơi kém); `num_leaves`/`feature_fraction`/lr 0.1 đều quanh
  baseline; lr 0.03 (nhiều cây hơn) nhỉnh cả 2 trục → generalize tốt hơn. → đòn bẩy chính = **objective + label_gain + lr**, không phải capacity.

## 4. Relabel (liked-aware / steep) — KẾT QUẢ ÂM (không cải thiện liked)

Giả thuyết: grade label theo *tương đối user* (liked = score>u_mean) sẽ kéo liked_ndcg lên. **Bác bỏ:**

| run | relabel | ndcg@10 | liked_ndcg@10 | so với default cùng objective |
|---|---|---|---|---|
| xendcg | default | .6956 | .5609 | (mốc) |
| xendcg_liked | liked (+1 grade nếu score>u_mean) | .6872 | .5614 | liked +.0005 (nhiễu), ndcg@10 **−.0084** |
| xendcg_steep | steep (10→5,9→4,8→3,7→2,else→1) | .6859 | .5594 | liked −.0015, ndcg@10 −.0097 |
| lrank_t20 | default | .6944 | .5593 | (mốc) |
| lrank_t20_liked | liked | .6846 | .5576 | cả hai giảm |
| lrank_t20_steep | steep | .6857 | .5570 | cả hai giảm |

**Đọc**: liked-aware relabel **không** đẩy được liked_ndcg@10 (chênh trong nhiễu) mà còn **hạ ndcg@10** ~.008–.01.
Lý do: graded label sẵn có (10→4…) đã mã hoá đủ mức ưa thích; bơm +1 cho liked làm méo phân phối grade → loạn
ordering tổng thể. **liked metric phản ứng với ranking tổng thể tốt hơn (lr/early-stop), KHÔNG với relabel.** →
Giữ grading mặc định. (Kết quả âm này có giá trị báo cáo: đã thử hướng "gắn label với liked" và bác bằng số.)

## 5. CHỐT — confirm full + export (DONE 2026-06-18)

User chốt **`lrank_t20_gainLin`** (ưu tiên ndcg@10). Train full 100k LOCAL 4-thread (~37', 2.949 trees) → eval → export:
```bash
venv/bin/python ranker/eval.py --baseline-only                                 # sanity gate PASS
# train full chỉ gainLin (train_one trực tiếp, overrides lambdarank t20 + label_gain=[0,1,2,3,4])
venv/bin/python ranker/eval.py --models ranker/models/lrank_t20_gainLin/model.txt   # select + test + val_cold
venv/bin/python ranker/export.py && venv/bin/python -m pytest ranker/tests -q   # → artifacts/ranker.txt + meta (16 passed)
```

**Số full vs coarse 25k** (val, two-stage @ α=1.0): full nâng đều như dự đoán (nhiều train data):

| | coarse 25k | **full 100k** |
|---|---|---|
| val ndcg@10 | .7134 | **.7272** |
| val liked_ndcg@10 | .5567 | **.5641** |
| val liked_r@10 | .2993 | .3048 |
| best_iteration | 1995 | 2949 |

## 6. Kết luận CHỐT (pool `final`, 2026-06-18)

- **Winner production = `lrank_t20_gainLin`** (lambdarank, t20, label_gain=[0,1,2,3,4], α=1.0, K=200). TEST:
  ndcg@10 .5323→**.7231**, liked_ndcg@10 .3903→**.5615**, r@100 .5387→**.6048**, liked_r@100 .6445→**.7182**.
- **Vượt MF ALS đã tune trên mọi metric head+mid**: ndcg@10 .7231 > MF ndcg-opt .7027 (+.020, thoải mái hơn winner v5 +.0047),
  r@100 .6048 > .5954 (v5 còn thua chỗ này), liked_ndcg@10 .5615 ≫ .5052 (+.056). Chỉ nhường deep-recall tail (r@200 trần pool .6758).
- **Đánh đổi đã chấp nhận**: gainLin max ndcg@10 nhưng liked nhỉnh hơn xendcg một chút khi lên full (.5641 vs coarse cho thấy
  căng thẳng thu hẹp ở full data) — chọn gainLin vì headline đồ án ưu tiên ndcg@10, và liked vẫn ≫ MF.
- **Relabel liked-aware / steep: bác** (không giúp liked, hại ndcg@10) → giữ grading mặc định.
- Feature importance đổi đáng chú ý so với v5: `cos_uv` leo lên #2 (label_gain tuyến tính khai thác cosine thô, không chỉ rank) — `docs/RESULTS.md §8`.
- Cold giữ **tách kênh serve** (cosine), α=1 dìm cold → không qua blend. ✅ test_cold final-exam đã chấm 1 lần (2026-06-18): full-catalog ndcg@10 .1397 / r@200 .4710; **honly (chỉ rank giữa anime mới = đúng UX) ndcg@10 .2368 / r@100 .6755 / r@200 .8261 / liked_r@100 .7306** (khớp val_cold, generalize); blend α=1 → ndcg@10 .0000 (xác nhận tách kênh) — `docs/RESULTS.md §7` + `docs/RANKER.md §7`.
