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

| run | objective / lever | ndcg@10 | ndcg@100 | **liked_ndcg@10** | liked_r@100 | r@10 | iter | sec |
|---|---|---|---|---|---|---|---|---|
| **lrank_t20_gainLin** | lambdarank t20, label_gain=[0,1,2,3,4] | **.7134** | .6078 | .5567 | .7196 | .2116 | 1995 | 530 |
| lrank_t20_gainExp | lambdarank t20, gain=[0,3,7,15,31] | .6996 | .6013 | .5608 | .7217 | .2084 | 2609 | 661 |
| **xendcg_lr03** | rank_xendcg, lr=0.03 | .6982 | .6012 | **.5632** | .7201 | .2102 | 3995 | 488 |
| xendcg_es10 | rank_xendcg, early-stop ndcg@10 only | .6981 | .6012 | .5627 | .7195 | .2095 | 2964 | 362 |
| xendcg_l127 | rank_xendcg, num_leaves=127 | .6963 | .6000 | .5614 | .7188 | .2095 | 1464 | 217 |
| xendcg | rank_xendcg (winner cũ) | .6956 | .6005 | .5609 | .7200 | .2089 | 1983 | 243 |
| lrank_t50 | lambdarank truncation 50 | .6955 | .6007 | .5593 | .7218 | .2082 | 1570 | 557 |
| xendcg_ff09 | rank_xendcg, feature/bagging .9 | .6955 | .6004 | .5610 | .7195 | .2092 | 1917 | 264 |
| lrank_t30 | lambdarank truncation 30 | .6950 | .5999 | .5585 | .7221 | .2078 | 1621 | 504 |
| lrank_t20_l127 | lambdarank t20, num_leaves=127 | .6948 | .5985 | .5604 | .7213 | .2075 | 1484 | 440 |
| lrank_t20 | lambdarank truncation 20 | .6944 | .5989 | .5593 | .7216 | .2070 | 2072 | 531 |
| xendcg_l255_mdl50 | rank_xendcg, leaves=255, min_data=50 | .6940 | .5984 | .5598 | .7188 | .2088 | 546 | 118 |
| xendcg_lr10 | rank_xendcg, lr=0.1 | .6914 | .5970 | .5580 | .7166 | .2073 | 685 | 108 |
| lrank_t10_es10 | lambdarank t10, early-stop ndcg@10 | .6879 | .5951 | .5549 | .7204 | .2051 | 1362 | 294 |
| lrank_t10 | lambdarank truncation 10 | .6874 | .5953 | .5546 | .7210 | .2047 | 1295 | 282 |
| lrank_t20_gainTop | lambdarank t20, gain=[0,1,3,7,31] | .6803 | .5926 | .5544 | .7216 | .2033 | 1820 | 471 |

**Đọc (coarse — confirm full ở §5):**
- 🥇 **ndcg@10**: `lrank_t20_gainLin` (label_gain **tuyến tính** [0,1,2,3,4]) dẫn rõ rệt (.7134, +.018 vs xendcg).
  Cơ chế: eval ndcg@10 **binary** (bất kỳ query item) — gain tuyến tính bớt dồn vào grade-4 (điểm 10) nên model
  xếp *mọi* positive lên đầu tốt hơn. Ngược lại `gainTop` ([0,1,3,7,31], dồn grade-4) **kém nhất** (.6803).
- 🥇 **liked_ndcg@10**: `xendcg_lr03` (lr thấp, ~4000 cây) cao nhất (.5632) + ndcg@10 hạng 3 (.6982) → **all-rounder
  tốt nhất**. Kế đến `xendcg_es10` (.5627). Nhóm xendcg (.561–.563) liked đều **cao hơn** nhóm lambdarank gainLin (.5567).
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

## 5. Việc còn lại để CHỐT (confirm full + chọn) — phiên sau

`train_lgbm.py` có **resume-skip** (bỏ qua config có `models/<run>/row.json`); data (`train.parquet` có
`target_score`, `pools/`) đã build → không build lại (trừ khi artifacts đổi). Confirm **2 ứng viên** trên FULL 100k:
- `lrank_t20_gainLin` (max ndcg@10) và `xendcg_lr03` (max liked + ndcg@10 mạnh).
```bash
# xoá row.json coarse của 2 config (resume-skip đang giữ chỗ), rồi train FULL (bỏ --subset):
rm -rf ranker/models/lrank_t20_gainLin ranker/models/xendcg_lr03
# tạm sửa SWEEP còn 2 dòng này (hoặc thêm flag chọn) rồi:
venv/bin/python -u ranker/src/train_lgbm.py
venv/bin/python ranker/eval.py        # Pareto select + test report + val_cold (CHỐT)
venv/bin/python ranker/export.py && venv/bin/python -m pytest ranker/tests -q   # → artifacts/ranker.txt + meta
```
⚠️ Coarse train trên 25k → số full sẽ **cao hơn** (nhiều train data); xếp hạng config kỳ vọng giữ nhưng **biên
gainLin vs xendcg_lr03 có thể đổi** — phải confirm full mới chốt.

## 6. Kết luận sơ bộ (chờ confirm full)

- Ranker trên pool `final` **hoạt động tốt**: ndcg@10 .5343→~.71, liked_ndcg@10 .3894→~.56 (vs cosine), α=1.
- **Hai ứng viên chốt** tuỳ trọng số ndcg@10 vs liked:
  - **`lrank_t20_gainLin`** — ndcg@10 cao nhất (.7134) nhưng liked chỉ mid (.5567, *thấp hơn* cả xendcg gốc).
  - **`xendcg_lr03`** — liked cao nhất (.5632) + ndcg@10 .6982; **cải thiện CẢ HAI** so với winner cũ xendcg
    (.6956/.5609). Hợp tiêu chí "ưu tiên ndcg + liked" hơn nếu không muốn hi sinh liked.
- **Relabel liked-aware / steep: bác** (không giúp liked, hại ndcg@10) → giữ grading mặc định.
- Theo selection rule (ndcg@10 chính + liked tie-break): gainLin thắng ndcg@10 rõ → ứng viên #1; nhưng nếu coi
  "không được tụt liked" là ràng buộc thì `xendcg_lr03` an toàn hơn. **Quyết định cuối sau confirm full.**
