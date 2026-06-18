# RANKER — GBDT rerank stage 2 (CHỐT 2026-06-18: lrank_t20_gainLin α=1 trên pool `final`, cold tách kênh)

> Thay thế hoàn toàn doc cũ (`legacy/docs/RANKER.md` — protocol cũ, số liệu vô giá trị).
> Đọc kèm: `artifacts/CONTRACT.md` (schema file), `docs/DATA_SPLIT.md` (protocol gốc),
> `ranker/CLAUDE.md` (firewall + lệnh chạy).

> ✅ **CHỐT 2026-06-18 trên pool `final`**: retriever `final` (no synopsis) → ranker **`lrank_t20_gainLin`** (lambdarank t20, label_gain=[0,1,2,3,4], α=1, K=200, full 100k, 2.949 trees). Số §7 đã khớp `final` (test ndcg@10 .7231 / liked_ndcg@10 .5615). Tổng hợp + bản đồ nguồn từng con số: `docs/RESULTS.md`; quá trình sweep 20 config: `docs/RANKER_EXPERIMENTS.md`. Nếu best.pt retriever đổi lại → chạy loop §9 (build_eval → gate → build_train → train → eval → export). Protocol/kiến trúc ổn định.

## 1. Vai trò & kiến trúc

Two-stage: retriever (two-tower) trả top-K cosine → **ranker rerank** để sửa head precision
(bar: MF ALS ndcg@10 .677 vs two-tower .51 — gap nằm ở head, đúng việc của ranker).

- **Model chính: LightGBM** (lambdarank-family, categorical native, group = 1 user) — train
  vài phút, model text nhỏ, retrain rẻ sau mỗi lần retriever export lại (best.pt còn đổi).
- **So sánh**: linear logistic (baseline — GBDT phải thắng nó mới đáng), neural ranker
  (DIN-attention + MLP, GPU Colab) — kiểm chứng GBDT có phải lựa chọn đúng.
- **Blend bắt buộc** (bài học từ ranker cũ vẫn đúng): điểm cuối =
  `(1−α)·rank_norm(cos_uv) + α·rank_norm(pred)` per pool — cosine làm sàn chống regress tail.
  α sweep trên val, serve áp Y HỆT (đọc `ranker_meta.json`).

## 2. Khác gì ranker cũ (vì sao rebuild)

| Vấn đề cũ | Fix |
|---|---|
| Eval protocol tự chế (3k user sample, tự split 80/20, mask thiếu) — số không so được với retriever | Dùng ĐÚNG queries/seen/history retriever export vào `artifacts/` (eval_queries_* + eval_seen + users_history); chấm toàn bộ ~14k eval user; **sanity gate** ép cosine baseline tái lập `eval_reference.json` trước khi train |
| Candidate train = target + 40 hard-neg + 40 random ≠ phân phối serve | Candidate = **top-200 pool thật** của retriever (cùng 1 code path `pool.py` cho train/eval/serve), không inject, không random-neg |
| Stream ratings.csv + tự dựng label → drift với prep retriever | Mọi thứ từ `users_history.parquet` (retriever export) — không đụng ratings.csv |
| Early stopping chưa kích hoạt (cap 1000 round) | `num_boost_round=4000` + `early_stopping(100)` |
| Không có feature importance, không leaderboard | row.json per run + `ranker_runs.csv` (Drive) + importance_gain trong meta |
| History feature cap 30 | Encode U cap 1024 (khớp retriever); feature sims/affinity cap 256 |

## 3. Data flow

```
retriever/export.py ──> artifacts/{item_vectors, user_tower, user_split,
                                   eval_queries_{val,test,val_cold}, eval_seen, users_history}
retriever/test_export.py ──> artifacts/eval_reference.json (cosine baseline đo QUA artifacts)
ranker/data_prep/build_eval.py  ──> train-data/pools/eval_{val,test,val_cold}{,_users}.parquet (depth 500)
ranker/data_prep/build_train.py ──> train-data/datasets/train{,_users}.parquet + build_meta.json (K=200)
        │ (+ cột target_score: raw score/cand → relabel sweep train_lgbm KHÔNG cần rebuild)
ranker/src/train_lgbm.py (LOCAL) ──> models/<run>/{model.txt, row.json} + models/leaderboard.csv (sweep grid)
        │ (--relabel steep|liked: grade lại label; row.json log thêm liked_ndcg@10 + liked_recall@100)
ranker/src/train_nn.py (LOCAL, MPS/CPU) ──> models/nn_*/{model.pt, row.json}  (comparator — optional)
ranker/eval.py            ──> blend sweep + Pareto select + test report + val_cold
                              (+ ghi models/eval_selection.json — bản ghi đầy đủ lúc chọn)
ranker/report_models.py    > models/<run>/results.txt (per-model: sweep α + val per-K + cold
                              diagnostic, kiểu baselines/*.txt — CHỈ VAL, giữ kỷ luật test)
ranker/export.py          ──> artifacts/ranker.txt + ranker_meta.json
```

**Discipline**: `eval_test` pool KHÔNG upload Drive (test ngoài tuning loop by construction).
`test_cold` = final exam — file queries chỉ tồn tại khi `retriever/export.py --final-exam`,
chấm qua `eval.py --final-exam` đúng 1 lần lúc chốt pipeline.

## 4. Train data

- User: 100k sample từ TRAIN split (`n_pos ≥ 5`, seed 42, RNG per-user → deterministic).
- Per user: positives (history full) split **support 80% / target 20%** (tie random — cùng cơ
  chế support/query của eval). U encode từ support (cap 1024).
- Pool = top-200 cosine, mask = support ∪ hard_neg ∪ **cold H** (train user không bao giờ có
  positive H → giữ cold trong pool = dạy model dìm cold) ∪ PAD/OOV. Target KHÔNG mask
  (mirror seen−query).
- **Label graded**: 10→4, 9→3, 7-8→2, 0/5/6→1, ngoài target→0. Drop group 0 positive
  (thực tế chỉ ~0.6% vì median query/user ~7 và pool@200 bắt ~66%).
- Kết quả build hiện tại: 99.4k groups × 200 = 19.9M rows, pos_rate .140,
  grade hist {1: 810k, 2: 1.19M, 3: 470k, 4: 321k}.
- Valid early-stopping = `pools/eval_val.parquet` slice 200 (label binary) — không build riêng.
- Noise chấp nhận (documented): train user không có eval_seen → PTW/on_hold có thể thành
  false-negative trong pool train. Eval không ảnh hưởng (dùng eval_seen thật).

## 5. Features (29 — thứ tự cố định `features.py::FEATURE_NAMES`)

| Nhóm | Features |
|---|---|
| Cross | cos_uv, **pool_rank**, hist_cos_max, hist_cos_mean, **hist_cos_top5_mean**, genre_aff, theme_aff, genre_overlap, **score_gap** (mal_score − u_mean_score) |
| User | u_n_rated, u_mean_score, u_std_score, u_account_age (profiles.csv), **support_len** |
| Item numeric | mal_score(+missing), log_scored_by/members/favorites, popularity, rank(+missing), episodes, recency_years |
| Item categorical (native) | type_code, source_code, rating_code, demo_code, era_code |

(**đậm** = mới so với 25 cũ.) NaN (u_account_age thiếu profile): LightGBM native;
NN/linear impute nanmean khi z-score.

**Cold policy (no-leak, áp trong `ItemFeatures.load` → mọi đường nhất quán)**: row `is_cold`
bị ép mal_score/scored_by/members/favorites/popularity/rank → impute-as-missing + flag
(anime mới lúc serve chưa có stats trưởng thành). Content/episodes/recency giữ (metadata công
bố từ đầu). Lưu ý: số val_cold vì thế KHÔNG so được với ranker cũ (cũ leak stats tương lai).

**Importance thực đo (gain, winner)**: pool_rank 124k ≫ hist_cos_max 80k ≫ log_scored_by 17k
> u_n_rated 15k > mal_score 14k > hist_cos_top5_mean 11k; **cos_uv thô chỉ 2.4k** — model dùng
*thứ hạng* cosine (pool_rank, chuẩn hoá per-user) thay giá trị thô. 2 feature mới đứng top
(pool_rank #1, hist_cos_top5_mean #6); support_len/score_gap đóng góp vừa (4.3k/2.1k). Bảng
đầy đủ + cách đọc: `docs/RESULTS.md §8`; số gốc: `ranker_meta.json::feature_importance_gain`.

## 6. Eval protocol (= retriever, qua artifacts)

- mask = `seen − query`; metrics mean-per-user recall@K / ndcg@K (binary relevance,
  IDCG = min(R_total, K)); R_total = tổng query kể cả ngoài pool → recall@K trong pool-D
  ≡ full ranking khi K ≤ D.
- Pool lưu depth 500 → ablation K∈{200,500} (`eval.py --k 500`) không cần re-encode;
  serve mặc định K=200 (trần r@200 trong pool: val .6505 / test .6524 — `ranker_meta.json::pool_ceiling`;
  K=500 nâng trần val lên .8147 đổi lấy latency).
- **Sanity gate** (`eval.py --baseline-only`, tự chạy đầu mọi lần eval): cosine baseline phải
  khớp `artifacts/eval_reference.json` (số test_export đo QUA artifacts, hơi thấp hơn số
  checkpoint trong CONTRACT do row H encode OOV) trong 2e-3. Fail = dừng.
- **Selection (val only)**: Pareto ≥ cosine trên {r@10, r@100, ndcg@10, ndcg@100} + ndcg@10
  strict > → max ndcg@10; fallback max ndcg@10 s.t. r@100 ≥ cosine. (r@200 bỏ — trần pool.)
- val_cold: rerank pool cold của val (debug, được phép). test warm: report sau khi chốt
  winner. test_cold: final exam.

## 7. Kết quả CHỐT (2026-06-18 — train full 100k LOCAL, pool `final`)

Production: **`lrank_t20_gainLin` (lambdarank, `lambdarank_truncation_level=20`, `label_gain=[0,1,2,3,4]`, 63 leaves), α=1.0, K=200**,
2.949 trees — Pareto-dominate cosine cả 4 metric selection. Đã export `artifacts/ranker.txt` + meta.
(Chọn từ sweep 20 config trên pool `final` — `docs/RANKER_EXPERIMENTS.md`; ưu tiên ndcg@10.)

**Test (chấm sau khi chốt trên val, pool `final`):**

| | ndcg@10 | r@10 | r@100 | ndcg@100 | liked_ndcg@10 | liked_r@100 |
|---|---|---|---|---|---|---|
| cosine (retriever-only) | .5323 | .1681 | .5387 | .5128 | .3903 | .6445 |
| **two-stage ★** | **.7231** | **.2178** | **.6048** | **.6126** | **.5615** | **.7182** |

val tương ứng: ndcg@10 .5343→**.7272**, liked_ndcg@10 .3894→**.5641**, r@100 .5388→.6042.
→ Two-stage **vượt MF ALS đã tune trên mọi metric head+mid** (ndcg@10 .7231 > MF ndcg-opt .7027; r@100 .6048 > .5954;
liked_ndcg@10 .5615 ≫ .5052); chỉ nhường deep-recall tail (r@200 kẹt trần pool .6758 < MF .7136/.7511) — việc của retriever.

**Model-class** (val, đo trên pool v5 cũ — kết luận GBDT-vs-NN-vs-linear không phụ thuộc pool): GBDT > NN_DIN .6923 > linear .6161 >
cosine — GBDT là lựa chọn đúng, NN không đáng phức tạp hoá serving. α=1 thắng tuyệt đối trên warm: candidate pool-matched +
label graded cho model đủ tin override hẳn cosine. Sweep α per-model + bảng gộp: `docs/RESULTS.md §5`.

**Cấu hình từng model** (code: `src/train_lgbm.py`, `src/train_nn.py`, `baselines/baseline_linear.py`):
- **GBDT (winner)**: LightGBM `rank_xendcg`, lr .05, 63 leaves, min_data_in_leaf 100,
  feature/bagging_fraction .8, num_boost_round 4000 + early_stopping(100) trên ndcg nội bộ
  của `pools/eval_val` (CHỈ để early-stop — số chính thức luôn là two-stage `metrics.py`).
  Best iteration 1747, train ~6.4k giây Colab. Sweep Colab các trục objective
  (lambdarank+truncation / xendcg) × lr × leaves — leaderboard `ranker_runs.csv` (Drive),
  local chỉ giữ winner.
- **NN DIN (comparator GPU)**: per candidate concat [V, U, U⊙V, 24 numeric z-scored (NaN→mean),
  5 cat emb dim 4, DIN-attention trên `hist_top64` (query = V, scaled dot)] → MLP 512→256→1;
  loss listwise softmax-CE trọng số gain `2^grade−1`; AdamW lr 1e-3, batch 32 group, 2 epoch,
  early-stop theo two-stage val ndcg@10 @ best α (eval mỗi 400 step). 6.212 steps / ~200s GPU.
- **Linear (sàn)**: logistic regression trên 24 numeric z-scored (bỏ 5 categorical), label
  binary `grade > 0`, sample 2M row, ~10s CPU. GBDT phải thắng nó mới chứng minh được giá trị
  của cây + categorical + tương tác.

**Cold — quyết định: TÁCH KÊNH PHỤC VỤ** (mode `separate_channel_cosine` trong
`ranker_meta.json::cold_serving`). 3 phương án đều ĐO trên val_cold trước khi quyết:
- ① *(loại)* Cold đi qua blend α=1 cùng warm: val_cold ndcg@10 .1572 → **.0008** —
  model học suppress item thiếu stats (đúng logic warm vì cold không bao giờ là đáp án
  warm) → giết kênh cold.
- ② *(loại)* In-list bypass (item cold giữ rank_norm(cos), warm theo pred): cold cứu về
  .1163 nhưng warm tụt (cold chiếm 12.5% pool warm) — thoả hiệp cả hai phía. (Ablation đo trên pool v5;
  quyết định thiết kế giữ nguyên cho `final` — không re-run vì cơ chế α=1-dìm-cold không đổi.)
- ③ **(CHỐT) Tách kênh**: main list = rerank α=1 trên candidate **warm-only** (lọc
  `is_cold` trước rerank → warm giữ nguyên .7272 val); item cold phục vụ **section riêng**
  xếp theo cosine retriever (`final` val_cold ndcg@10 .1398 — lợi thế cấu trúc two-tower). Zero regress
  cả hai phía; mỗi stage làm đúng việc của nó; service hiển thị 2 khối ("Gợi ý cho bạn"
  + "Anime mới cho bạn").

**Held-out xác nhận — test_cold final exam (chấm ĐÚNG 1 lần, 2026-06-18, K=200 α=1):** kênh
phục vụ cold = cosine retriever đo trên test_cold: r@100 **.3414** / r@200 **.4710** / ndcg@10
**.1397** (8.510 user) — gần như khớp val_cold (.3373 / .4664 / .1398) → **generalize, không
overfit**. Ngược lại nếu ép cold qua blend α=1 (chỉ để kiểm chứng cơ chế): ndcg@10 → **.0000**,
r@100 → **.0099** — lặp lại đúng hiện tượng val_cold (α=1 dìm cold tận đáy), khẳng định quyết
định ③. ⇒ chất lượng cold thực tế user thấy = cosine .1397. Số đầy đủ: `eval_selection.json::{baseline_test_cold, test_cold_metrics}`.

## 8. ranker_meta.json (schema cho service)

`feature_names` (thứ tự X khi predict), `categorical_features`, `grading`, `k_retrieve`,
`blend_alpha` + công thức `blend`, `cold_feature_policy`, `hist_feat_cap`/`eval_history_cap`,
`feature_importance_gain`, `val/test/val_cold metrics + baselines`, `pool_ceiling`,
`train_provenance` (n_groups/rows/k_pool/seed/git_rev/source_checkpoint), `generated`.

Service (khớp `cold_serving` trong meta): dựng U (user_encode.py) → top-K cosine từ
item_vectors (mask seen) → **tách warm/cold theo `is_cold`** → warm: tính 29 feature ĐÚNG
thứ tự (`features.py` + `pool.cross_features`) → booster.predict → blend α (α=1 → sort
thẳng theo pred) → main list; cold: section riêng sort theo cosine. Tham chiếu:
`service/backend/recommend.py`.

## 9. Retrain loop (khi best.pt retriever đổi)

```bash
venv/bin/python retriever/export.py && venv/bin/python retriever/test_export.py
venv/bin/python ranker/data_prep/build_eval.py      # ~30s
venv/bin/python ranker/eval.py --baseline-only      # sanity gate
venv/bin/python ranker/data_prep/build_train.py     # ~1 phút, in list upload Drive (cho NN)
venv/bin/python ranker/src/train_lgbm.py            # LightGBM sweep grid LOCAL → models/<run>/ + leaderboard.csv
venv/bin/python ranker/src/train_lgbm.py --relabel liked   # (tuỳ chọn) grade liked-aware — xem docs/RANKER_EXPERIMENTS.md
# (NN DIN comparator: ranker/src/train_nn.py LOCAL trên MPS/CPU — optional, không chốt qua NN)
venv/bin/python ranker/eval.py                      # select + test report + val_cold
venv/bin/python ranker/export.py \
  && venv/bin/python -m pytest ranker/tests -q
```
