# RANKER — GBDT rerank stage 2 (rebuild + CHỐT 2026-06-11: xendcg α=1, cold tách kênh)

> Thay thế hoàn toàn doc cũ (`legacy/docs/RANKER.md` — protocol cũ, số liệu vô giá trị).
> Đọc kèm: `artifacts/CONTRACT.md` (schema file), `docs/DATA_SPLIT.md` (protocol gốc),
> `ranker/CLAUDE.md` (firewall + lệnh chạy).

> ⚠️ Số liệu = snapshot **2026-06-11** (retriever `v5_hist64_ep2` → ranker `xendcg_lr05_l63`). Retriever còn tune → mỗi lần best.pt đổi phải chạy lại loop §9, số sẽ đổi theo; số mới nhất: root `PROGRESS.md`. Tổng hợp mọi kết quả + bản đồ nguồn từng con số: `docs/RESULTS.md`. Protocol/kiến trúc/quyết định thiết kế thì ổn định.

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
        │ (upload Drive cho NN: datasets/* + pools/eval_val* + item_vectors.npy)
ranker/src/train_lgbm.py (LOCAL) ──> models/<run>/{model.txt, row.json} + models/leaderboard.csv (sweep grid)
ranker/train.ipynb (Colab, NN DIN) ──> Drive runs_ranker/<ver>/nn_*/{model.pt, row.json}
        │ (tải winner NN về models/<run>/ nếu chốt NN)
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

## 7. Kết quả CHỐT (2026-06-11 — Colab runs, full 100k user)

Production: **`xendcg_lr05_l63` (rank_xendcg, lr .05, 63 leaves), α=1.0, K=200** —
Pareto-dominate cosine cả 4 metric selection. Đã export `artifacts/ranker.txt` + meta.

**Val (two-stage pool 200, 14,029 users) — α best mỗi model:**

| run | ndcg@10 | r@10 | r@100 | ndcg@100 |
|---|---|---|---|---|
| cosine (baseline) | .5207 | .1526 | .5146 | .4820 |
| linear (local) | .6161 @ α=.5 | .1773 | .5485 | .5359 |
| nn_din (Colab GPU) | .6923 @ α=1 | .2074 | .5758 | .5838 |
| **xendcg_lr05_l63 ★** | **.7103 @ α=1** | **.2147** | **.5801** | **.5937** |

**Test (chấm 1 lần sau khi chốt trên val):** ndcg@10 .5155 → **.7074** (+.1919),
r@10 .1516 → **.2137** (+.0621), r@100 .5160 → .5811, ndcg@100 .4789 → .5917.
→ Two-stage giờ **vượt bar MF ALS** (ndcg@10 .6771) ở head precision; r@200 vẫn kẹt trần
pool .6524 (test) — việc của retriever, không phải ranker.

Đọc bảng: GBDT > NN (−.018) > linear (−.094) > cosine — GBDT là lựa chọn đúng, NN không
đáng phức tạp hoá serving. α=1 thắng tuyệt đối trên warm (khác ranker cũ cần α=.5):
candidate pool-matched + label graded cho model đủ tin để override hẳn cosine. Sweep α
per-model (GBDT/NN đơn điệu tăng theo α; linear đỉnh α=.5 rồi tụt — cần cosine làm sàn):
`ranker/models/<run>/results.txt`, bảng gộp `docs/RESULTS.md §5`.

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
  .1163 nhưng warm tụt .7103 → .6434 (cold chiếm 12.5% pool warm) — thoả hiệp cả hai phía.
- ③ **(CHỐT) Tách kênh**: main list = rerank α=1 trên candidate **warm-only** (lọc
  `is_cold` trước rerank → warm giữ nguyên .7103); item cold phục vụ **section riêng**
  xếp theo cosine retriever (giữ nguyên .1572 — lợi thế cấu trúc two-tower). Zero regress
  cả hai phía; mỗi stage làm đúng việc của nó; service hiển thị 2 khối ("Gợi ý cho bạn"
  + "Anime mới cho bạn").

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
# (NN DIN comparator vẫn trên Colab ranker/train.ipynb — tải winner NN về models/<run>/ nếu cần)
venv/bin/python ranker/eval.py                      # select + test report + val_cold
venv/bin/python ranker/export.py \
  && venv/bin/python -m pytest ranker/tests -q
```
