# RANKER — GBDT rerank stage 2 (rebuild 2026-06-11)

> Thay thế hoàn toàn doc cũ (`legacy/docs/RANKER.md` — protocol cũ, số liệu vô giá trị).
> Đọc kèm: `artifacts/CONTRACT.md` (schema file), `docs/DATA_SPLIT.md` (protocol gốc),
> `ranker/CLAUDE.md` (firewall + lệnh chạy).

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
                                   eval_queries_{val,test,val_cold}, eval_seen,
                                   users_history, eval_reference.json}
ranker/src/build_eval.py  ──> data/pools/eval_{val,test,val_cold}{,_users}.parquet   (depth 500)
ranker/src/build_train.py ──> data/datasets/train{,_users}.parquet + build_meta.json (K=200)
        │ (upload Drive: datasets/* + pools/eval_val* + item_vectors.npy)
ranker/train.ipynb (Colab) ──> Drive runs_ranker/<ver>/<run>/{model.*, row.json} + ranker_runs.csv
        │ (tải winner về data/models/<run>/)
ranker/src/eval.py        ──> blend sweep + Pareto select + test report + val_cold
ranker/src/export.py      ──> artifacts/ranker.txt + ranker_meta.json
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

## 6. Eval protocol (= retriever, qua artifacts)

- mask = `seen − query`; metrics mean-per-user recall@K / ndcg@K (binary relevance,
  IDCG = min(R_total, K)); R_total = tổng query kể cả ngoài pool → recall@K trong pool-D
  ≡ full ranking khi K ≤ D.
- Pool lưu depth 500 → ablation K∈{200,500} (`eval.py --k 500`) không cần re-encode;
  serve mặc định K=200 (trần r@200 .6505; K=500 nâng trần lên .8147 đổi lấy latency).
- **Sanity gate** (`eval.py --baseline-only`, tự chạy đầu mọi lần eval): cosine baseline phải
  khớp `artifacts/eval_reference.json` (số test_export đo QUA artifacts, hơi thấp hơn số
  checkpoint trong CONTRACT do row H encode OOV) trong 2e-3. Fail = dừng.
- **Selection (val only)**: Pareto ≥ cosine trên {r@10, r@100, ndcg@10, ndcg@100} + ndcg@10
  strict > → max ndcg@10; fallback max ndcg@10 s.t. r@100 ≥ cosine. (r@200 bỏ — trần pool.)
- val_cold: rerank pool cold của val (debug, được phép). test warm: report sau khi chốt
  winner. test_cold: final exam.

## 7. Trạng thái & số liệu (2026-06-11 — baseline local, CHƯA có run Colab thật)

Cosine baseline (two-stage pool, khớp eval_reference): val ndcg@10 **.5207** / r@10 .1526;
test .5155 / .1516; val_cold ndcg@10 .1572 / r@10 .0767. Pool ceiling r@200: val .6505.

| run (val, α best) | ndcg@10 | r@10 | ghi chú |
|---|---|---|---|
| cosine | .5207 | .1526 | baseline |
| linear (full data, local) | .6154 @ α=.5 | .1773 | bar tối thiểu cho GBDT |
| smoke GBDT (3k user, 50 round) | .6530 @ α=1 | .1905 | placeholder — Colab sweep sẽ thay |
| smoke NN (2k group, 1 epoch) | .6354 @ α=1 | — | placeholder |

⚠ **Cold regress ở α=1**: smoke GBDT val_cold ndcg@10 .1572 → .0239 (model học popularity
mà cold không có + cold bị loại khỏi train pool). Quyết định khi có run thật, các phương án:
(a) chọn α thấp hơn nếu warm chịu được; (b) serve rule "candidate cold giữ điểm cosine"
(bypass blend); (c) chấp nhận (cold đã có kênh riêng từ retriever). Đo rồi quyết, không đoán.

`artifacts/ranker.txt` hiện = smoke placeholder (flow export đã verify). Export thật sau
khi Colab sweep xong.

## 8. ranker_meta.json (schema cho service)

`feature_names` (thứ tự X khi predict), `categorical_features`, `grading`, `k_retrieve`,
`blend_alpha` + công thức `blend`, `cold_feature_policy`, `hist_feat_cap`/`eval_history_cap`,
`feature_importance_gain`, `val/test/val_cold metrics + baselines`, `pool_ceiling`,
`train_provenance` (n_groups/rows/k_pool/seed/git_rev/source_checkpoint), `generated`.

Service: dựng U (user_encode.py) → top-K từ item_vectors → tính 29 feature ĐÚNG thứ tự
(`features.py` + `pool.cross_features`) → booster.predict → blend α → sort.

## 9. Retrain loop (khi best.pt retriever đổi)

```bash
venv/bin/python retriever/export.py && venv/bin/python retriever/test_export.py
venv/bin/python ranker/src/build_eval.py            # ~30s
cd ranker/src && ../../venv/bin/python eval.py --baseline-only   # sanity gate
venv/bin/python ranker/src/build_train.py           # ~1 phút, in list upload Drive
# → Colab ranker/train.ipynb (sweep + leaderboard) → tải winner về data/models/ →
cd ranker/src && ../../venv/bin/python eval.py      # select + test report + val_cold
cd ranker/src && ../../venv/bin/python export.py && cd ../.. \
  && venv/bin/python -m pytest ranker/tests -q
```
