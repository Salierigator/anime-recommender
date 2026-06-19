# LIKED_METRIC — đo "chất lượng đầu list": liked-recall@k + liked-ndcg@k

> Nguồn viết đồ án cho phần **metric đo chất lượng** (bổ sung cạnh recall/ndcg binary). Hai metric
> **report-only** thêm vào cả retriever lẫn ranker để trả lời: "thứ gợi ra ở đầu list có phải item
> user *thật sự thích* không", chứ không chỉ "item user *sẽ tương tác*". Bổ sung cho
> `TWO_TOWER_MODEL.md` (protocol eval retriever) + `RANKER.md` (two-stage metric). Code:
> `retriever/src/metrics.py`, `ranker/src/metrics.py`, `ranker/eval.py`, `retriever/train.ipynb`.
>
> Trạng thái (2026-06-16): **ĐÃ TRIỂN KHAI + verify local** (tests + sanity gate pass). Là tinh
> chỉnh *đo lường* — KHÔNG đổi label/objective/selection (xem §5). Tài liệu này thay cho `REFINE.md`
> (phân tích gốc) — REFINE đã gập vào đây.

## 0. Bối cảnh — vì sao cần

Phát hiện gốc (chuỗi phân tích trong REFINE §3–§4): ranker **train graded** theo điểm
([`ranker/src/config.py::grade`](../ranker/src/config.py) `10→4,9→3,7-8→2,0/5/6→1`) nhưng **đo +
chọn model hoàn toàn binary** (`eval_pool` labels 0/1 = candidate ∈ query). Hệ quả: con số headline
(test ndcg@10 binary, hiện .7231) chỉ trả lời *"có đẩy một item-đã-tương-tác lên đầu không"*, **chưa bao giờ** đo
*"có đẩy item user thật sự thích (điểm cao theo thang riêng) lên đầu không"*. Hai model — một cái nhồi
đầu list bằng score-5/chưa-chấm, một cái nhồi 9/10 — cho **ndcg@10 binary y hệt**.

Nguyên tắc: **đo trước khi tối ưu**. Hai metric dưới đây *đo* chất lượng đầu list (report-only),
chưa đụng tới objective/selection.

**Quyết định phạm vi** (chốt với user, 2026-06-16):
- Làm **liked-recall@k + liked-ndcg@k** — KHÔNG làm graded-ndcg (gọn hơn: tái dùng đúng công thức
  recall/ndcg sẵn có, chỉ đổi *tập relevant*; không cần grade per-candidate / ideal-gains).
- "liked" = **per-user above-own-mean** (z-score τ=0): xem §1.

## 1. Định nghĩa metric (chuẩn, linh hoạt mọi @k)

Per user, top-k là **ranking ĐÃ CÓ** (không đổi gì so với metric binary). Dùng lại đúng
`discount[i] = 1/log2(i+2)` (i 0-index) và `idcg_cum = cumsum(discount)` của metric hiện tại.

**Liked set** của user u:
```
u_mean(u) = mean{ support_score : support_score ≥ 1 }      # rated mean, từ FULL support
liked(u)  = { query item a : score_a ≥ 1  AND  score_a > u_mean(u) }
R_liked   = |liked(u)|                                      # gồm cả query KHÔNG lọt pool/top-k
```
- **Per-user above own mean** (τ=0): liked ⇔ chấm trên trung bình của *chính user*. score 8 của
  user mean-9 → **không** liked; score 7 của user mean-5 → **liked**.
- τ=0 nên **không cần u_std/σ_floor** (std chỉ scale, không đổi dấu); user chấm đều (u_std=0) hoặc
  không có rated support tự bị loại (không có liked).
- `score=0` (completed không chấm) **luôn non-liked** (không normalize được).
- `u_mean` lấy từ **full support** ở cả 2 stage → retriever ≈ ranker. Leak-clean: support là input
  hợp lệ, query bị giấu khỏi support.

**Hai metric** (mean trên user có `R_liked>0`, kèm `n_users_liked` = coverage):
```
liked_recall@k = (#liked trong top-k) / R_liked
liked_ndcg@k   : binary relevance = liked(u)
                 DCG@k  = Σ_{i<k} [ranked_i ∈ liked] · discount[i]
                 IDCG@k = idcg_cum[min(R_liked, k) − 1]      # liked đứng đầu lý tưởng
                 = DCG@k / IDCG@k
```

**Bất biến quan trọng:** ranking + seen-mask **KHÔNG đổi** (vẫn mask = `seen − TOÀN BỘ query`, không
phải `seen − liked`). Chỉ đổi *tập đếm là hit* + *mẫu số*. Nhờ vậy liked-metric so **trực tiếp** được
với binary trên cùng một ranking. Pin số tính tay: `*/tests/test_metrics.py::test_liked_*`.

## 2. Triển khai (plumbing)

Trục: query item cần mang **score** để xác định liked. Trước đây score bị bỏ rơi (prep có nhưng
không ghi).

### Retriever
| File | Thay đổi |
|---|---|
| [`data_prep/05_history_examples.py`](../retriever/data_prep/05_history_examples.py) | carry `score` vào examples (warm + cold) — biến `pos` đã có sẵn cột score |
| [`src/data.py`](../retriever/src/data.py) | `ExamplesDataset.score` (đọc + slice song song trong subset/user_frac); CHỈ dùng eval, KHÔNG vào train |
| [`src/metrics.py`](../retriever/src/metrics.py) | `group_scores`, `load_query_scores`, `support_mean`; `evaluate(query_scores=…)` cộng liked-metric; `run_cold_eval` truyền cold query scores |
| [`export.py`](../retriever/export.py) | `eval_queries_{split}.parquet` thêm cột `score: int8` (cross-firewall → ranker đọc) + CONTRACT |

### Ranker
| File | Thay đổi |
|---|---|
| [`src/pool.py`](../ranker/src/pool.py) | `load_queries` trả `(queries, query_scores)`; `PoolWriter` ghi cột `label_liked` (per-candidate) + `r_liked` (per-user side-table) — optional, train pool ghi zeros |
| [`data_prep/build_eval.py`](../ranker/data_prep/build_eval.py) | dựng `liked_set` per user từ `u_mean=stats["u_mean_score"]` (gate `u_n_rated>0`) → `label_liked` + `r_liked` |
| [`src/metrics.py`](../ranker/src/metrics.py) | `eval_pool(label_liked=…, r_liked=…)` cộng liked-metric (mirror đúng công thức retriever) |
| [`eval.py`](../ranker/eval.py) | `LIKED_METRICS`, `liked_arrays()`; báo cáo liked ở `baseline_gate`/`report_split`/Δ-vs-cosine; lưu vào `eval_selection.json`. **Selection vẫn binary** (`SEL_METRICS`) — kỷ luật report-only |

### Notebook leaderboard (`retriever/train.ipynb`)
- `eval_run` (cell 5): truyền `query_scores`; **chỉ ghi CORE-K** (cắt mạnh) — `recall@{100,200}`,
  `ndcg@10`, `liked_recall@{100,200}`, `liked_ndcg@10`, `n_users_liked` cho val+test. (Vẫn eval đủ
  `eval_ks`, chỉ không ghi ra → bảng gọn; headline `test_recall@200` giữ.)
- `rebuild_leaderboard(force=False)` (cell 5): `force=True` bỏ cache `row.json`/csv cũ → **eval lại
  từ best.pt** để run CŨ có cột liked mới.
- Cell 9: `rebuild_leaderboard(force=True)` (migrate `runs.csv` 1 lần) + display `_show` gọn.
- Cell 10 (cold): thêm `cold_liked_recall@200`/`cold_liked_ndcg@10`/`cold_n_users_liked`, cắt core-k,
  `FORCE=True` để migrate `cold_runs.csv`. **Cold = DIAGNOSTIC-ONLY** (cold không qua ranker, tín hiệu
  liked thưa — đừng dùng quyết định).

## 3. Cách chạy (re-measure)

```bash
# 1) regenerate examples có score (deterministic, SEED=42; stream ratings.csv ~3.4GB)
venv/bin/python retriever/data_prep/05_history_examples.py

# 2) re-export artifacts (eval_queries_* có score) + regenerate eval_reference.json
venv/bin/python retriever/export.py && venv/bin/python retriever/test_export.py

# 3) rebuild ranker eval pools (pool có label_liked, users có r_liked)
venv/bin/python ranker/data_prep/build_eval.py --splits val test val_cold

# 4) tests (2 suite chạy RIÊNG — import flat đụng tên module)
venv/bin/python -m pytest retriever/tests -q
venv/bin/python -m pytest ranker/tests   -q

# 5) sanity gate ranker (binary tái lập eval_reference + in liked cho cosine baseline)
venv/bin/python ranker/eval.py --baseline-only
# full (model vs cosine, có Δ liked): venv/bin/python ranker/eval.py
```
Colab notebook: chạy cell 9 (`force=True`) + cell 10 (`FORCE=True`) một lần để re-eval mọi best.pt
trên Drive → ghi đè `runs.csv`/`cold_runs.csv` với cột liked; xong set lại `False`.

## 4. Kết quả verify (local, 2026-06-16)

Tests: retriever **31 passed**, ranker **16 passed** (gồm 4 test liked pin-số-tính-tay).
Sanity gate **PASS** — binary tái lập `eval_reference.json` y hệt ⇒ plumbing score **không phá**
protocol.

**Liked-metric của RETRIEVER `final` (cosine baseline thuần, full-catalog harness — `runs.csv`/`cold_runs.csv`
@ step 31500; là đường đo full-catalog, KHÁC pool harness — xem `TWO_TOWER_MODEL.md §10.1`):**

| split | n_users | n_liked | liked_recall@100 | liked_recall@200 | liked_ndcg@10 |
|---|--:|--:|--:|--:|--:|
| val | 14,029 | 12,478 | .6543 | .7799 | .3136 |
| test | 14,250 | 12,638 | .6500 | .7754 | .3145 |
| val_cold *(diagnostic)* | 8,388 | 6,361 | .4074 | .5387 | .0921 |

Quan sát: warm `liked_recall` > `recall` binary (test liked_recall@100 .65 vs recall .5462 — item user
yêu được retriever surface tốt hơn item chỉ-tương-tác); `n_users_liked < n_users` (coverage ~89% warm —
user chấm ít / toàn `score=0` rụng).

**Liked-metric production của two-stage (ranker `lrank_t20_gainLin`, pool harness):** test
`liked_ndcg@10` **.3903 → .5615** (+.1712 vs cosine baseline), `liked_recall@100` .6445 → .7182;
`liked_recall@200` .7690 = .7690 (= trần pool K=200 — rerank không đổi tập top-200). Số đầy đủ + nguồn:
[RESULTS.md §6](RESULTS.md) (`ranker_meta.json`/`eval_selection.json`). (Lưu ý: liked retriever ở bảng trên
đo full-catalog; liked two-stage đo trên pool — so trực tiếp trong CÙNG pool harness là cosine .3903 →
ranker .5615.)

## 5. Decisions & những gì BÁC (recorded)

- **Label/objective/selection KHÔNG đổi**: retriever vẫn binary recall stage (REFINE §2 recorded);
  ranker selection vẫn Pareto trên `SEL_METRICS` binary (REFINE §3 kỷ luật). Liked = **report-only**.
- **BÁC graded-ndcg**: cân nhắc nhưng bỏ — liked-ndcg/recall gọn hơn (không grade per-candidate /
  ideal-gains), trả lời cùng câu hỏi.
- **BÁC global threshold (score≥8/≥9)**: chọn per-user above-own-mean (đúng "thích" theo thang riêng).
- **BÁC z≥τ với τ>0**: chọn τ=0 (above mean) → không cần u_std/σ_floor/shrinkage, robust hơn.
- Cột bảng leaderboard: cắt **core-k** (thủ phạm phình là metric-k); GIỮ cột config (provenance rẻ,
  vary trong sweep) — gọn bằng display `_show`.
- Favorites-as-hard-positive (REFINE §6, serve-only) **CHƯA làm** — ngoài phạm vi liked-metric.

## 6. Trạng thái backport sang các doc khác (cập nhật 2026-06-19)

Liked-metric report-only giờ đã phản ánh khắp docs:

- **`BASELINES.md`** ✅ — liked-metric mọi baseline (harness `_eval.py`) + tune-on-val.
- **`RANKER.md`** ✅ — §6 nêu liked report-only (selection vẫn Pareto binary); `eval_selection.json`
  có key `liked_*`; pool có `label_liked`, users-table có `r_liked`.
- **`TWO_TOWER_MODEL.md`** ✅ — §7.3 nêu liked-metric + param `query_scores`. (`retriever/CLAUDE.md` =
  tóm tắt, không bắt buộc chi tiết liked.)
- **`RESULTS.md`** ✅ — §4 baselines + §6 two-stage đều có cột/số liked production.
- **`TRAIN_DATA.md`** + **`artifacts/CONTRACT.md` (auto-gen)** — `examples/*` + `eval_queries_*` có cột
  `score: int8` (CONTRACT tự đúng sau export; TRAIN_DATA mô tả schema cốt lõi).
- **`PROJECT_STRUCTURE.md`** — firewall contract `eval_queries_*` (chi tiết cột ở `artifacts/CONTRACT.md`).
