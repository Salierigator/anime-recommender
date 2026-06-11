# RESULTS — sổ kết quả + bản đồ nguồn số liệu

Một chỗ tra MỌI con số của pipeline (retriever / ranker / two-stage / baselines) + chỉ rõ **số nào lấy từ file nào** và **vì sao có 2 biến thể số retriever**. Đây là file tổng hợp — số gốc luôn nằm trong các file máy-sinh được liệt kê ở §1.

> ⚠️ Snapshot **2026-06-11**: retriever `v5_hist64_ep2` (2 epoch) + ranker `xendcg_lr05_l63`. Retriever còn tune trên Colab → mỗi lần best.pt đổi, TOÀN BỘ số ở đây đổi theo (chạy lại loop `docs/RANKER.md §9` rồi cập nhật file này). Trạng thái mới nhất: root `PROGRESS.md`.

---

## 1. Bản đồ nguồn — số nào nằm ở đâu

| Cần số gì | File nguồn (máy sinh) | Ghi chú |
|---|---|---|
| **Retriever full-catalog, serve-path** (warm val/test + cold val) — số CHÍNH thức để báo cáo | `artifacts/eval_reference.json` | do `retriever/test_export.py` đo QUA artifacts (row H→OOV) |
| Retriever theo checkpoint — so run-vs-run khi tune | bảng v5 trong `PROGRESS.md` + leaderboard `runs.csv` (Drive) + `artifacts/CONTRACT.md` (val của best.pt) | checkpoint-path, cao hơn serve-path ~0.5–1đ (xem §2) |
| Baselines retriever (TEST warm + cold) | `retriever/baselines/*.txt` | phương pháp: `docs/BASELINES.md` |
| Ranker per-model: **sweep α** + val + cold diagnostic | `ranker/models/<run>/results.txt` (+ `row.json`: hyperparam, train_sec) | CHỈ VAL — kỷ luật giữ test sạch |
| Ranker leaderboard mọi run Colab | `ranker_runs.csv` (Drive) | ngoài repo |
| **Two-stage CHỐT**: val + test + cold + pool ceiling + feature importance + provenance | `artifacts/ranker_meta.json` (bản ghi lúc chọn: `ranker/models/eval_selection.json`) | test chấm đúng 1 lần sau khi chốt trên val |
| test_cold (cold final) | **CHƯA TỒN TẠI** — final exam | chấm 1 lần lúc chốt toàn pipeline: `retriever/export.py --final-exam` → `ranker/eval.py --final-exam` |

**Trả lời nhanh "kết quả test ở đâu":**
- *Retriever-only (test, full catalog)* = `eval_reference.json::test_warm` — và nó ≡ hàng `baseline_test` (cosine) trong `ranker_meta.json`. Hai số trùng nhau **by construction**: pool two-stage = top-200 cosine, nên recall/ndcg@K trong pool ≡ full ranking khi K ≤ 200 (mọi hit ngoài pool không thể vào top-K; mẫu số R đếm đủ query kể cả ngoài pool). Sanity gate ép trùng trong 2e-3.
- *Ranker / two-stage (test)* = `ranker_meta.json::test_metrics` (ndcg@10 **.7074**).
- *"Retriever + ranker"* không có số thứ ba riêng — two-stage CHÍNH LÀ retriever + ranker end-to-end (cosine top-200 → rerank), tức `test_metrics` ở trên.

## 2. Hai biến thể số retriever — đọc cho đúng

| Biến thể | Cache item encode thế nào | Dùng khi nào | Nguồn |
|---|---|---|---|
| **Checkpoint-path** | mọi item id thật (row H = id thật chưa train, vector noise) | so run-vs-run lúc tune (Colab) | `PROGRESS.md` bảng v5, `runs.csv`, CONTRACT.md |
| **Serve-path** | row H encode **id→OOV** (content thật — đúng serving) | số chính thức / mọi số ranker + service | `eval_reference.json`, mọi số trong `ranker_meta.json` |

Serve-path warm thấp hơn checkpoint-path ~0.5–1 điểm (vd test r@200 .6608 → .6524) vì 1.142 row H từ noise-vector trở thành content-vector "hợp lý" → distractor mạnh hơn. Đây là chủ đích: số phải khớp cái user thật nhìn thấy.

Ngoài ra khi đọc recall@K nhỏ: có **trần lý thuyết** do R > K (warm test trần r@10 ≈ .408, r@200 ≈ .993 — `docs/DATA_SPLIT.md §8`); và two-stage có **trần pool** r@200 = .6505 val / .6524 test (`ranker_meta.json::pool_ceiling`).

## 3. Retriever — two-tower `v5_hist64_ep2` (serve-path, full catalog)

Config thắng (nguồn: PROGRESS + CONTRACT): d=128, MLP [256], use_item_id (id_dim 128), τ=.07, logq_alpha=1, history_source=cache, history_pool=mean, **train_hist_len=64**, id_dropout=.1, bs=8192, 2 epoch (Colab A100). Checkpoint: epoch=1, step=16000.

| Slice | r@10 | r@50 | r@100 | r@200 | r@500 | ndcg@10 | ndcg@100 | n_users |
|---|---|---|---|---|---|---|---|---|
| warm **val** | .1526 | .3842 | .5146 | .6505 | .8147 | .5207 | .4820 | 14,029 |
| warm **test** | .1516 | .3847 | .5160 | .6524 | .8164 | .5155 | .4789 | 14,250 |
| **cold val** (val_cold) | .0767 | .2070 | .2925 | .3881 | .5471 | .1572 | .1926 | 8,388 |

Cold thêm pooled hitrate (150.335 pairs): @10 .0720 · @100 .2964 · @200 .4074 · @500 .5881. test_cold: chưa chấm (final exam).

So run-vs-run v5 (checkpoint-path, WARM TEST, Colab — chỉ để thấy lever nào ăn):

| run | đòn bẩy | r@10 | r@200 | ndcg@10 |
|---|---|---|---|---|
| v5_alpha05_ep2 | logq_alpha=.5 | .1201 | .6053 | .3446 |
| v5_embed_ep2 | history_source=embed | .1531 | .6465 | .4639 |
| v5_itemid128_ep2 | control (hist32) | .1556 | .6526 | .5072 |
| **v5_hist64_ep2** ★ | train_hist_len=64 | .1582 | .6608 | .5135 |

## 4. Baselines retriever (TEST — chi tiết phương pháp: `docs/BASELINES.md`)

| method | r@10 | r@100 | r@200 | r@500 | ndcg@10 |
|---|---|---|---|---|---|
| random | .0005 | .0045 | .0093 | .0230 | .0025 |
| content (mean+IDF) | .0368 | .1577 | .2344 | .3779 | .0945 |
| meta_popular | .0848 | .3198 | .4387 | .6156 | .3362 |
| popular | .0865 | .3321 | .4516 | .6279 | .3527 |
| itemknn (K=200) | .1211 | .4105 | .5722 | .7979 | .4685 |
| **MF ALS-64 fold-in** (bar) | **.1951** | **.5759** | **.6989** | **.8352** | **.6771** |

Cold (test_cold): content r@100 .1320 / r@200 .2177 / hit@500 .3784 · meta_popular r@200 .0999 · random r@200 .0086 · popular/itemknn/mf = **N/A by construction**.

## 5. Ranker — so sánh model (VAL, two-stage pool 200, 14.029 users)

Nguồn: `ranker/models/<run>/{results.txt,row.json}`. Mỗi model lấy α tốt nhất của chính nó:

| model | α best | ndcg@10 | r@10 | r@100 | ndcg@100 | train | ghi chú |
|---|---|---|---|---|---|---|---|
| cosine (retriever-only) | — | .5207 | .1526 | .5146 | .4820 | — | = α=0 |
| linear (logistic) | .5 | .6161 | .1773 | .5485 | .5359 | 10s CPU | 24 numeric z-scored, label binary, 2M rows |
| nn_din (DIN+MLP) | 1.0 | .6923 | .2074 | .5758 | .5838 | 201s GPU | 6.212 steps, 2 epoch |
| **xendcg_lr05_l63** ★ | **1.0** | **.7103** | **.2147** | **.5801** | **.5937** | 6.438s Colab | 1.747 trees |

**Sweep α (val ndcg@10)** — hành vi blend là kết quả đáng phân tích trong đồ án:

| α | 0 | .25 | .4 | .5 | .6 | .75 | 1.0 |
|---|---|---|---|---|---|---|---|
| linear | .5207 | .6120 | .6157 | **.6161** | .6144 | .6087 | .5849 |
| nn_din | .5207 | .6160 | .6361 | .6455 | .6554 | .6668 | **.6923** |
| xendcg | .5207 | .6247 | .6440 | .6536 | .6639 | .6780 | **.7103** |

Đọc: GBDT/NN **đơn điệu tăng theo α** → model đủ tin để override hẳn cosine (khác ranker cũ phải blend α=.5); linear đạt đỉnh ở α=.5 rồi tụt — không đủ mạnh để đứng một mình, cần cosine làm sàn. Selection rule (Pareto vs cosine) + lý do chọn GBDT: `docs/RANKER.md §6-7`.

## 6. Two-stage CHỐT — TEST (chấm 1 lần sau khi chốt trên val)

`xendcg_lr05_l63`, α=1.0, K=200 (nguồn: `ranker_meta.json`):

| TEST (14,250 users) | r@10 | r@50 | r@100 | r@200 | ndcg@10 | ndcg@50 | ndcg@100 | ndcg@200 |
|---|---|---|---|---|---|---|---|---|
| cosine (retriever-only) | .1516 | .3847 | .5160 | .6524 | .5155 | .4545 | .4789 | .5309 |
| **two-stage** | **.2137** | **.4698** | **.5811** | .6524 | **.7074** | **.5892** | **.5917** | .6090 |
| Δ | +.0621 | +.0851 | +.0651 | 0 (trần pool) | **+.1919** | +.1347 | +.1127 | +.0781 |

So bar MF ALS (full catalog, test): ndcg@10 two-stage **.7074 > .6771** MF → two-stage vượt bar ở head precision (so sánh hợp lệ — top-10 của two-stage cũng là top-10 full-catalog, xem §1). Trung thực ở tail: r@200 two-stage .6524 (kẹt trần pool) **< MF .6989**, r@500 two-tower .8164 < MF .8352 — recall tail vẫn là việc của retriever (tune tiếp / nâng K), không phải ranker.

## 7. Cold — kênh serve (quyết định: tách kênh, `docs/RANKER.md §7`)

| val_cold (8.388 users) ndcg@10 | giá trị |
|---|---|
| ① cold qua blend α=1 (diagnostic) | .0008 — bị model dìm |
| ② in-list bypass | .1163, nhưng warm tụt .7103→.6434 |
| ③ **kênh riêng theo cosine (CHỐT)** | **.1572** — zero regress cả 2 phía |

Kênh cosine cold per-K: xem bảng §3 hàng cold val. So sánh cấu trúc: two-tower cold r@200 .3881 ≈ **1.8×** content baseline (.2177, test_cold) trong khi MF/KNN/popular = 0 — đây là claim "gợi ý được anime mới" của đồ án (so khác slice val/test, đọc như order-of-magnitude).

## 8. Feature importance (gain, LightGBM winner — `ranker_meta.json::feature_importance_gain`)

| # | feature | gain | # | feature | gain |
|---|---|---|---|---|---|
| 1 | pool_rank | 123.967 | 7 | rank | 9.215 |
| 2 | hist_cos_max | 79.661 | 8 | log_favorites | 8.774 |
| 3 | log_scored_by | 16.681 | 9 | recency_years | 7.326 |
| 4 | u_n_rated | 14.859 | 10 | episodes | 5.520 |
| 5 | mal_score | 13.896 | 11 | log_members | 5.476 |
| 6 | hist_cos_top5_mean | 10.566 | … | **cos_uv** | **2.433** |

Đọc (3 ý cho đồ án): (a) **pool_rank ≫ cos_uv** — model dùng *thứ hạng* cosine trong pool (chuẩn hoá per-user, cây split dễ) chứ không phải giá trị cosine thô → tín hiệu retriever vẫn là xương sống nhưng ở dạng rank; (b) nhóm **match-với-history** (hist_cos_max/top5) đứng top — rerank chủ yếu tinh chỉnh theo độ giống các anime user đã thích nhất; (c) nhóm **prior chất lượng/phổ biến** (scored_by, mal_score, rank, favorites) bù phần popularity mà cosine thiếu. Categorical (type/source/...) đóng góp thấp — content đã nằm trong V.

## 9. Tái tạo / cập nhật số

```bash
venv/bin/python retriever/export.py && venv/bin/python retriever/test_export.py   # §2-3: eval_reference.json
venv/bin/python retriever/baselines/<name>.py                                     # §4: baselines/*.txt
venv/bin/python ranker/report_models.py                                           # §5: models/<run>/results.txt (chỉ VAL)
venv/bin/python ranker/eval.py                                                    # §6: select + test + cold → eval_selection.json
venv/bin/python ranker/export.py                                                  # ranker_meta.json
# final exam (1 LẦN, lúc chốt toàn pipeline):
venv/bin/python retriever/export.py --final-exam && venv/bin/python ranker/eval.py --final-exam
```
