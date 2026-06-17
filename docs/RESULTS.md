# RESULTS — sổ kết quả + bản đồ nguồn số liệu

Một chỗ tra MỌI con số của pipeline (retriever / ranker / two-stage / baselines) + chỉ rõ **số nào lấy từ file nào** và **vì sao có 2 biến thể số retriever**. Đây là file tổng hợp — số gốc luôn nằm trong các file máy-sinh được liệt kê ở §1.

> ⚠️ Snapshot **2026-06-11**: retriever `v5_hist64_ep2` (2 epoch) + ranker `xendcg_lr05_l63`. Retriever còn tune trên Colab → mỗi lần best.pt đổi, TOÀN BỘ số ở đây đổi theo (chạy lại loop `docs/RANKER.md §9` rồi cập nhật file này). Trạng thái mới nhất: root `PROGRESS.md`.
>
> 🔴 **CHỐT `final` (no synopsis) — 2026-06-17**: config retriever cuối = **`final`** (`history_source=embed`, `train_hist_len=128`, 10 epoch, d128, τ.07, logQ α=1, **synopsis OFF**), **ưu tiên cold**. Synopsis (`final_syn`) đã test on/off và **bị bác** (warm↑ nhưng cold↓ — `docs/SYNOPSIS_EMB.md`). Số warm/cold của `final`: **§3b** dưới (checkpoint-path).
>
> ⚠️ **PENDING re-export**: `best.pt`/`artifacts/`/`eval_reference.json` hiện vẫn là **`final_syn`** (`CONTRACT.md` step 41000), `ranker.txt` còn ở `v5_hist64_ep2` cũ hơn. MỌI số serve-path/two-stage §3–§8 dựa trên `v5_hist64_ep2` → **STALE**; serve-path chính thức của `final` **chưa đo**. Khi tải best.pt=`final`: re-export → test_export → retrain ranker (`docs/RANKER.md §9`) rồi cập nhật file này.

---

## 1. Bản đồ nguồn — số nào nằm ở đâu

| Cần số gì | File nguồn (máy sinh) | Ghi chú |
|---|---|---|
| **Retriever full-catalog, serve-path** (warm val/test + cold val) — số CHÍNH thức để báo cáo | `artifacts/eval_reference.json` | do `retriever/test_export.py` đo QUA artifacts (row H→OOV) |
| Retriever theo checkpoint — so run-vs-run khi tune | bảng v5 trong `PROGRESS.md` + leaderboard `runs.csv` (Drive) + `artifacts/CONTRACT.md` (val của best.pt) | checkpoint-path, cao hơn serve-path ~0.5–1đ (xem §2) |
| **Thử nghiệm chọn final** (synopsis on/off, subset HP-search, search runs) | leaderboard `runs.csv` + `runs/v5/<run>/config.json` (Drive, provenance đầy đủ) | phương pháp + thiết kế: `docs/EXPERIMENTS.md` |
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

> ⚠️ Số `final` ở **§3b** hiện là **checkpoint-path** (artifacts vẫn là `final_syn`) → serve-path official của `final` chưa tồn tại; đo lại sau re-export. Số serve-path "chính thức" §3/§6/§7 vẫn là `v5_hist64_ep2` (STALE).

Ngoài ra khi đọc recall@K nhỏ: có **trần lý thuyết** do R > K (warm test trần r@10 ≈ .408, r@200 ≈ .993 — `docs/DATA_SPLIT.md §8`); và two-stage có **trần pool** r@200 = .6505 val / .6524 test (`ranker_meta.json::pool_ceiling`).

## 3. Retriever — two-tower `v5_hist64_ep2` (serve-path, full catalog) 🔴 STALE (config cũ; final = §3b)

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

## 3b. Retriever final = `final` (no synopsis) — checkpoint-path + cold ⏳ serve-path chờ re-export

Config CHỐT (2026-06-17): d=128, MLP [256], use_item_id (id_dim 128), τ=.07, logq_alpha=1, **history_source=embed**, history_pool=mean, **train_hist_len=128**, id_dropout=.15, bs=16384, **10 epoch**, **synopsis OFF**. best_step 31500.

> ⚠️ Số dưới là **checkpoint-path** (run-vs-run, nguồn `runs.csv`/`cold_runs.csv` row `final`/`final_syn`). `artifacts/best.pt` hiện vẫn `final_syn` → **serve-path chính thức của `final` CHƯA đo** (sẽ thấp hơn ~0.5–1đ, xem §2). Số serve-path official + two-stage cập nhật sau re-export + retrain ranker.

**Warm (test) — ablation synopsis OFF (`final`) vs ON (`final_syn`):**

| run | recall@100 | recall@200 | ndcg@10 | liked_recall@200 | liked_ndcg@10 |
|---|---|---|---|---|---|
| **`final` (OFF)** ★ | .5462 | .6852 | .4242 | .7754 | .3145 |
| `final_syn` (ON, bị bác) | .5580 | .6949 | .4886 | .7835 | .3603 |

**Cold (val_cold, 8.388 user, H→OOV) — lý do chốt OFF:**

| run | recall@100 | recall@200 | liked_recall@200 | honly_recall@200 | ndcg@10 |
|---|---|---|---|---|---|
| **`final` (OFF)** ★ | .3374 | .4664 | .5387 | .8234 | .1398 |
| `final_syn` (ON, bị bác) | .2546 | .3515 | .3905 | .7576 | .1494 |

synopsis OFF thắng cold rõ rệt (recall@200 **+.115**, liked_recall@200 +.148) trong khi chỉ kém warm chút (recall@200 −.010) — head-precision warm là việc của ranker. Vì cold serve = cosine trực tiếp (tách kênh, §7), cold gain chảy thẳng ra "Anime mới". Ablation đầy đủ + cơ chế: `docs/SYNOPSIS_EMB.md`; loss ablation (logQ/τ/β/m_hardneg): `docs/EXPERIMENTS.md §4`. (test_cold = final exam, **chưa chấm**.)

## 4. Baselines retriever (TEST — chi tiết phương pháp: `docs/BASELINES.md`)

> Snapshot **2026-06-17**: +liked-metric mọi baseline; 3 baseline personalized (MF/itemknn/content) **tune-on-val** (chi tiết `docs/BASELINES.md §3`). Số binary của random/popular/meta_popular không đổi vs 2026-06-11 (sanity gate). Cột `lr@k`/`lndcg@10` = liked (report-only, n_users_liked=12,638 warm).

| method | r@10 | r@100 | r@200 | r@500 | ndcg@10 | lr@100 | lr@200 | lndcg@10 |
|---|---|---|---|---|---|---|---|---|
| random | .0005 | .0044 | .0088 | .0219 | .0022 | .0046 | .0092 | .0011 |
| content (mean+IDF) | .0368 | .1577 | .2344 | .3779 | .0945 | .1580 | .2356 | .0495 |
| meta_popular | .0848 | .3198 | .4387 | .6156 | .3362 | .4085 | .5367 | .2456 |
| popular | .0865 | .3321 | .4516 | .6279 | .3527 | .4309 | .5568 | .2629 |
| itemknn (K=50) | .1177 | .4976 | .6592 | .8231 | .4638 | .5875 | .7403 | .3395 |
| **MF ndcg-opt** (f128/α1) | **.2087** | **.5954** | **.7136** | **.8405** | **.7027** | **.6960** | **.7986** | **.5052** |
| **MF recall-opt** (f128/α10) | **.1982** | **.6223** | **.7511** | **.8797** | **.6374** | **.7213** | **.8328** | **.4442** |

MF báo **per-axis** (config tốt nhất mỗi trục; sweep f∈{64,128}×α∈{1,10,40}, reg.05, iter15, refit FULL). MF gốc cũ f64/α1 (.6989/.6771) bị f128/α1 vượt cả 2 trục → đã loại.

Cold (test_cold): content r@100 .1320 / r@200 .2177 / hit@500 .3784 (lr@200 .2103) · meta_popular r@200 .0999 / hit@500 .1559 (lr@200 .1466) · random r@200 .0086 · popular/itemknn/mf = **N/A by construction**.

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

So bar MF ALS **đã tune** (full catalog, test — `docs/BASELINES.md §5`): ndcg@10 two-stage **.7074 > .7027** MF ndcg-opt (f128/α1) → two-stage vẫn vượt bar head-precision nhưng **biên rất sát** (+.0047; so với MF cũ under-tuned .6771 thì +.0303 — đừng dùng số cũ). Ở tail, MF tune mạnh hơn rõ: r@200 two-stage .6524 (kẹt trần pool) **< MF recall-opt .7511**, r@500 two-tower .8164 < MF .8797 — recall tail là việc của retriever (tune tiếp / nâng K). ⚠️ **Hệ quả cho đồ án**: warm-only, two-stage chỉ *nhỉnh* MF ở ndcg@10 và *thua* tail → lợi thế thuyết phục nhất của kiến trúc 2-stage là **cold-start** (§7, "Anime mới"): MF/KNN = 0 by construction còn two-tower r@200 cold .3881 ≈ 1.8–2.2× content. Nên neo narrative vào cold + khả năng kết hợp recall(retriever)+precision(ranker)+cold, không chỉ "thắng MF warm".

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
