# RESULTS — sổ kết quả + bản đồ nguồn số liệu

Một chỗ tra MỌI con số của pipeline (retriever / ranker / two-stage / baselines) + chỉ rõ **số nào lấy từ file nào** và **vì sao có 2 biến thể số retriever**. Đây là file tổng hợp — số gốc luôn nằm trong các file máy-sinh được liệt kê ở §1.

> ⚠️ Snapshot **2026-06-11**: retriever `v5_hist64_ep2` (2 epoch) + ranker `xendcg_lr05_l63`. Retriever còn tune trên Colab → mỗi lần best.pt đổi, TOÀN BỘ số ở đây đổi theo (chạy lại loop `docs/RANKER.md §9` rồi cập nhật file này). Trạng thái mới nhất: root `PROGRESS.md`.
>
> 🔴 **CHỐT `final` (no synopsis) — 2026-06-17**: config retriever cuối = **`final`** (`history_source=embed`, `train_hist_len=128`, 10 epoch, d128, τ.07, logQ α=1, **synopsis OFF**), **ưu tiên cold**. Synopsis (`final_syn`) đã test on/off và **bị bác** (warm↑ nhưng cold↓ — `docs/SYNOPSIS_EMB.md`). Số warm/cold của `final`: **§3b** dưới (checkpoint-path).
>
> ✅ **Re-export DONE (2026-06-17)**: `best.pt`/`artifacts/`/`eval_reference.json` giờ là **`final`** (`CONTRACT.md` epoch=7 step=31500, synopsis OFF). **Serve-path official của `final` đã đo** → §3b (bảng serve-path).
>
> ✅ **RANKER RETRAIN DONE — CHỐT 2026-06-18 (pool `final`)**: `ranker.txt` = **`lrank_t20_gainLin`** (lambdarank t20, label_gain=[0,1,2,3,4], α=1.0, K=200, 2.949 trees, train full local). Số two-stage **§6** đã khớp `final`: test ndcg@10 **.7231** / liked_ndcg@10 **.5615** — **vượt MF ndcg-opt trên mọi metric head+mid**, chỉ nhường deep-recall tail (trần pool). §5 = so model-class trên pool v5 cũ (giữ làm bằng chứng GBDT); §3 retriever = v5 STALE (final ở §3b).

---

## 1. Bản đồ nguồn — số nào nằm ở đâu

| Cần số gì | File nguồn (máy sinh) | Ghi chú |
|---|---|---|
| **Retriever full-catalog, serve-path** (warm val/test + cold val) — số CHÍNH thức để báo cáo | `artifacts/eval_reference.json` | do `retriever/test_export.py` đo QUA artifacts (row H→OOV) |
| Retriever theo checkpoint — so run-vs-run khi tune | bảng v5 trong `PROGRESS.md` + leaderboard `runs.csv`/`cold_runs.csv` (Drive; bản local: `retriever/runs/`) + `artifacts/CONTRACT.md` (val của best.pt) | checkpoint-path, cao hơn serve-path ~0.5–1đ (xem §2) |
| **Thử nghiệm chọn final** (synopsis on/off, subset HP-search, search runs) | leaderboard `runs.csv`/`cold_runs.csv` (bản local `retriever/runs/`) + `runs/v5/<run>/config.json` (Drive, provenance đầy đủ) | phương pháp + thiết kế: `docs/EXPERIMENTS.md` |
| Baselines retriever (TEST warm + cold) | `retriever/baselines/*.txt` | phương pháp: `docs/BASELINES.md` |
| Ranker per-model: **sweep α** + val + cold diagnostic | `ranker/models/<run>/results.txt` (+ `row.json`: hyperparam, train_sec) | CHỈ VAL — kỷ luật giữ test sạch |
| Ranker leaderboard mọi run Colab | `ranker_runs.csv` (Drive) | ngoài repo |
| **Two-stage CHỐT**: val + test + cold + pool ceiling + feature importance + provenance | `artifacts/ranker_meta.json` (bản ghi lúc chọn: `ranker/models/eval_selection.json`) | test chấm đúng 1 lần sau khi chốt trên val |
| test_cold (cold final) | ✅ **đã chấm 1 lần 2026-06-18** → `ranker/models/eval_selection.json::{baseline_test_cold, test_cold_metrics}` (số: §7) | held-out, chấm qua `retriever/export.py --final-exam` → `ranker/eval.py --final-exam`; file queries đã xoá lại để giữ kỷ luật |

**Trả lời nhanh "kết quả test ở đâu":**
- *Retriever-only (test, full catalog)* = `eval_reference.json::test_warm` (hiện = `final`: r@100 .5387, r@200 .6758, ndcg@10 .5323). Nó ≡ hàng `baseline_test` (cosine) trong `ranker_meta.json` **by construction** (pool two-stage = top-200 cosine, recall/ndcg@K trong pool ≡ full ranking khi K ≤ 200; sanity gate ép trùng trong 2e-3). ✅ **Đã trùng** sau khi retrain ranker trên `final` (cosine test r@100 .5387 / ndcg@10 .5323).
- *Ranker / two-stage (test)* = `ranker_meta.json::test_metrics` (ndcg@10 **.7231**, `lrank_t20_gainLin`).
- *"Retriever + ranker"* không có số thứ ba riêng — two-stage CHÍNH LÀ retriever + ranker end-to-end (cosine top-200 → rerank), tức `test_metrics` ở trên.

## 2. Hai biến thể số retriever — đọc cho đúng

| Biến thể | Cache item encode thế nào | Dùng khi nào | Nguồn |
|---|---|---|---|
| **Checkpoint-path** | mọi item id thật (row H = id thật chưa train, vector noise) | so run-vs-run lúc tune (Colab) | `PROGRESS.md` bảng v5, `runs.csv`, CONTRACT.md |
| **Serve-path** | row H encode **id→OOV** (content thật — đúng serving) | số chính thức / mọi số ranker + service | `eval_reference.json`, mọi số trong `ranker_meta.json` |

Serve-path warm thấp hơn checkpoint-path ~0.5–1 điểm (vd test r@200 .6608 → .6524) vì 1.142 row H từ noise-vector trở thành content-vector "hợp lý" → distractor mạnh hơn. Đây là chủ đích: số phải khớp cái user thật nhìn thấy.

> ✅ Số `final` ở **§3b** giờ có **cả hai**: checkpoint-path (run-vs-run, `runs.csv`) **và serve-path official** (`eval_reference.json`, đo qua artifacts đã re-export). Lưu ý đặc thù `final`: serve-path ndcg@10 **cao hơn** checkpoint-path (.5323 vs .4242) — ngược chiều quy luật recall — vì `final` train `history_source=embed`: lúc eval-train (checkpoint-path) row H = id thật chưa train = noise-vector làm distractor đầu bảng → dìm ndcg@10; serve-path H→OOV (content) bớt distractor → ndcg@10 phục hồi. Recall vẫn theo quy luật cũ (serve hơi thấp hơn). ✅ Số two-stage §6/§7 đã là `final` (ranker `lrank_t20_gainLin` retrain xong).

Ngoài ra khi đọc recall@K nhỏ: có **trần lý thuyết** do R > K (warm test trần r@10 ≈ .408, r@200 ≈ .993 — `docs/DATA_SPLIT.md §8`); và two-stage có **trần pool** r@200 = .6758 val/test (`ranker_meta.json::pool_ceiling`, pool `final`).

## 3. Retriever — two-tower `v5_hist64_ep2` (serve-path, full catalog) 🔴 STALE (config cũ; final = §3b)

Config thắng (nguồn: PROGRESS + CONTRACT): d=128, MLP [256], use_item_id (id_dim 128), τ=.07, logq_alpha=1, history_source=cache, history_pool=mean, **train_hist_len=64**, id_dropout=.1, bs=8192, 2 epoch (Colab A100). Checkpoint: epoch=1, step=16000.

| Slice | r@10 | r@50 | r@100 | r@200 | r@500 | ndcg@10 | ndcg@100 | n_users |
|---|---|---|---|---|---|---|---|---|
| warm **val** | .1526 | .3842 | .5146 | .6505 | .8147 | .5207 | .4820 | 14,029 |
| warm **test** | .1516 | .3847 | .5160 | .6524 | .8164 | .5155 | .4789 | 14,250 |
| **cold val** (val_cold) | .0767 | .2070 | .2925 | .3881 | .5471 | .1572 | .1926 | 8,388 |

Cold thêm pooled hitrate (150.335 pairs): @10 .0720 · @100 .2964 · @200 .4074 · @500 .5881. (test_cold = final exam: chấm trên config `final`, không phải v5 này — xem §7.)

So run-vs-run v5 (checkpoint-path, WARM TEST, Colab — chỉ để thấy lever nào ăn):

| run | đòn bẩy | r@10 | r@200 | ndcg@10 |
|---|---|---|---|---|
| v5_alpha05_ep2 | logq_alpha=.5 | .1201 | .6053 | .3446 |
| v5_embed_ep2 | history_source=embed | .1531 | .6465 | .4639 |
| v5_itemid128_ep2 | control (hist32) | .1556 | .6526 | .5072 |
| **v5_hist64_ep2** ★ | train_hist_len=64 | .1582 | .6608 | .5135 |

## 3b. Retriever final = `final` (no synopsis) — serve-path official ✅ + checkpoint-path ablation

Config CHỐT (2026-06-17): d=128, MLP [256], use_item_id (id_dim 128), τ=.07, logq_alpha=1, **history_source=embed**, history_pool=mean, **train_hist_len=128**, id_dropout=.15, bs=16384, **10 epoch**, **synopsis OFF**. best_step 31500.

**Serve-path official** (nguồn `artifacts/eval_reference.json`, đo qua artifacts đã re-export 2026-06-17, row H→OOV — đây là số CHÍNH thức để báo cáo retriever-only):

| Slice | r@10 | r@50 | r@100 | r@200 | r@500 | ndcg@10 | ndcg@100 | ndcg@200 | n_users |
|---|---|---|---|---|---|---|---|---|---|
| warm **val** | .1678 | .4030 | .5388 | .6758 | .8339 | .5343 | .5152 | .5692 | 14,029 |
| warm **test** | .1681 | .4032 | .5387 | .6758 | .8359 | .5323 | .5128 | .5665 | 14,250 |
| **cold val** (val_cold) | .0710 | .2287 | .3373 | .4664 | .6465 | .1398 | .2018 | .2411 | 8,388 |

Cold thêm pooled hitrate (150.335 pairs): @10 .0721 · @100 .3461 · @200 .4710 · @500 .6478. test_cold: ✅ **đã chấm 1 lần** (held-out, §7: cosine r@100 .3414 / r@200 .4710 / ndcg@10 .1397 — khớp val_cold, generalize). So serve-path `final` vs serve-path `v5_hist64_ep2` (§3): warm pool recall@200 .6758 > .6524, cosine ndcg@10 .5323 > .5155 → pool feeding ranker **tốt hơn** mốc đã cho .7074; cold recall@200 .4664 ≫ .3881.

> ℹ️ **Checkpoint-path khác serve-path** (đặc thù `final`): xem cảnh báo §2 — serve-path ndcg@10 (.5323) cao hơn checkpoint-path (.4242) do `history_source=embed` + H-noise lúc eval-train. Bảng ablation OFF/ON dưới là **checkpoint-path** (run-vs-run, để so synopsis trên cùng đường đo).

**Warm (test, checkpoint-path) — ablation synopsis OFF (`final`) vs ON (`final_syn`):**

| run | recall@100 | recall@200 | ndcg@10 | liked_recall@200 | liked_ndcg@10 |
|---|---|---|---|---|---|
| **`final` (OFF)** ★ | .5462 | .6852 | .4242 | .7754 | .3145 |
| `final_syn` (ON, bị bác) | .5580 | .6949 | .4886 | .7835 | .3603 |

**Cold (val_cold, 8.388 user, H→OOV) — lý do chốt OFF:**

| run | recall@100 | recall@200 | liked_recall@200 | honly_recall@200 | ndcg@10 |
|---|---|---|---|---|---|
| **`final` (OFF)** ★ | .3374 | .4664 | .5387 | .8234 | .1398 |
| `final_syn` (ON, bị bác) | .2546 | .3515 | .3905 | .7576 | .1494 |

synopsis OFF thắng cold rõ rệt (recall@200 **+.115**, liked_recall@200 +.148) trong khi chỉ kém warm chút (recall@200 −.010) — head-precision warm là việc của ranker. Vì cold serve = cosine trực tiếp (tách kênh, §7), cold gain chảy thẳng ra "Anime mới". Ablation đầy đủ + cơ chế: `docs/SYNOPSIS_EMB.md`; loss ablation (logQ/τ/β/m_hardneg): `docs/EXPERIMENTS.md §4`. (test_cold = final exam: ✅ đã chấm 1 lần, §7.)

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

## 5. Ranker — so sánh model-class (VAL, two-stage pool 200) — pool `v5` (giữ làm bằng chứng "vì sao GBDT")

> ℹ️ Bảng dưới là **so sánh lớp model** (linear vs NN-DIN vs GBDT) đo trên **pool `v5` cũ** — kết luận **GBDT > NN > linear**
> vẫn đứng và là lý do chọn LightGBM (không phụ thuộc pool). **Config production CHỐT** = `lrank_t20_gainLin` retrain
> trên pool **`final`** (val ndcg@10 **.7272**, liked_ndcg@10 **.5641**); sweep đầy đủ trên `final` ở **`docs/RANKER_EXPERIMENTS.md`**,
> số test chốt ở **§6**. Hàng `xendcg .7103` dưới = winner v5 cũ (không còn là production).

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

## 6. Two-stage CHỐT — TEST (chấm 1 lần sau khi chốt trên val) ✅ pool `final`

> ✅ **CHỐT 2026-06-18 (pool `final`)**: winner = **`lrank_t20_gainLin`** (lambdarank, `lambdarank_truncation_level=20`,
> `label_gain=[0,1,2,3,4]`), **α=1.0, K=200**, 2.949 trees, train full 100k local 4-thread ~37'. Chọn theo
> ưu tiên ndcg@10 + liked (RANKER_EXPERIMENTS.md). Số dưới = `ranker_meta.json::test_metrics` (đã re-export).

| TEST (14,250 users) | r@10 | r@50 | r@100 | r@200 | ndcg@10 | ndcg@50 | ndcg@100 | ndcg@200 |
|---|---|---|---|---|---|---|---|---|
| cosine (retriever-only) | .1681 | .4032 | .5387 | .6758 | .5323 | .4816 | .5128 | .5665 |
| **two-stage** | **.2178** | **.4835** | **.6048** | .6758 | **.7231** | **.6053** | **.6126** | .6305 |
| Δ | +.0497 | +.0803 | +.0661 | 0 (trần pool) | **+.1908** | +.1237 | +.0998 | +.0640 |

Liked-metric (test, 12.638 users có ≥1 liked query): liked_ndcg@10 .3903 → **.5615** (+.1712); liked_recall@100 .6445 → **.7182** (+.0737). val tương ứng: ndcg@10 .5343→**.7272**, liked_ndcg@10 .3894→**.5641**.

So bar MF ALS **đã tune** (full catalog, test — `docs/BASELINES.md §5`): two-stage giờ **vượt MF ndcg-opt (f128/α1) trên TẤT CẢ metric head+mid**: ndcg@10 **.7231 > .7027** (+.0204, thoải mái hơn hẳn winner cũ +.0047), ndcg@100 .6126 > .6014, r@10 .2178 > .2087, **r@100 .6048 > .5954** (winner cũ còn thua chỗ này), liked_ndcg@10 **.5615 ≫ .5052** (+.0563), liked_r@100 .7182 > .6960. Chỉ **thua deep-recall tail**: r@200 two-stage .6758 (kẹt **trần pool** = retriever r@200) **< MF recall-opt .7511 / ndcg-opt .7136**; r@500 retriever-final .8359 < MF .8797 — tail là việc của retriever (tune tiếp / nâng K). ⚠️ **Hệ quả cho đồ án**: warm-only two-stage đã **thắng MF rõ ở head-precision + liked**, chỉ nhường tail; cộng thêm **cold-start** (§7, "Anime mới"): MF/KNN = 0 by construction còn two-tower r@200 cold .4664 ≫ content. Narrative: kết hợp recall(retriever) + precision/liked(ranker) + cold — không phải chỉ "nhỉnh MF ndcg@10".

## 7. Cold — kênh serve (quyết định: tách kênh, `docs/RANKER.md §7`)

| val_cold (8.388 users) ndcg@10 | giá trị |
|---|---|
| ① cold qua blend α=1 (diagnostic) | .0002 — bị model dìm (gainLin/final) |
| ③ **kênh riêng theo cosine (CHỐT)** | **.1398** — zero regress (= cosine retriever `final`, §3b) |

α=1 dìm cold tận đáy (val_cold ndcg@10 .1398→.0002, r@100 .3373→.0087) vì model học trên pool warm-only → quyết định **tách kênh serve** giữ nguyên: cold (`is_cold`) xếp theo cosine retriever, KHÔNG qua blend. Kênh cosine cold per-K: §3b hàng cold val (`final`: r@100 .3373 / r@200 .4664 / ndcg@10 .1398). So sánh cấu trúc: two-tower cold r@200 **.4664** trong khi MF/KNN/popular = 0 by construction — claim "gợi ý được anime mới" của đồ án.

**Held-out — test_cold final exam (chấm ĐÚNG 1 lần 2026-06-18, K=200 α=1; nguồn `eval_selection.json::{baseline_test_cold, test_cold_metrics}`):**

| test_cold (8.510 users; liked n=6.341) | r@10 | r@100 | r@200 | ndcg@10 | ndcg@100 | hit@100 | hit@200 | liked_r@100 | liked_r@200 | liked_ndcg@10 |
|---|---|---|---|---|---|---|---|---|---|---|
| ③ kênh phục vụ = cosine retriever (CHỐT) | .0751 | **.3414** | **.4710** | **.1397** | .2009 | .3492 | .4766 | **.4135** | **.5484** | **.0957** |
| ① cold ép qua blend α=1 (diagnostic) | .0000 | .0099 | .4710 | **.0000** | .0053 | .0251 | .4766 | .0118 | .5484 | .0000 |

Số kênh cosine khớp val_cold (r@100 .3373 / r@200 .4664 / ndcg@10 .1398; liked_r@100 .4074 / liked_ndcg@10 .0920) → **generalize, không overfit**; blend α=1 lặp lại đúng hiện tượng dìm cold (.1398→.0002 val ≈ .1397→.0000 test). ⇒ chất lượng cold user thực tế thấy = cosine **.1397** ndcg@10 / **.3414** r@100 / liked_r@100 **.4135**. Đây là **lần chấm cuối** test_cold của toàn pipeline (file `eval_queries_test_cold.parquet` đã xoá lại giữ kỷ luật). Liked + pooled-hitrate đầy đủ trong `eval_selection.json::{baseline_test_cold, test_cold_metrics}`. *(Diagnostic "cold-item-only / honly" — candidate giới hạn trong 1.142 item H — KHÔNG nằm trong final exam; chỉ đo trên val_cold ở retriever notebook → `retriever/runs/cold_runs.csv` cột `cold_honly_recall@K`.)*

## 8. Feature importance (gain, LightGBM winner — `ranker_meta.json::feature_importance_gain`)

Winner `lrank_t20_gainLin` (gain, làm tròn nghìn — `ranker_meta.json::feature_importance_gain`):

| # | feature | gain | # | feature | gain |
|---|---|---|---|---|---|
| 1 | pool_rank | 4.047.656 | 7 | hist_cos_top5_mean | 118.274 |
| 2 | **cos_uv** | **672.714** | 8 | mal_score | 101.637 |
| 3 | hist_cos_max | 396.728 | 9 | rank | 97.178 |
| 4 | support_len | 242.792 | 10 | hist_cos_mean | 75.618 |
| 5 | log_favorites | 162.458 | 11 | theme_aff | 68.494 |
| 6 | log_scored_by | 125.555 | 12 | genre_aff | 60.663 |

Đọc (3 ý cho đồ án): (a) **pool_rank vẫn #1** (model dùng *thứ hạng* cosine per-user, cây split dễ) nhưng ở config gainLin này **cos_uv leo lên #2** (cosine thô đóng góp mạnh hơn hẳn winner v5 cũ — lambdarank label_gain tuyến tính khai thác cả giá trị cosine, không chỉ rank) → tín hiệu retriever là xương sống ở **cả hai** dạng; (b) nhóm **match-với-history** (hist_cos_max/top5/mean + genre/theme_aff) đứng cao — rerank tinh chỉnh theo độ giống anime user đã thích; (c) nhóm **prior chất lượng/phổ biến** (favorites, scored_by, mal_score, rank) bù popularity mà cosine thiếu. Categorical (type/source/...) đóng góp thấp — content đã nằm trong V.

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
