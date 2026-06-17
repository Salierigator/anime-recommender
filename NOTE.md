# NOTE — MF baseline mạnh thế này, two-stage có vượt được metric WARM không?

> Phân tích (2026-06-17) trả lời câu hỏi: sau khi tune, `retriever/baselines/mf.txt` đã rất mạnh trên
> warm → kiến trúc two-stage (retriever two-tower → ranker LightGBM) còn cửa vượt MF trên **warm** không,
> hay lợi thế phải neo vào chỗ khác?
>
> ⚠️ **Tính chất của note này**: số two-stage hiện có (.7074) đo trên pool `v5_hist64_ep2` (retriever cũ).
> Retriever vừa chốt `final` và re-export, nhưng **ranker CHƯA retrain** trên pool `final` → đây là phân
> tích **trần lý thuyết + dự đoán**, không phải số đo mới. Muốn chốt bằng số: retrain ranker (`docs/RANKER.md §9`)
> rồi đo lại. Nguồn số: `retriever/baselines/mf.txt`, `artifacts/eval_reference.json`, `docs/RESULTS.md §4/§6`,
> `runs.csv`.

---

## TL;DR

**Không — trên warm thuần, two-stage KHÔNG thắng MF một cách thuyết phục.**

- **ndcg@10 (head precision)**: two-stage *hoà / nhỉnh cực sát* MF — biên ±.005, nằm trong vùng nhiễu run-to-run. Không phải lợi thế đáng kể.
- **recall (tail, @200/@500)**: two-stage **thua chắc** MF, và đây là **trần cấu trúc** không phá được bằng ranker.
- **Lợi thế thật của two-stage = cold-start** (MF/itemKNN/popular = 0 by construction) + tính bổ trợ recall(retriever)+precision(ranker)+cold, **không phải** "đè MF trên warm".

Narrative đồ án nên neo vào cold + bổ trợ, đừng bán câu chuyện "two-stage thắng MF warm".

---

## 1. Mốc so sánh (warm test, full catalog)

**MF ALS đã tune per-axis** (`mf.txt`, f128, reg .05, iter 15, refit full train):

| MF config | recall@100 | recall@200 | recall@500 | ndcg@10 | liked_ndcg@10 |
|---|---|---|---|---|---|
| **ndcg-optimal** (α=1) | .5954 | .7136 | .8405 | **.7027** | **.5052** |
| **recall-optimal** (α=10) | **.6223** | **.7511** | **.8797** | .6374 | .4442 |

α (confidence weighting) đánh đổi head↔tail: α thấp → ndcg@10 cao / recall sâu thấp; α cao → ngược lại (cơ chế: `docs/BASELINES.md §3.6`). MF "lấy được cả hai trục tốt nhất" nếu cho chọn config theo trục.

**Two-stage hiện có** (`docs/RESULTS.md §6`, xendcg α=1, K=200, **pool retriever cũ `v5_hist64_ep2`**):

| | recall@100 | recall@200 | ndcg@10 |
|---|---|---|---|
| cosine (retriever-only) | .5160 | .6524 | .5155 |
| **two-stage** | .5811 | .6524 | **.7074** |

**Retriever `final` serve-path** (`eval_reference.json`, pool MỚI feed ranker sau re-export):

| slice | recall@100 | recall@200 | recall@500 | ndcg@10 |
|---|---|---|---|---|
| warm test | .5387 | **.6758** | .8359 | **.5323** |
| cold val | .3373 | .4664 | .6465 | .1398 |

---

## 2. Điểm mấu chốt: TRẦN POOL (pool ceiling)

Two-stage = retriever trả **top-K=200** cosine → ranker **rerank trong pool đó**. Ranker chỉ *sắp xếp lại*
những gì retriever đã lấy — **không thêm được** item retriever bỏ sót. Hệ quả toán học:

> **recall@K của two-stage (K ≤ 200) bị chặn trên bởi recall@200 của retriever.**
> Riêng recall@200 của two-stage **= recall@200 của retriever** y hệt (rerank không đổi *tập* top-200, chỉ đổi thứ tự).

Với `final`: trần pool = retriever serve-path **recall@200 = .6758**.

So với MF:
- **recall@200**: two-stage ≤ **.6758** < MF recall-opt **.7511**, < MF ndcg-opt **.7136**. → **thua, không thể hoà.**
- **recall@500**: retriever .8359 < MF recall-opt **.8797**; hơn nữa pool K=200 nên two-stage còn không "với" tới @500 nếu không nâng K.

→ **Trên recall tail, two-stage thua MF là cấu trúc.** Đổi từ `final_syn`/`v5` sang `final` có nhích trần pool lên (.6758 vs v5 .6524) nhưng vẫn dưới MF xa. Muốn vượt recall phải **làm retriever recall > MF recall** (tune tiếp) hoặc **nâng K** (đánh đổi precision/latency + pool ranker phình) — không phải việc ranker làm được.

## 3. Head precision (ndcg@10): hoà / nhỉnh sát, biên mong manh

Đây là trục two-stage có cửa, vì ranker được sinh ra để sửa head:
- Two-stage cũ **.7074** vs MF ndcg-opt **.7027** → +.0047. *Có* vượt, nhưng biên < .005 — cỡ nhiễu giữa các lần train (đừng coi là thắng chắc).
- **Pool `final` tốt hơn cho ranker**: cosine ndcg@10 .5323 > v5 .5155, và pool recall@200 .6758 > .6524 (nhiều positive trong pool hơn để ranker kéo lên top-10). → có cơ sở để two-stage trên `final` **giữ hoặc nhích** ndcg@10 ≥ .7074, tức vẫn ~ngang/nhỉnh MF ndcg-opt.
- **Lưu ý**: retriever-only ndcg@10 của `final` **không** quyết định ndcg@10 two-stage — ranker dựng lại head từ feature (pool_rank, hist_cos_max, mal_score; `docs/RESULTS.md §8`). Cái quyết định là **chất lượng pool** (recall + thứ tự thô), và pool `final` tốt hơn.

→ Kết luận trục này: **hoà đến nhỉnh-rất-sát MF**. Không đủ để gọi là "vượt warm" một cách thuyết phục; cần retrain để xác nhận con số.

## 4. recall@100: MF vẫn nhỉnh

Two-stage cũ .5811 < MF ndcg-opt .5954 < MF recall-opt .6223. Pool `final` r@100 .5387 (> v5 .5160) + trần
rộng hơn → two-stage r@100 có thể vượt .5811, *có khả năng* chạm ~MF ndcg-opt .5954, nhưng **khó vượt MF
recall-opt .6223**. Tức ở @100 cũng nghiêng về hoà/thua nhẹ.

## 5. Lợi thế thật: COLD-START (chỗ MF = 0)

| method | cold recall@200 | cold ndcg@10 | ghi chú |
|---|---|---|---|
| MF ALS | **N/A = 0** | **0** | item_factors của H không được học (H cách ly khỏi train) |
| itemKNN | **N/A = 0** | **0** | H không có co-occurrence train |
| popular | **N/A = 0** | **0** | popularity train của H = 0 |
| content (mean+IDF) | .2177 (test_cold) | .0411 | comparator cold |
| **retriever `final`** | **.4664** (val_cold) | **.1398** | ≈ **2.1×** content; ∞× so với CF cổ điển |

Đây là claim cấu trúc mạnh nhất: **không model CF cổ điển nào gợi ý được anime mới**, kể cả MF đã tune.
Retriever `final` được chọn (thay `final_syn`) chính vì cold tốt hơn (cold r@200 +.115 so với synopsis ON —
`docs/SYNOPSIS_EMB.md`). Cold serve = cosine retriever trực tiếp (tách kênh, `docs/RANKER.md §7`), nên cold
gain chảy thẳng ra mục "Anime mới" — không bị ranker dìm.

## 6. Kết luận & việc cần làm để chốt bằng số

**Trả lời câu hỏi**: MF tune xong mạnh đến mức **two-stage không vượt được warm một cách thuyết phục** —
chỉ *hoà/nhỉnh sát* ở ndcg@10 (biên cỡ nhiễu) và *thua cấu trúc* ở recall tail (trần pool .6758 < MF .7511).
Việc đổi sang `final` nhích pool warm lên chút (lợi cho ndcg@10) nhưng **không lấp được khoảng cách recall**
tới MF — recall là việc của retriever, không phải ranker.

Vậy giá trị của kiến trúc two-stage **không** nằm ở "thắng MF trên warm" mà ở:
1. **Cold-start** — gợi ý được anime mới (MF/KNN = 0).
2. **Bổ trợ** — retriever lo recall/cold, ranker lo head-precision warm; gộp lại phủ cả hai mặt mà một MF
   đơn lẻ không tối ưu đồng thời (MF phải chọn α theo trục).
3. **Khả năng mở rộng feature** ở stage ranker (popularity, recency, history-match) mà MF thuần không có.

**Để trả lời bằng số thật (chưa làm — ngoài scope re-export)**: retrain ranker trên pool `final`
(`docs/RANKER.md §9`: build_eval → sanity gate → build_train → Colab → eval → export), rồi đối chiếu
two-stage(final) vs `mf.txt`. Dự đoán: ndcg@10 ≥ .7074 (hoà/nhỉnh MF ndcg-opt .7027), recall@200 ≈ .6758
(vẫn < MF .7511). Note này sẽ được thay bằng số đo sau khi retrain.
