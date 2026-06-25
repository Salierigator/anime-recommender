# METRIC — Định nghĩa & lý lẽ các metric đánh giá (retriever + ranker)

> Nguồn source-of-truth cho **phần metric** của đồ án. Phủ bốn khối: (1) khung đánh giá chung
> (per-user, ranking, mask); (2) **metric nền** `recall@k` / `ndcg@k` (+ pooled hit-rate cho cold);
> (3) **liked-metric** đo *chất lượng đầu list* (`liked_recall@k` / `liked_ndcg@k`) cùng định nghĩa
> "liked" và lý do chọn ngưỡng; (4) kết quả + các lựa chọn thiết kế đã chốt/đã bác.
>
> Cùng một bộ công thức được cả hai stage dùng — retriever (harness full-catalog, rank toàn ~22.8k
> anime) và ranker (harness pool, rerank top-200) — nên số hai stage so được 1:1. Cơ chế harness/
> protocol eval (split, support–query, cold) ở `DATA_SPLIT.md` + `TWO_TOWER_MODEL.md §7`; bản đồ mọi
> con số ở `RESULTS.md`.

---

## 1. Khung đánh giá chung

Mọi metric ở đây tính **mean-per-user** trên một bảng xếp hạng đã cố định:

- **Candidate set**: các item thật được phép xuất hiện (`isfinite(logq)` — loại PAD/OOV). Retriever
  rank toàn catalog; ranker rank trong pool top-200 của retriever.
- **Relevant set** của user u: `query(u)` = các positive *held-out* (item user thật sự tương tác,
  được giấu khỏi history khi eval). `R = |query(u)|`.
- **Seen-mask** (bất biến, dùng chung mọi metric): trước khi lấy top-k, đặt `−∞` cho mọi item thuộc
  `seen(u) − query(u)` — tức gạt những gì user đã thấy *trừ chính các đáp án đang chấm*. **Không** mask
  thẳng `seen(u)` vì `query ⊆ seen` (sẽ xoá luôn đáp án). Đây là điểm tế nhị nhất của protocol.
- **Ranking**: `ranked = top-k_max` item theo score giảm dần (sau khi đã áp candidate + seen-mask).

Tất cả các metric bên dưới chỉ **đọc** ra từ cùng một `ranked` này — đổi metric **không** đổi ranking.

**K báo cáo**: `k ∈ {10, 50, 100, 200, 500}`. Headline warm của retriever là `recall@200` (độ phủ pool
cho stage sau); headline *chất lượng* là `ndcg@10`. Liked-metric báo core-k `liked_recall@{100,200}` +
`liked_ndcg@10`.

---

## 2. Metric nền: `recall@k` & `ndcg@k`

Đặt chiết khấu vị trí (rank i, 0-index) và tổng tích luỹ lý tưởng:

```
discount[i] = 1 / log2(i + 2)              # rank 1 (i=0) → 1/log2(2) = 1
idcg_cum[k] = Σ_{i=0..k-1} discount[i]      # DCG khi k item relevant xếp đầu hoàn hảo
hit[i]      = 1 nếu ranked[i] ∈ query(u), ngược lại 0
```

**Recall@k** — phần đáp án lọt top-k:
```
recall@k(u) = (Σ_{i<k} hit[i]) / R
```
Vì `R = |query(u)|` có thể > k, ngay cả ranking hoàn hảo cũng cho `recall@k < 1` khi `R > k` — đây là
hành vi đúng, không phải lỗi.

**NDCG@k** — vừa thưởng *trúng* vừa thưởng *xếp cao*, chuẩn hoá về [0,1]:
```
DCG@k(u)  = Σ_{i<k} hit[i] · discount[i]
IDCG@k(u) = idcg_cum[ min(R, k) ]
ndcg@k(u) = DCG@k(u) / IDCG@k(u)
```
Chuẩn hoá theo **`min(R, k)`** (không phải `R`): IDCG là cấu hình lý tưởng *trong giới hạn k chỗ* —
nhờ vậy `ndcg@k = 1` vẫn đạt được khi `R > k` (không thể nhét hết relevant vào top-k). Đây là biến thể
"ndcg@k với ideal cắt ở k", chuẩn cho bài toán retrieval.

**Gộp user**: lấy trung bình cộng trên các user có `R > 0`; báo kèm `n_users` (số user được chấm).

**Pooled hit-rate@k (chỉ cho cold slice)**: cold-item là lát mỏng → per-user `recall` nhiễu. Bổ sung
micro-average gộp mọi cặp (user, query):
```
hitrate@k = (Σ_u #hit@k của u) / (Σ_u R_u)
```
kèm `n_pairs = Σ_u R_u`. Dùng để đọc tín hiệu cold ổn định hơn; warm không cần.

> **Relevance nền là nhị phân**: `relevant ⇔ candidate ∈ query`, **bất kể điểm số**. Một item user
> chấm 5 và một item chấm 10 đều là "1" như nhau. Chính giới hạn này dẫn tới liked-metric ở §3–§6.

---

## 3. Vì sao cần thêm metric "chất lượng đầu list"

Ranker được **train có phân hạng** theo điểm (`grade`: `10→4, 9→3, 7–8→2, còn lại→1`) nhưng nếu **đo +
chọn model hoàn toàn nhị phân**, con số headline (`ndcg@10` nhị phân) chỉ trả lời *"có đẩy một item
đã-tương-tác lên đầu không"* — **chưa bao giờ** đo *"có đẩy item user thật sự thích lên đầu không"*. Hai
model — một cái nhồi đầu list bằng item chấm-5/chưa-chấm, một cái nhồi item 9–10 — có thể cho **`ndcg@10`
nhị phân y hệt**.

Nguyên tắc: **đo trước khi tối ưu**. Liked-metric dưới đây *đo* chất lượng đầu list nhưng giữ kỷ luật
**report-only** — không đổi label, không đổi objective, không đổi tiêu chí selection (xem §8). Nó chỉ
trả lời một câu hỏi sắc hơn trên **cùng bảng xếp hạng** mà model đã sinh ra.

---

## 4. Liked-metric: `liked_recall@k` & `liked_ndcg@k`

**Bất biến cốt lõi**: ranking và seen-mask **không đổi** (vẫn mask `seen − TOÀN BỘ query`, không phải
`seen − liked`). Chỉ đổi hai thứ: **tập được tính là hit** và **mẫu số**. Nhờ vậy liked-metric so
*trực tiếp* được với metric nhị phân trên đúng một ranking.

Với tập `liked(u) ⊆ query(u)` (định nghĩa ở §5) và `R_liked = |liked(u)|`:
```
liked_recall@k(u) = (Σ_{i<k} [ranked[i] ∈ liked(u)]) / R_liked
liked_ndcg@k(u)   = ( Σ_{i<k} [ranked[i] ∈ liked(u)] · discount[i] ) / idcg_cum[ min(R_liked, k) ]
```
Công thức y hệt §2, chỉ thay relevance nhị phân `query` → `liked`. `R_liked` đếm **toàn bộ** liked-query
của user (kể cả cái không lọt pool/top-k) nên mẫu số không bị thổi phồng.

**Gộp user**: trung bình trên các user có `R_liked > 0`; báo kèm `n_users_liked` (coverage). User không
có liked nào (xem §5) bị loại khỏi trung bình — đây là lý do `n_users_liked < n_users`.

---

## 5. Định nghĩa "liked"

"liked" = item mà user chấm **cao theo thang của chính họ** (per-user), tổng quát hoá bằng z-score với
ngưỡng τ:

```
u_mean(u), u_std(u) = mean, std của { support_score ≥ 1 }      # từ FULL support history (rated)
z(a)                = (score_a − u_mean(u)) / u_std(u)
liked(u)            = { a ∈ query(u) : score_a ≥ 1  AND  z(a) > τ }
R_liked             = |liked(u)|
```

Dự án chốt **τ = 0** ⇒ điều kiện rút gọn còn **`score_a > u_mean(u)`** ("trên trung bình của chính
mình"). Lý do chọn τ ở §6. Các quy ước đi kèm:

- **τ = 0 triệt tiêu `u_std`**: `(score − u_mean)/u_std > 0 ⟺ score > u_mean` (vì `u_std > 0`). Vì vậy
  định nghĩa **không cần ước lượng `u_std`** — không phải đặt `σ_floor`/shrinkage cho user chấm ít.
- **`score = 0`** (completed nhưng không chấm điểm) **luôn non-liked** — không chuẩn hoá được.
- **`u_mean` lấy từ FULL support** ở cả hai stage (retriever ≈ ranker) → định nghĩa nhất quán. Leak-clean:
  support là input hợp lệ của user, query đã bị giấu khỏi support khi eval.
- User **không có rated support** (`u_mean` không xác định) → không có liked nào → bị loại khỏi
  liked-metric (rơi vào phần `n_users − n_users_liked`).

Ví dụ trực giác: user mean-9 chấm một item 8 → **không** liked (dưới gu của họ); user mean-5 chấm một
item 7 → **liked**. Đây chính là điểm liked-metric vượt một ngưỡng toàn cục cứng (score ≥ 8): nó tự co
giãn theo người chấm gắt hay rộng.

---

## 6. Lý do chọn τ = 0

**Điểm mấu chốt: τ là tham số *định nghĩa metric*, không phải hyperparameter tune được.** Không thể chọn
τ bằng "τ nào cho `liked_ndcg` cao nhất" — đổi τ là đổi *cả thứ đang đo lẫn con số*, nên so điểm giữa
các τ là vô nghĩa (vòng lặp luẩn quẩn). "Phù hợp" phải xét qua **tính chất của tập liked** mà τ sinh ra.

Phân tích trên **test split** (từ `artifacts/eval_queries_test` + `users_history`, 14,250 user,
748,751 query positive; con số tái lập được vì chỉ đọc artifacts đã chốt):

| τ | coverage (`n_users_liked`/14,250) | % rated-positive là liked | median `R_liked` |
|---:|---:|---:|---:|
| **0.00** | **88.7% (12,638)** | **50.5%** | 15 |
| 0.50 | 85.2% | 32.3% | 10 |
| 1.00 | 74.0% | 16.5% | 6 |
| 1.50 | 50.5% | 6.3% | 3 |
| 2.00 | 22.6% | 1.8% | 2 |

Bốn tiêu chí đánh giá một τ "phù hợp":

1. **Coverage / đại diện** — metric nên nói về *đa số* user, không phải một thiểu số co lại. τ = 0 phủ
   **88.7%**; từ τ ≥ 1.5 coverage tụt **dưới 50%** → metric biến thành chuyện của nhóm nhỏ. Loại τ ≥ 1.5.
2. **Robustness (parameter-free)** — phân phối số rating mỗi user lệch nặng: median `n_rated = 104`
   nhưng **7.0% user chỉ rated ≤ 1 item** (`u_std` *không xác định*), **12.5% rated ≤ 10** (`u_std`
   nhiễu), 5.4% không rated gì. Với mọi **τ ≠ 0**, z-score cần `u_std` đáng tin → buộc phải bịa
   `σ_floor`/shrinkage và vẫn mong manh cho nhóm thưa. **τ = 0 né hẳn `u_std`** (xem §5) → bền vững và
   không thêm tham số tự do.
3. **Face validity** — phân phối `u_mean` qua các user: median **7.90** (IQR 7.45–8.43). Vậy τ = 0
   ("trên `u_mean`") tương đương xấp xỉ **"score ≥ 8 cho user trung vị"** — đúng ngưỡng "rất hay" của
   thang MAL, nhưng diễn đạt *per-user* (đó mới là giá trị của z-score so với ngưỡng cứng).
4. **Selectivity** — τ = 0 bắt **~50% rated-positive** (median 15 liked/user). Đây là chỗ τ = 0 *rộng
   tay*: nó đo "trên trung bình", không phải "cực thích".

**Kết luận**: **τ = 0 là lựa chọn chính đáng và dễ biện minh nhất** — lý do mạnh nhất là (2) + (3):
parameter-free, bền với 7–12% user thưa rating, và ngưỡng ngầm ≈ score 8 rất có nghĩa. Ngoài ra τ = 0
cho `n_users_liked = 12,638` ở test — **trùng khít** con số production (§7) → định nghĩa nhất quán end-to-end.

**Caveat thành thật (ghi rõ để đọc metric đúng)**: vì τ = 0 *rộng*, `liked_ndcg@10` trả lời *"top-10 có
item trên-mức-trung-bình-của-user không"*, chưa hẳn *"item user CỰC thích"*. "Sweet-spot band" theo tính
chất là **τ ∈ [0, 1.0]** (tại τ = 1.0 vẫn còn coverage 74%, bắt top ~16% rated-positive); một định nghĩa
"liked" sắc hơn có thể chọn τ ≈ 0.5–1.0, đánh đổi coverage. τ = 0 nằm ở đầu-rộng của band đó. Dự án giữ
τ = 0 và mô tả rõ liked = "above-own-mean (permissive)" để người đọc hiểu đúng phạm vi metric.

> Không tồn tại "τ tối ưu khách quan" vì chưa có tín hiệu ngoài (vd *favorites*) để hiệu chỉnh liked
> (xem §8) — đây thuần là một lựa chọn định nghĩa, và τ = 0 là default có lý lẽ vững nhất.

---

## 7. Kết quả (source-of-truth)

**Retriever `final` (cosine thuần, harness full-catalog, step 31500):**

| split | `n_users` | `n_users_liked` | `liked_recall@100` | `liked_recall@200` | `liked_ndcg@10` |
|---|--:|--:|--:|--:|--:|
| val | 14,029 | 12,478 | .6543 | .7799 | .3136 |
| test | 14,250 | 12,638 | .6500 | .7754 | .3145 |
| val_cold *(diagnostic)* | 8,388 | 6,361 | .4074 | .5387 | .0921 |

Quan sát: warm `liked_recall` > `recall` nhị phân (test `liked_recall@100` .6500 vs `recall@100` .5462)
— item user *yêu* được retriever surface tốt hơn item chỉ-tương-tác. `n_users_liked/n_users ≈ 89%` khớp
coverage τ = 0 ở §6 (phần còn lại là user chấm ít / toàn `score = 0`).

**Two-stage production (ranker `lrank_t20_gainLin`, harness pool, so trong cùng pool với cosine
baseline):** `liked_ndcg@10` **.3903 → .5615** (**+.1712**), `liked_recall@100` .6445 → .7182;
`liked_recall@200` .7690 = .7690 (= trần pool K = 200 — rerank không đổi *tập* top-200, chỉ đổi thứ tự
bên trong). Nguồn đầy đủ: `RESULTS.md §6` (`ranker_meta.json` / `eval_selection.json`).

> ⚠️ Liked retriever (bảng trên) đo **full-catalog**; liked two-stage đo trên **pool**. So liked
> *production* phải trong cùng pool harness: cosine .3903 → ranker .5615.

**Cold = diagnostic-only**: cold-item không đi qua ranker và tín hiệu liked rất thưa → dùng để soi, không
dùng để ra quyết định.

---

## 8. Lựa chọn thiết kế & phương án đã BÁC

- **Report-only (không đổi label/objective/selection)**: retriever vẫn là stage recall nhị phân; ranker
  vẫn selection Pareto trên các metric **nhị phân** (`SEL_METRICS`). Liked chỉ *báo cáo* — kỷ luật để
  tránh tự lừa bằng metric mới chưa hiểu hết.
- **BÁC graded-ndcg** (gán gain theo grade per-candidate + ideal-gains): liked-recall/ndcg gọn hơn (tái
  dùng đúng công thức §2, chỉ đổi tập relevant), trả lời cùng câu hỏi "đầu list có phải item user thích".
- **BÁC ngưỡng toàn cục** (`score ≥ 8` hoặc `≥ 9`): chọn per-user above-own-mean — "thích" theo thang
  riêng của từng người (xem ví dụ §5), không phạt user chấm gắt / thưởng user chấm rộng.
- **BÁC z ≥ τ với τ > 0**: chọn τ = 0 — phân tích §6 cho thấy nó parameter-free, bền với user thưa
  rating, và face-valid (≈ score 8), trong khi τ > 0 vừa cần `u_std` đáng tin vừa cắt coverage.
- **Favorites-as-hard-positive** (tín hiệu "thích" tường minh từ MAL favorites): **chưa làm** — ngoài
  phạm vi liked-metric; cũng là lý do hiện chưa có anchor ngoài để hiệu chỉnh τ.

---

*Cross-ref: protocol eval & harness — `DATA_SPLIT.md`, `TWO_TOWER_MODEL.md §7`; metric nền cho baselines
— `BASELINES.md`; metric two-stage & selection — `RANKER.md §6`; bản đồ mọi con số — `RESULTS.md`.*
