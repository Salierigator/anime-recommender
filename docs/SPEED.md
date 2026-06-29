# SPEED — hiệu năng phục vụ (online latency & throughput)

Đo **chi phí phục vụ một gợi ý** của từng phương pháp (baselines / two-tower / two-stage) để bổ
sung trục *efficiency* cho trục *accuracy* ở `docs/RESULTS.md`. Mục đích chính: **biện minh quyết
định kiến trúc two-stage** — accuracy tăng mạnh (ndcg@10 .5323→.7231, `RESULTS.md §6`) đổi lấy bao
nhiêu độ trễ, và độ trễ đó nằm ở đâu.

> Số gốc (máy sinh): `benchmark_speed.txt` (chạy `venv/bin/python benchmark_speed.py 3000`).
> Phân tách rerank feature/model: probe rời (xem §5). Snapshot dưới = run **2026-06-29**, Apple M3.

---

## 1. Đo cái gì — định nghĩa "tốc độ recommend"

Mỗi request = lịch sử 1 user → danh sách top-20 anime. **Vùng tính giờ** (online, lặp mỗi request):

```
fetch history → encode/score → mask seen → top-K   (+ rerank LightGBM cho two-stage)
```

**Loại khỏi đồng hồ** (chỉ làm 1 lần lúc khởi động, không phải chi phí mỗi request): load artifact,
train offline (ALS / item-sim / popularity count / content matrix), lookup tên anime để hiển thị.

Hai chỉ số:
- **per-request latency** (E=1, batch=1): p50 / mean / p90 / p95 tính bằng ms — phản ánh độ trễ user thật cảm nhận.
- **throughput** (batch 64): rec/s — server xử lý nhiều request song song bằng phép toán vector hoá.

## 2. Điều kiện đo & kiểm soát công bằng

| Hạng mục | Giá trị |
|---|---|
| Host | **Apple M3 · 8 logical cores · 16 GB RAM · macOS** |
| Stack | python 3.9.6 · torch 2.8.0 · lightgbm 4.6.0 · numpy 2.0.2 |
| Device | **CPU-only** (tắt CUDA), **pin 4 threads** đồng đều mọi method (torch + BLAS/OpenBLAS/MKL + implicit) |
| Sample | 3.000 user **warm-test** ngẫu nhiên (seed=123) |
| top_k | 20 (danh sách cuối trả user); k_retrieve (two-stage) = 200 |
| Candidate pool | 22.821 anime có `logq` hữu hạn |
| Warmup | 5 request trước khi bấm giờ (loại JIT/cache lạnh) |

Kiểm soát công bằng đáng nêu khi bảo vệ:
- **Pin threads đồng đều** → không method nào ăn gian số core.
- **Tách offline/online sạch** → đo đúng chi phí *mỗi request*, không tính phần khởi động.
- **Percentile p50/p90/p95** → bắt cả tail latency, không chỉ trung bình.
- **Cô lập tiến trình**: baselines (`retriever/src`) và model (`ranker/src` qua `Recommender`) đụng tên
  module flat (`config`, `metrics`) nên chạy ở **2 subprocess riêng**, rồi gộp kết quả.

## 3. Kết quả — latency + throughput

```
method              p50     mean      p90      p95      rec/s
--------------------------------------------------------------
random             0.09     0.09     0.10     0.11      18236
popular            0.06     0.06     0.07     0.07      65731
meta_popular       0.06     0.06     0.06     0.07      66060
content            0.80     0.86     1.00     1.11       5179
itemknn            0.18     0.19     0.25     0.27      16104
mf                 0.30     0.32     0.39     0.44       7684
two-tower          0.21     0.23     0.27     0.29      16042
two-stage          6.56     6.62     6.93     7.08        192
```

Đọc:
- **Non-personalized (popular/meta_popular)**: ~0.06ms — chỉ là 1 vector điểm cố định expand cho mọi user; nhanh nhất, throughput >65k rec/s.
- **content / itemknn / mf**: 0.2–0.8ms — score full catalog (matmul hoặc gather sim), vẫn sub-ms.
- **two-tower (retriever đơn)**: 0.21ms — encode user tower + cosine full-catalog top-20. **Rẻ ngang baselines mạnh nhất** dù là deep model, vì item vectors precompute, online chỉ còn 1 matmul.
- **two-stage**: 6.56ms — chậm hơn ~1–2 bậc, nhưng vẫn **dưới ngưỡng latency tương tác** (p50 6.6ms, p95 7.1ms). Đây là cái giá của accuracy.

## 4. Two-stage — chi phí nằm ở đâu

```
retrieval (cosine top-200)    p50=0.44  mean=0.45     (~7%)
rerank   (29 feat + LightGBM) p50=6.12  mean=6.16     (~93%)
tổng                          p50=6.56  mean=6.62
```

**Retrieval gần như miễn phí** (0.44ms — chỉ nhỉnh hơn two-tower đơn vì lấy top-200 thay vì top-20 và
loại cold_idx). **Toàn bộ chi phí two-stage nằm ở rerank.**

## 5. Bóc tách rerank — bottleneck là chính model, không phải feature

Probe rời (cùng host, 4 threads, 3000 user, warmup 5) tách rerank thành 2 mảnh:

| Thành phần rerank | p50 | mean | Tỉ trọng |
|---|---|---|---|
| feature (`cross_features` + `build_frame` pandas, 29 cột × 200 cand) | 0.92ms | 0.91ms | **~13%** |
| **LightGBM `predict` + argsort top-20** | **5.46ms** | **5.86ms** | **~87%** |

⇒ Bottleneck là **inference GBDT**: booster `lrank_t20_gainLin` có **2.949 cây** (`RESULTS.md §6`), mỗi
request chấm 200 candidate → ~200 × 2.949 ≈ **590k lượt duyệt cây**. Feature assembly (kể cả overhead
pandas DataFrame) chỉ chiếm ~13%. argsort top-20 từ 200 phần tử không đáng kể.

**Hệ quả tối ưu hoá (nếu cần giảm latency):** latency rerank tỉ lệ thuận với **#trees × k_retrieve**.
Giảm k_retrieve 200→100 hoặc cắt #trees (early-stopping chặt hơn) sẽ cắt latency gần tuyến tính — đây
là núm vặn rõ ràng, không cần đổi kiến trúc. Tối ưu pandas (mảng numpy thẳng vào `predict`) chỉ cứu được ~13%.

## 6. Defensibility — câu hỏi hội đồng có thể hỏi & trả lời

**"Chạy trên máy gì? Số 7ms có ý nghĩa không?"** → Apple M3, CPU-only, 4 threads (stamp trong header
file). Số tuyệt đối phụ thuộc host; **kết luận bền vững là thứ hạng tương đối** giữa các method và phép
phân rã chi phí — không đổi khi đổi máy. 6.6ms p50 dưới ngưỡng tương tác nên two-stage production-viable.

**"So baselines 1-tầng với two-stage 2-tầng có công bằng không?"** → Đo là **tổng chi phí phục vụ một
top-20** của từng phương pháp đúng như nó được deploy (end-to-end). Không phải so từng tầng; là so
"tốn bao nhiêu để ra danh sách cuối". Framing này đúng và đã pin threads đồng đều.

**"Rerank 6ms — đó là LightGBM hay là code chậm?"** → Đã đo (§5): **~87% là `booster.predict`** (2.949
cây × 200 cand), ~13% feature build. Không phải overhead implementation; là chi phí model thật.

**"MF fit subset 15k user — có làm méo latency không?"** → Không. Online latency của MF = fold-in +
matmul item-factors, **phụ thuộc factors=128 và N, KHÔNG phụ thuộc #train-user**. Subset chỉ để fit
nhanh; latency phục vụ trung thực. itemknn fit FULL train (K=50).

**"Vì sao two-tower (0.21ms) rẻ ngang baseline mà accuracy cao hơn?"** → Item vectors precompute
offline; online chỉ encode user tower (1 lượt MLP qua history) + 1 matmul cosine. Chi phí deep model
đã trả lúc train, không phải lúc serve.

## 7. Giới hạn / caveat (nêu trong báo cáo để trung thực)

- **Một run, máy chia sẻ tải.** Percentile bắt tail *trong* run nhưng không có độ lệch run-to-run; số
  tuyệt đối dao động ~10% giữa các lần (vd two-stage p50 6.5–7.1ms tuỳ tải nền). Thứ hạng + phân rã ổn định.
- **Chỉ đo warm user.** Đường **cold không qua rerank** (tách kênh serve, `RESULTS.md §7`): cold xếp
  theo cosine retriever ≈ tốc độ tầng retrieval (~0.5ms), **rẻ hơn** two-stage warm. Không benchmark riêng vì nó là tập con của retrieval.
- **CPU-only.** GPU sẽ đổi cán cân (two-tower/LightGBM hưởng lợi khác nhau); con số này phản ánh deploy CPU thực tế của service.
- **batch=64 throughput** dùng vector hoá theo lô — server thật có thể không luôn gom đủ 64 request đồng thời, nên rec/s là trần lý thuyết khi tải cao.

## 8. Tái tạo

```bash
venv/bin/python benchmark_speed.py 3000     # → benchmark_speed.txt (mặc định 500 nếu bỏ N)
```
Script tự spawn 2 worker subprocess (baselines / model), pin threads, stamp host+stack vào header.
Phân tách rerank (§5): probe rời tách `cross_features+build_frame` vs `booster.predict` trên cùng setup.
