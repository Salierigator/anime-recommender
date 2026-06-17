# SYNOPSIS_EMB — embedding synopsis trong ItemTower

> **❌ KHÔNG VÀO CONFIG FINAL (2026-06-17)**: đã test kỹ ablation on/off cùng config `v_final` —
> synopsis (frozen `all-MiniLM-L6-v2`, dim 64) **cải thiện warm** (head-precision) nhưng **regress cold
> mạnh** (recall@200 −.115, liked_recall@200 −.148, ngay cả honly cold-vs-cold −.066; §Cold ablation).
> Vì retriever **ưu tiên cold** (cold serve = cosine trực tiếp, tách kênh — `docs/RANKER.md §7`) nên
> **config final chốt `final` = synopsis OFF**. Code synopsis GIỮ NGUYÊN (chỉ `use_synopsis=False`),
> có thể bật lại nếu sau này phục vụ cold qua kênh khác. Bối cảnh thử nghiệm chung: `docs/EXPERIMENTS.md §1`.
>
> ✅ **Re-export DONE (2026-06-17)**: `best.pt`/`artifacts/` giờ là `final` (synopsis OFF — `artifacts/CONTRACT.md`
> epoch=7 step=31500). Serve-path official của `final` đã đo (`docs/RESULTS.md §3b`). Bảng ablation dưới giữ
> **checkpoint-path** (run-vs-run, `runs.csv`/`cold_runs.csv`) để so synopsis on/off trên cùng đường đo. Còn lại:
> retrain ranker trên pool `final` (`docs/RANKER.md §9`).

## Thiết kế (đã code)

- **Embedding fix cứng (frozen)**: `retriever/data_prep/07_synopsis_emb.py` gọi `all-MiniLM-L6-v2`
  (384 dim) 1 LẦN offline trên toàn bộ synopsis → `train-data/synopsis_emb.npy [num_items,384]`
  + `synopsis_low_info.npy [num_items] bool` + `synopsis_meta.json`. L2-normalize sẵn trong script
  (→ export/serve nhất quán). KHÔNG train, TÁCH khỏi vòng re-export retriever. Swap `bge-small`/`gte-small`
  (cùng 384) sau này qua `cfg.synopsis_emb_file` mà không đụng code.
- **Projection trainable (1 phần model)**: `ItemTower` (`retriever/src/model.py`) chiếu raw 384 →
  `cfg.synopsis_dim` bằng `_mlp` — biến thể `final_syn` (đã test, bị bác) dùng **64**
  (`synopsis_proj_hidden=[]` = Linear thuần; `[128]` nếu underfit). Concat vào content path (sau studios,
  trước id). `in_dim` tự cộng synopsis_dim. (Mặc định knob `synopsis_dim=48`. **Config final = OFF**, nhánh
  này không kích hoạt.)
- **Low-info → vec học được**: row `synopsis_low_info` (NaN | <50 ký tự | placeholder "No synopsis…",
  ~15% + PAD/OOV) KHÔNG dùng projection mà thay bằng `no_synopsis` (Parameter học được) → tower phớt
  lờ synopsis rác, không làm hỏng phần synopsis tốt. Thay SAU projection nên né NaN do normalize vec ~0.
- **Tiền xử lý** (trong 07): strip `(Source: ...)` ở đuôi (~32.9% dính); cờ low_info như trên.

## Dùng

```bash
# 1) sinh artifact (local CPU, 1 lần, sau prep 01):
venv/bin/python retriever/data_prep/07_synopsis_emb.py --device cpu
# -> đẩy synopsis_emb.npy + synopsis_low_info.npy lên Drive recommender_train_colab/train-data/
# 2) bật khi train: cfg.use_synopsis=True (notebook cell 3 / search space / CLI --synopsis)
```

`export.py` tự đóng synopsis vào `item_vectors.npy` (chảy qua `refresh_item_cache`) — KHÔNG cần sửa.
Knob synopsis nằm trong `TwoTowerConfig` (`use_synopsis`, `synopsis_dim`, `synopsis_proj_hidden`,
`synopsis_normalize`, `synopsis_emb_file`, `synopsis_low_info_file`) → search lật được như mọi đòn bẩy.

## Kết quả ablation on/off (cùng config v_final)

Ablation **sạch** trên Colab: hai run **cùng config v_final** (history_source=embed, train_hist_len=128,
10 epoch, adam, τ.07, d128, logQ α=1), **chỉ khác** `use_synopsis`:

| run | use_synopsis | synopsis_dim |
|---|---|---|
| `final` ★ (config CHỐT, = best.pt/artifacts hiện tại) | OFF | — |
| `final_syn` (bị bác) | ON | 64 |

Số **checkpoint-path** (run-vs-run; nguồn `runs.csv` warm + `cold_runs.csv` cold @ best_step; serve-path
chính thức đo lại sau re-export). Δ = ON − OFF (tác động của synopsis; **+** = synopsis giúp).

### Warm (test) — synopsis CẢI THIỆN, dồn vào head-precision

| metric | OFF (`final`) | ON (`final_syn`) | Δ |
|---|---|---|---|
| recall@100       | .5462 | .5580 | +.0118 |
| recall@200       | .6852 | .6949 | +.0097 |
| **ndcg@10**      | .4242 | .4886 | **+.0644** |
| liked_recall@200 | .7754 | .7835 | +.0081 |
| liked_ndcg@10    | .3145 | .3603 | +.0458 |

**Đọc**: synopsis giúp warm — nhưng gain dồn vào **head-precision / ranking quality** (ndcg@10 +.064,
liked_ndcg@10 +.046) chứ không phải recall thô (recall@200 chỉ +.010, recall@100 +.012). Đúng kỳ vọng của
một tín hiệu content bổ sung cho 9 feature categorical: nó tinh chỉnh thứ tự đầu danh sách (nơi id đã mang
tín hiệu collaborative), không mở rộng vùng phủ.

> 🔧 **Sửa bug nhãn (2026-06-17)**: bảng cũ ghi cột "ndcg@10" = .4961/.5208 (val) .4937/.5188 (test) —
> **đó thực ra là ndcg@100** (≈ `CONTRACT.md` final_syn ndcg@100 .5207). ndcg@10 THẬT (runs.csv) thấp hơn
> nhưng khoảng cách synopsis LỚN hơn (+.064 thay vì +.025). Bảng trên đã dùng số đúng (cột `test_ndcg@10`).

### Cold ablation (val_cold, 8388 user, H→OOV) — synopsis REGRESS, đây là lý do chốt OFF

| metric | OFF (`final`) ★ | ON (`final_syn`) | Δ |
|---|---|---|---|
| recall@100       | .3374 | .2546 | **−.0828** |
| recall@200       | .4664 | .3515 | **−.1149** |
| liked_recall@200 | .5387 | .3905 | **−.1482** |
| honly_recall@200 | .8234 | .7576 | **−.0658** |
| ndcg@10          | .1398 | .1494 | +.0096 |

(test_cold = final exam, **chưa chấm** — số trên là val_cold để debug, đủ để quyết hướng vì gap rất lớn,
vượt xa noise .004.)

**Đọc**: synopsis làm cold **kém đi rõ rệt** ở mọi recall — ngược hẳn động cơ ban đầu ("synopsis kỳ vọng
giúp cold nhiều nhất vì cold chỉ có content"). Chỉ ndcg@10 cold nhỉnh tí (+.0096, cùng hiệu ứng head như
warm) trong khi recall sụt — nhất quán: synopsis chỉnh đầu danh sách nhưng đẩy nhiều positive cold ra khỏi
top-K.

## Vì sao synopsis +warm nhưng −cold? (phân tích cơ chế)

Mâu thuẫn warm↑ / cold↓ không phải nhiễu — có cơ chế rõ. Tách 2 lớp:

**(A) "Warm sắc hơn chèn cold" (full-catalog).** Ở cold eval realistic, candidate = TOÀN bộ 22.8k item
(warm + cold H). Synopsis làm vector warm đặc trưng hơn → khi rank full-catalog, warm dồn lên đầu và
**chèn** item cold (vốn yếu hơn) khỏi top-K. Đây là tác động gián tiếp.

**(B) Synopsis tự nó làm biểu diễn cold TỆ hơn (honly).** Quan trọng hơn: chế độ **honly** (candidate CHỈ
tập cold H, cả hai run đều encode id→OOV — loại trừ hoàn toàn "warm chèn") vẫn cho OFF > ON (+.066). Tức
synopsis **trực tiếp** làm vector cold kém phân biệt. Ba lý do cộng hưởng:

1. **Co-adaptation với id**: id-dropout chỉ 15% → 85% bước train ItemTower thấy **id thật**. Projection
   synopsis + MLP học cách hữu ích KHI CÓ id (như một lớp refine trên nền id collaborative). Lúc cold
   (id→OOV), tín hiệu điều kiện đó mất nền → thành nhiễu lệch thay vì bổ trợ.
2. **Cạnh tranh capacity**: concat 64 chiều synopsis vào content path làm MLP phân bổ lại trọng số, "ăn"
   vào phần dành cho 9 feature cấu trúc (genres/themes/studios/type/…) — vốn là TẤT CẢ những gì cold dựa
   vào (id đã OOV). Đánh đổi này lỗ ở cold.
3. **MiniLM frozen ít discriminative cho anime**: `all-MiniLM-L6-v2` là sentence-encoder tổng quát, không
   fine-tune cho "tương đồng gu xem anime"; synopsis ngắn + nhiều mô-típ → embedding gom theo *văn phong
   bề mặt* hơn là tín hiệu taste. Warm có id sửa lại; cold không có gì sửa → nhiễu synopsis lấn át.

**Tổng**: synopsis là tín hiệu *refine-trên-nền-id* (giúp warm head-precision) chứ không phải tín hiệu
*content-độc-lập* đủ mạnh để cải thiện cold. Vì sản phẩm phục vụ cold bằng cosine retriever trực tiếp
(tách kênh, `docs/RANKER.md §7`), ta chốt **OFF**.

## Ladder thử nghiệm (xem `docs/EXPERIMENTS.md`)

1. ✅ **DONE** — `use_synopsis=True` (MiniLM) vs OFF, đã quét dim 48/64: warm cải thiện, đã nâng lên
   `synopsis_dim=64`. **Nhưng** ablation cold (final vs final_syn) cho thấy regress cold → **BÁC khỏi
   final** (chốt OFF, §Cold ablation + §Vì sao).
2. ⏳ Chỉ làm nếu muốn phục vụ cold qua kênh riêng — `synopsis_proj_hidden=[128]`, swap
   `bge-small-en-v1.5` (cùng 384), hoặc tách synopsis khỏi co-adaptation với id (tăng id-dropout / freeze
   id khi học projection). Không thuộc config final hiện tại.
