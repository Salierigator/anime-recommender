# SYNOPSIS_EMB — embedding synopsis trong ItemTower

> **✅ XÁC NHẬN (2026-06-16)**: synopsis (frozen `all-MiniLM-L6-v2`, dim 64) vào **config final** —
> cải thiện **warm** (ablation on/off, §Kết quả). Cold: chờ số (§Cold — TODO). Làm giàu content path
> của retriever bằng synopsis (≈100% tiếng Anh). Thiết kế + ladder: bên dưới; bối cảnh thử nghiệm
> chung: `docs/EXPERIMENTS.md §1`.

## Thiết kế (đã code)

- **Embedding fix cứng (frozen)**: `retriever/data_prep/07_synopsis_emb.py` gọi `all-MiniLM-L6-v2`
  (384 dim) 1 LẦN offline trên toàn bộ synopsis → `train-data/synopsis_emb.npy [num_items,384]`
  + `synopsis_low_info.npy [num_items] bool` + `synopsis_meta.json`. L2-normalize sẵn trong script
  (→ export/serve nhất quán). KHÔNG train, TÁCH khỏi vòng re-export retriever. Swap `bge-small`/`gte-small`
  (cùng 384) sau này qua `cfg.synopsis_emb_file` mà không đụng code.
- **Projection trainable (1 phần model)**: `ItemTower` (`retriever/src/model.py`) chiếu raw 384 →
  `cfg.synopsis_dim` bằng `_mlp` — config final dùng **64** (`synopsis_proj_hidden=[]` = Linear thuần;
  `[128]` nếu underfit). Concat vào content path (sau studios, trước id). `in_dim` tự cộng synopsis_dim.
  (Mặc định knob `synopsis_dim=48`; final chọn 64.)
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

## Kết quả (ablation on/off — warm)

Ablation **sạch** trên Colab: hai run **cùng config v_final** (history_source=embed, train_hist_len=128,
10 epoch, adam, τ.07, d128), **chỉ khác** `use_synopsis`:

| run | use_synopsis | synopsis_dim |
|---|---|---|
| `final` | OFF | — |
| `final_syn` ★ (= best.pt hiện tại) | ON | 64 |

Số **checkpoint-path** (run-vs-run, nguồn `runs.csv`; serve-path chính thức sẽ đo lại sau khi chốt):

| slice | metric | OFF (`final`) | ON (`final_syn`) | Δ |
|---|---|---|---|---|
| val  | ndcg@10 | .4961 | .5208 | **+.0247** |
| val  | r@200   | .6852 | .6935 | +.0083 |
| val  | r@10    | .1583 | .1702 | +.0119 |
| test | ndcg@10 | .4937 | .5188 | **+.0251** |
| test | r@200   | .6852 | .6949 | +.0097 |

**Đọc**: synopsis cải thiện warm — mạnh nhất ở **ndcg@10** (+.025, vượt rõ ngưỡng noise ~.004 của
`DATA_SPLIT.md §8`); recall@200 tăng nhẹ nhưng vẫn trên noise. Gain dồn vào **head-precision / ranking
quality** hơn là recall thô, đúng kỳ vọng của một tín hiệu content bổ sung cho 9 feature categorical.

## Cold — TODO

Ablation trên CHỈ đo warm (`runs.csv` warm-only). Tác động synopsis lên **cold** (anime mới, content-only,
id→OOV) — nơi synopsis *kỳ vọng* giúp nhiều nhất — **chưa có số on/off cùng config**. Để placeholder,
user cập nhật sau.

> ⚠️ KHÔNG so cold serve-path hiện tại (val_cold r@200 .3515) với số doc cũ (.3881): số cũ là của base
> `v5_hist64_ep2` (cache, hist64, 2ep) — khác config, không phải `final` no-syn → không kết luận hướng
> cold từ cặp này. Chờ ablation cold đúng (final vs final_syn).

## Ladder thử nghiệm (xem `docs/EXPERIMENTS.md`)

1. ✅ **DONE** — Baseline `use_synopsis=True, dim=48, hidden=[]` (MiniLM) vs control OFF: warm cải thiện
   (bảng trên), đã nâng và chọn **`synopsis_dim=64`**.
2. ⏳ Chưa làm — `synopsis_proj_hidden=[128]`, rồi swap `bge-small-en-v1.5` (cùng 384).
