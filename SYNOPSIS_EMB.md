# SYNOPSIS_EMB — embedding synopsis trong ItemTower

> **ĐÃ IMPLEMENT** (2026-06-14, gate `cfg.use_synopsis`, mặc định OFF). Làm giàu content path
> của retriever bằng synopsis (≈100% tiếng Anh). Chi tiết thử nghiệm + số liệu: `docs/EXPERIMENTS.md`.

## Thiết kế (đã code)

- **Embedding fix cứng (frozen)**: `retriever/data_prep/07_synopsis_emb.py` gọi `all-MiniLM-L6-v2`
  (384 dim) 1 LẦN offline trên toàn bộ synopsis → `train-data/synopsis_emb.npy [num_items,384]`
  + `synopsis_low_info.npy [num_items] bool` + `synopsis_meta.json`. L2-normalize sẵn trong script
  (→ export/serve nhất quán). KHÔNG train, TÁCH khỏi vòng re-export retriever. Swap `bge-small`/`gte-small`
  (cùng 384) sau này qua `cfg.synopsis_emb_file` mà không đụng code.
- **Projection trainable (1 phần model)**: `ItemTower` (`retriever/src/model.py`) chiếu raw 384 →
  `cfg.synopsis_dim` (mặc định 48 ≈ ngang khối genres/themes/studios) bằng `_mlp` —
  `cfg.synopsis_proj_hidden=[]` (Linear thuần) | `[128]` nếu underfit. Concat vào content path
  (sau studios, trước id). `in_dim` tự cộng synopsis_dim.
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

## Ladder thử nghiệm (xem `docs/EXPERIMENTS.md`)

1. **Baseline**: `use_synopsis=True, synopsis_dim=48, synopsis_proj_hidden=[]` (MiniLM) vs control
   (`use_synopsis=False`) — đo warm (recall@200) + cold (anime mới chỉ có content). Bật/tắt 1 đòn bẩy.
2. Nếu có cải thiện → thử `synopsis_dim=64`, `synopsis_proj_hidden=[128]`, rồi swap `bge-small-en-v1.5`.
