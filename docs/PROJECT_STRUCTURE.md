# PROJECT_STRUCTURE.md

Doc kiến trúc sống: **cây thư mục + luồng hoạt động end-to-end** của recommender. File này giữ context tổng thể; chi tiết từng chặng nằm ở `docs/` và `CLAUDE.md` riêng trong mỗi mảng.

---

## 1. Overview

Anime recommender **2-stage**, build thành 3 mảng độc lập:

- **Retrieval** (`retriever/`) — Two-Tower (PyTorch) học user/item embedding, ANN top-K cosine từ ~22.8k anime. *Đang REBUILD v2 (data + protocol) — trạng thái: root `PROGRESS.md`.*
- **Ranking** (`ranker/`) — GBDT rerank top-K của retriever bằng feature chi tiết. *Có bản v1 — STALE, retrain sau khi retriever v2 chốt.*
- **Service** (`service/`) — web app: nhận user → retrieval → rerank → trả kết quả (+ Jikan/MAL cho metadata hiển thị). *Làm sau.*

**Chìa khoá song song hoá = `artifacts/` (firewall).** 3 mảng KHÔNG import code train của nhau; chỉ giao tiếp qua **file ổn định** trong `artifacts/` (retriever GHI; ranker + service ĐỌC). Vì checkpoint `retriever/checkpoints/best.pt` đã tồn tại, có thể export **artifacts provisional** để unblock ranker/service ngay cả khi retriever còn đang tune.

**Một venv chung** ở root (1 `requirements.txt` + 1 `venv/`): cho phép service import thẳng *định nghĩa model* của retriever (vd `UserTower`) — KHÔNG cần `common/` hay ONNX. Đây là rào *đóng gói* được gỡ; rào *dữ liệu* (`artifacts/`) vẫn giữ.

---

## 2. Cây thư mục

```
anime-recommender/
├── CLAUDE.md                 # guideline + context tổng (retriever/ và ranker/ có CLAUDE.md riêng)
├── PROGRESS.md               # ★ trạng thái rebuild v2 + số liệu + resume — đọc đầu tiên
├── PROJECT_STRUCTURE.md      # file này
├── requirements.txt          # deps chung (torch + lightgbm + pytest…)
├── venv/                     # 1 venv chung — không đọc
│
├── data/                     # raw CSV + notebook cleaning — BLOCKED (xem CLAUDE.md §0)
├── cleaned-data/             # output cleaning — BLOCKED, input chung read-only
├── data-sample/              # 5 dòng đầu mỗi file — nguồn DUY NHẤT xem schema
├── data_audit/               # audit tầng data chung
├── docs/                     # CLEANING.md · TRAIN_DATA.md (v2) · TWO_TOWER_MODEL.md (v2) · RANKER.md (stale)
├── legacy/                   # docs + kết quả protocol v1 (gitignored, chỉ tham khảo)
│
├── artifacts/                # ★ FIREWALL: retriever GHI, ranker+service ĐỌC (xem §4) — v1-STALE
│
├── retriever/                # [REBUILD v2] Two-Tower retrieval — chi tiết: retriever/CLAUDE.md
│   ├── src/                  #   config/data/model/loss/metrics/train.py — import flat, CWD=src
│   ├── tests/                #   pytest invariants (padding/masking/OOV) — không cần train-data
│   ├── baselines/            #   _eval + rand/popular/meta_popular/content_based/itemknn/mf
│   ├── data_prep/            #   prep_config.py + 01..06 + 99_verify (cleaned-data → train-data)
│   ├── train-data/           #   artifacts v2 (gitignored; có cold_items + eval_seen + examples cold)
│   ├── checkpoints/          #   best.pt (gitignored; persist Drive/Colab)
│   ├── export.py             #   ⚠ v1-STALE — sửa theo schema v2 sau khi retriever chốt
│   └── train.ipynb           #   notebook điều khiển train (Colab, VERSION v5)
│
├── ranker/                   # [v1-STALE] GBDT reranker — retrain sau khi retriever v2 chốt
│   └── src/
│
└── service/                  # [LÀM SAU] web app
    ├── backend/              #   FastAPI: top-K cosine + ranker + Jikan/MAL
    └── frontend/
```

---

## 3. Luồng end-to-end

Chuỗi từ raw data tới serving; mỗi chặng trỏ doc chi tiết:

1. **Cleaning** — `data/` (raw, blocked) → notebook cleaning trong `data/` (doc `docs/CLEANING.md`) → `cleaned-data/`. Drop bot/orphan, k-core, chuẩn hoá schema.
2. **Train-data** — `cleaned-data/` → `retriever/data_prep/01..06` (+ `99_verify`, doc `docs/TRAIN_DATA.md`) → `retriever/train-data/`. Re-index id, encode feature, split cold-by-user, history/hard-neg, logQ.
3. **Train retriever** — `retriever/train-data/` → `retriever/src/` (Two-Tower, doc `docs/TWO_TOWER_MODEL.md`) → `retriever/checkpoints/best.pt`. InfoNCE + logQ + hard-neg, eval cold-by-user. Train thật trên Colab GPU.
4. **Export** — `best.pt` + `retriever/train-data/` → `retriever/export.py` *(làm sau)* → `artifacts/` (item vectors, user_tower, id map, user_split).
5. **Train ranker** — `artifacts/` (cosine + vectors) + `cleaned-data/` (feature thô) → `ranker/` GBDT *(làm sau)* → `artifacts/ranker.txt`.
6. **Serve** — `service/backend` *(làm sau)* đọc `artifacts/` (vectors + user_tower + ranker + id map) → top-K cosine → rerank → ghép metadata Jikan/MAL → `service/frontend`.

---

## 4. `artifacts/` — contract (firewall)

| File | Ghi | Đọc | Nội dung |
|---|---|---|---|
| `item_vectors.npy` | retriever | ranker, service | `[N, 128]` L2-norm, 1 dòng/anime |
| `item_index.parquet` | retriever | ranker, service | row → `anime_idx` → MAL id (đúng thứ tự vector) |
| `user_tower.pt` (+ spec) | retriever | service (, ranker) | encode `(history, gender, joined)` → U |
| `user_split.parquet` | retriever | ranker, eval | tập train/val/test user (DÙNG CHUNG — bắt buộc cho two-stage eval) |
| `ranker.txt` | **ranker** | service | model GBDT đã train |
| `CONTRACT.md` | retriever | tất cả | shape/dtype/version |

Nguồn artifact retriever lấy từ `retriever/train-data/`: `item_vectors` ← chạy item-tower trên `item_features.parquet`; `item_index` ← `anime_id_map.parquet`; `user_split` ← `_user_split.parquet`; `user_tower.pt` ← `checkpoints/best.pt` + `feature_spec.json`. (`export.py` làm việc này — viết sau.)

**Kỷ luật firewall:** ranker/service chỉ **import định nghĩa model** + **đọc `artifacts/`**; tuyệt đối không import code train của retriever.

---

## 5. Trạng thái & trình tự build song song

> Trạng thái chi tiết + số liệu + việc tiếp theo: **root `PROGRESS.md`** (luôn mới hơn file này).

- **Retriever** *(REBUILD v2 — đang chạy)*: train-data v2 đã build + verify (labels mới, cold-item H, history full, eval_seen); src v2 + tests xong; baselines v2 đo lại. Tiếp: upload train-data v2 lên Drive, train v5 trên Colab.
- **Ranker** *(v1-stale)*: artifacts + `ranker.txt` hiện theo protocol/checkpoint v1. Sau khi retriever v2 chốt: sửa `export.py` theo schema v2 (nếu dùng `history_source='embed'` phải đóng gói thêm `hist_emb`), export lại, retrain ranker (CPU, vài phút), đo lại two-stage.
- **Service** *(làm sau)*: backend đọc `artifacts/` + Jikan/MAL → 2 user-flow; frontend. Ban đầu dùng ranker stub, cuối swap ranker thật.

**Việc còn hoãn:** export v2, retrain ranker, `service/`, two-stage eval harness theo protocol v2.
