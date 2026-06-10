# PROJECT_STRUCTURE.md

Doc kiến trúc sống: **cây thư mục + luồng hoạt động end-to-end** của recommender. File này giữ context tổng thể; chi tiết từng chặng nằm ở `docs/` và `CLAUDE.md` riêng trong mỗi mảng.

---

## 1. Overview

Anime recommender **2-stage**, build thành 3 mảng độc lập:

- **Retrieval** (`retriever/`) — Two-Tower (PyTorch) học user/item embedding, ANN top-K cosine từ ~22.8k anime. *HOÀN THÀNH pipeline (data + protocol + model + export); đang tune thêm epoch trên Colab — trạng thái: root `PROGRESS.md`.*
- **Ranking** (`ranker/`) — GBDT rerank top-K của retriever bằng feature chi tiết. *Code cũ — REBUILD TIẾP THEO trên artifacts mới (context: `PROGRESS.md`).*
- **Service** (`service/`) — web app: nhận user → retrieval → rerank → trả kết quả (+ Jikan/MAL cho metadata hiển thị). *Làm sau.*

**Chìa khoá song song hoá = `artifacts/` (firewall).** 3 mảng KHÔNG import code train của nhau; chỉ giao tiếp qua **file ổn định** trong `artifacts/` (retriever GHI; ranker + service ĐỌC). Vì checkpoint `retriever/checkpoints/best.pt` đã tồn tại, có thể export **artifacts provisional** để unblock ranker/service ngay cả khi retriever còn đang tune.

**Một venv chung** ở root (1 `requirements.txt` + 1 `venv/`): cho phép service import thẳng *định nghĩa model* của retriever (vd `UserTower`) — KHÔNG cần `common/` hay ONNX. Đây là rào *đóng gói* được gỡ; rào *dữ liệu* (`artifacts/`) vẫn giữ.

---

## 2. Cây thư mục

```
anime-recommender/
├── CLAUDE.md                 # guideline + context tổng (retriever/ và ranker/ có CLAUDE.md riêng)
├── PROGRESS.md               # ★ trạng thái + handoff build ranker — đọc đầu tiên
├── PROJECT_STRUCTURE.md      # file này
├── requirements.txt          # deps chung (torch + lightgbm + pytest…)
├── venv/                     # 1 venv chung — không đọc
│
├── data/                     # raw CSV + notebook cleaning — BLOCKED (xem CLAUDE.md §0)
├── cleaned-data/             # output cleaning — BLOCKED, input chung read-only
├── data-sample/              # 5 dòng đầu mỗi file — nguồn DUY NHẤT xem schema
├── data_audit/               # audit tầng data chung
├── docs/                     # CLEANING.md · DATA_SPLIT.md · TRAIN_DATA.md · TWO_TOWER_MODEL.md
├── legacy/                   # docs + kết quả cũ (gitignored, chỉ tham khảo)
│
├── artifacts/                # ★ FIREWALL: retriever GHI, ranker+service ĐỌC (xem §4)
│
├── retriever/                # Two-Tower retrieval — chi tiết: retriever/CLAUDE.md
│   ├── src/                  #   config/data/model/loss/metrics/train.py — import flat, CWD=src
│   ├── tests/                #   pytest invariants (padding/masking/OOV) — không cần train-data
│   ├── baselines/            #   _eval + rand/popular/meta_popular/content_based/itemknn/mf
│   ├── data_prep/            #   prep_config.py + 01..06 + 99_verify (cleaned-data → train-data)
│   ├── train-data/           #   artifacts prep (gitignored; cold_items + eval_seen + examples cold)
│   ├── checkpoints/          #   best.pt (gitignored; persist Drive/Colab)
│   ├── export.py             #   best.pt → artifacts/ (validation: test_export.py)
│   └── train.ipynb           #   notebook điều khiển train (Colab, VERSION v5)
│
├── ranker/                   # [REBUILD TIẾP THEO] GBDT reranker — code cũ, làm lại trên artifacts mới
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
4. **Export** — `best.pt` + `retriever/train-data/` → `retriever/export.py` → `artifacts/` (item vectors, user_tower, id map, user_split). Chạy lại mỗi khi `best.pt` đổi; validate bằng `retriever/test_export.py`.
5. **Train ranker** — `artifacts/` (cosine + vectors) + `cleaned-data/` (feature thô) → `ranker/` GBDT *(tiếp theo)* → `artifacts/ranker.txt`.
6. **Serve** — `service/backend` *(làm sau)* đọc `artifacts/` (vectors + user_tower + ranker + id map) → top-K cosine → rerank → ghép metadata Jikan/MAL → `service/frontend`.

---

## 4. `artifacts/` — contract (firewall)

| File | Ghi | Đọc | Nội dung |
|---|---|---|---|
| `item_vectors.npy` | retriever | ranker, service | `[N, 128]` L2-norm, 1 dòng/anime (row item cold = content-only, id→OOV) |
| `item_index.parquet` | retriever | ranker, service | row → `anime_idx` → MAL id (đúng thứ tự vector) + `is_cold` |
| `user_tower.pt` (+ spec) | retriever | service (, ranker) | encode `(history, gender, joined)` → U |
| `user_split.parquet` | retriever | ranker, eval | tập train/val/test user (DÙNG CHUNG — bắt buộc cho two-stage eval) |
| `ranker.txt` | **ranker** | service | model GBDT đã train |
| `CONTRACT.md` | retriever | tất cả | shape/dtype/version |

Nguồn artifact retriever lấy từ `retriever/train-data/`: `item_vectors` ← chạy item-tower trên `item_features.parquet` (row cold encode id→OOV); `item_index` ← `anime_id_map.parquet` + `cold_items.parquet`; `user_split` ← `_user_split.parquet`; `user_tower.pt` ← `checkpoints/best.pt` + `feature_spec.json`. (`retriever/export.py` làm việc này; `retriever/test_export.py` validate firewall-style.)

**Kỷ luật firewall:** ranker/service chỉ **import định nghĩa model** + **đọc `artifacts/`**; tuyệt đối không import code train của retriever.

---

## 5. Trạng thái & trình tự build song song

> Trạng thái chi tiết + số liệu + việc tiếp theo: **root `PROGRESS.md`** (luôn mới hơn file này).

- **Retriever** *(pipeline hoàn thành)*: data + protocol + model + baselines + export đều chạy và verify. Best hiện tại: `v5_hist64_ep2` (2 epoch) — đã export `artifacts/`; còn tune thêm epoch/knob trên Colab, mỗi lần có best.pt mới thì re-run export.
- **Ranker** *(TIẾP THEO)*: rebuild trên artifacts mới + protocol mới — context đầy đủ ở root `PROGRESS.md`. `artifacts/ranker.txt` cũ không còn giá trị.
- **Service** *(làm sau)*: backend đọc `artifacts/` + Jikan/MAL → 2 user-flow; frontend. Ban đầu dùng ranker stub, cuối swap ranker thật.

**Việc còn hoãn:** rebuild ranker, `service/`, two-stage eval harness.
