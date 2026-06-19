# PROJECT_STRUCTURE.md

Doc kiến trúc sống: **cây thư mục + luồng hoạt động end-to-end** của recommender. File này giữ context tổng thể; chi tiết từng chặng nằm ở `docs/` và `CLAUDE.md` riêng trong mỗi mảng.

> Trạng thái **2026-06-18**: pipeline **đồng bộ hoàn toàn** trên config/pool `final`. Retriever
> chốt `final` (no synopsis, 2026-06-17, serve-path official [RESULTS.md §3b](RESULTS.md)); ranker
> chốt `lrank_t20_gainLin` trên pool `final` (2026-06-18, [RESULTS.md §6](RESULTS.md)). Số liệu
> tổng hợp + trạng thái mới nhất: root `PROGRESS.md`. Kiến trúc/protocol mô tả trong file ổn định.

---

## 1. Overview

Anime recommender **2-stage**, build thành 3 mảng độc lập:

- **Retrieval** (`retriever/`) — Two-Tower (PyTorch) học user/item embedding chung 1 không gian, ANN top-K cosine từ ~22.8k anime. *✅ HOÀN THÀNH pipeline (data + protocol + model + baselines + export); còn tune thêm epoch/knob trên Colab — không chặn gì.*
- **Ranking** (`ranker/`) — LightGBM rerank top-200 của retriever bằng 29 feature chi tiết. *✅ CHỐT 2026-06-18 (pool `final`): `lrank_t20_gainLin` (lambdarank t20), blend α=1, test ndcg@10 .5323 → **.7231** (vượt MF ndcg-opt .7027), liked_ndcg@10 .5615; item cold tách kênh serve — chi tiết `docs/RANKER.md`.*
- **Service** (`service/`) — web app: nhận user → retrieval → rerank → trả kết quả (+ Jikan/MAL cho metadata hiển thị). *🟨 Backend CLI XONG (`service/backend/recommend.py`, 3 nguồn user); FastAPI + frontend chưa.*

**Chìa khoá song song hoá = `artifacts/` (firewall).** 3 mảng KHÔNG import code train của nhau; chỉ giao tiếp qua **file ổn định** trong `artifacts/` (retriever GHI; ranker GHI riêng `ranker.txt`+`ranker_meta.json`; service ĐỌC tất cả). Khi retriever có best.pt mới → re-export → ranker retrain theo loop `docs/RANKER.md §9` — service không phải đổi code.

**Một venv chung** ở root (1 `requirements.txt` + 1 `venv/`): cho phép service import thẳng *định nghĩa model* của retriever (vd `UserTower`) — KHÔNG cần `common/` hay ONNX. Đây là rào *đóng gói* được gỡ; rào *dữ liệu* (`artifacts/`) vẫn giữ.

---

## 2. Cây thư mục

```
anime-recommender/
├── CLAUDE.md                 # guideline + context tổng (retriever/ranker/service có CLAUDE.md riêng)
├── PROGRESS.md               # ★ trạng thái + số liệu mới nhất — đọc đầu tiên
├── AGENT.md                  # guideline cho agent build frontend (mảng service)
├── requirements.txt          # deps chung (torch + lightgbm + implicit + pytest…)
├── venv/                     # 1 venv chung — không đọc
│
├── data/                     # raw CSV + notebook cleaning — BLOCKED (xem CLAUDE.md §0)
├── cleaned-data/             # output cleaning (details/profiles/ratings.csv) — BLOCKED, input read-only
├── data-sample/              # 5 dòng đầu mỗi file — nguồn DUY NHẤT xem schema
├── data_audit/               # audit tầng data chung (script + output per-column)
├── docs/                     # CLEANING · DATA_DISTRIBUTIONS · DATA_SPLIT · TRAIN_DATA · TWO_TOWER_MODEL · SYNOPSIS_EMB · EXPERIMENTS · LIKED_METRIC · BASELINES · RANKER · RANKER_EXPERIMENTS · RESULTS · file này
├── legacy/                   # docs + kết quả cũ (gitignored, chỉ tham khảo)
│
├── artifacts/                # ★ FIREWALL giữa 3 mảng (xem §4 + artifacts/CONTRACT.md tự sinh)
│
├── retriever/                # Two-Tower retrieval — chi tiết: retriever/CLAUDE.md
│   ├── data_prep/            #   prep_config.py + 01..06 + 99_verify (cleaned-data → train-data)
│   ├── src/                  #   config/data/model/loss/metrics/train.py — import flat, CWD=src
│   ├── tests/                #   pytest invariants (collate/loss/metrics/model) — không cần train-data
│   ├── baselines/            #   _eval.py + rand/popular/meta_popular/content_based/itemknn/mf (+ *.txt kết quả)
│   ├── train-data/           #   artifacts prep (gitignored)
│   ├── checkpoints/          #   best.pt (gitignored; tải từ Drive runs/)
│   ├── export.py             #   best.pt → artifacts/ (validate + eval_reference: test_export.py)
│   ├── test_export.py
│   └── train.ipynb           #   notebook điều khiển train (Colab, VERSION v5)
│
├── ranker/                   # LightGBM rerank stage 2 — chi tiết: ranker/CLAUDE.md + docs/RANKER.md
│   ├── data_prep/            #   build_eval.py (pools depth 500) + build_train.py (datasets K=200)
│   ├── src/                  #   config/pool/features/metrics/train_lgbm/train_nn/user_encode.py
│   ├── baselines/            #   baseline_linear.py (logistic — GBDT phải thắng nó)
│   ├── tests/                #   pytest (features/metrics/no_leak)
│   ├── train-data/           #   pools + datasets (gitignored)
│   ├── models/               #   <run>/{model, row.json, results.txt} + eval_selection.json (gitignored)
│   ├── eval.py               #   sanity gate + blend sweep + Pareto select + test report
│   ├── export.py             #   winner → artifacts/ranker.txt + ranker_meta.json
│   ├── report_models.py      #   models/<run>/results.txt (per-model, chỉ VAL)
│   └── train.ipynb           #   notebook Colab (NN DIN only — LightGBM sweep train LOCAL: src/train_lgbm.py)
│
└── service/                  # web app — chi tiết: service/CLAUDE.md
    ├── backend/              #   recommend.py (CLI XONG — 3 nguồn user, 2 section warm/cold)
    │                         #   + mal_api.py (MAL v2 + Jikan v4) + cache/ (gitignored)
    └── frontend/             #   CHƯA — làm cùng FastAPI wrap
```

---

## 3. Luồng end-to-end

Chuỗi từ raw data tới serving; mỗi chặng trỏ doc chi tiết:

1. **Cleaning** — `data/` (raw, blocked) → notebook cleaning trong `data/` (doc `docs/CLEANING.md`) → `cleaned-data/`. Drop bot/orphan, k-core, chuẩn hoá schema. *Đã chốt, không chạy lại.*
2. **Train-data** — `cleaned-data/` → `retriever/data_prep/01..06` (+ `99_verify`, doc `docs/TRAIN_DATA.md`) → `retriever/train-data/`. Re-index id, encode feature, split 2 trục (cold-user 90/5/5 + cold-item H), history FULL, hard-neg, logQ.
3. **Train retriever** — `retriever/train-data/` → `retriever/src/` (Two-Tower, doc `docs/TWO_TOWER_MODEL.md`) → `retriever/checkpoints/best.pt`. InfoNCE + logQ + hard-neg; tuning trên warm val, cold đo slice riêng (id→OOV). Train thật trên Colab GPU. Baselines so sánh: `docs/BASELINES.md`.
4. **Export** — `best.pt` + `retriever/train-data/` → `retriever/export.py` → `artifacts/` (item vectors, user_tower, id map, user_split, users_history, eval_seen, eval_queries). Chạy lại mỗi khi `best.pt` đổi; `retriever/test_export.py` validate firewall-style + ghi `eval_reference.json` (mốc cho sanity gate ranker).
5. **Train ranker** — `artifacts/` (pool top-200 cosine + vectors + history) + `cleaned-data/` (feature thô item/user) → `ranker/` (doc `docs/RANKER.md`): build_eval → sanity gate → build_train → Colab sweep → eval.py select → `artifacts/ranker.txt` + `ranker_meta.json`.
6. **Serve** — `service/backend/recommend.py` đọc `artifacts/` (vectors + user_tower + ranker + meta): dựng U → top-K cosine (mask seen) → **tách warm/cold theo `is_cold`** → warm: 29 feature + LightGBM rerank (α=1) = main list; cold: section riêng sort cosine (đúng `ranker_meta.json::cold_serving`) → ghép metadata Jikan/MAL. FastAPI + frontend: chưa.

---

## 4. `artifacts/` — contract (firewall)

Schema chi tiết + version: **`artifacts/CONTRACT.md`** (tự sinh bởi export, không sửa tay). Tóm tắt:

| File | Ghi | Đọc | Nội dung |
|---|---|---|---|
| `item_vectors.npy` | retriever | ranker, service | `[N, 128]` float32 L2-norm, row == anime_idx (row cold encode id→OOV, content-only) |
| `item_index.parquet` | retriever | ranker, service | `anime_idx` → `mal_id` (−1 cho PAD/OOV) + `is_cold` |
| `user_tower.pt` | retriever | ranker, service | user-side state_dict + pooling cfg + user_features spec → encode `(history, gender, joined)` → U |
| `user_split.parquet` | retriever | ranker | `username, user_idx, split` ∈ {train,val,test} — bắt buộc cho two-stage eval |
| `users_history.parquet` | retriever | ranker, service | MỌI user: gender/joined + history_ids/scores FULL + hard_neg_ids — ranker dựng U + train data từ đây, KHÔNG stream ratings.csv |
| `eval_seen.parquet` | retriever | ranker | eval user → seen_ids (mọi status) — nguồn duy nhất cho seen-mask |
| `eval_queries_{val,test,val_cold}.parquet` | retriever | ranker | positive held-out (query) per slice — two-stage chấm ĐÚNG protocol retriever |
| `eval_queries_test_cold.parquet` | retriever | ranker | **final exam** — CHỈ tồn tại khi `export.py --final-exam`; default export xoá |
| `eval_reference.json` | retriever (`test_export.py`) | ranker | metrics cosine baseline đo QUA artifacts — sanity gate `ranker/eval.py --baseline-only` phải tái lập trước khi train |
| `ranker.txt` | **ranker** | service | model LightGBM production (hiện: `lrank_t20_gainLin`) |
| `ranker_meta.json` | **ranker** | service | feature_names (29, đúng thứ tự), blend_alpha, k_retrieve, **cold_serving**, grading, metrics, provenance |
| `CONTRACT.md` | retriever | tất cả | shape/dtype/version + checkpoint metrics (tự sinh) |

**Kỷ luật firewall:** ranker/service chỉ **import định nghĩa model** (vd `UserTower`) + **đọc `artifacts/`**; tuyệt đối không import code train của retriever. Ranker là mảng duy nhất ngoài retriever được GHI (2 file của nó).

---

## 5. Trạng thái & việc còn mở

> Trạng thái chi tiết + số liệu + việc tiếp theo: **root `PROGRESS.md`** (luôn mới hơn file này).

- **Retriever** *(✅ pipeline hoàn thành)*: data + protocol + model + baselines + export đều chạy và verify. **Chốt config `final` (no synopsis, 2026-06-17)** — synopsis test on/off bị bác vì regress cold (`docs/SYNOPSIS_EMB.md`). ✅ **re-export xong** (`best.pt`/`artifacts/` = `final`, serve-path official `docs/RESULTS.md §3b`); ranker đã retrain trên pool `final` → pipeline đồng bộ.
- **Ranker** *(✅ RE-CHỐT 2026-06-18 pool `final`)*: `artifacts/ranker.txt` = production (`lrank_t20_gainLin` lambdarank t20, α=1, test ndcg@10 **.7231** / liked_ndcg@10 .5615, vượt MF ndcg-opt mọi head+mid — `docs/RESULTS.md §6`); cold tách kênh serve. ✅ test_cold final exam đã chấm 1 lần (2026-06-18, cosine cold ndcg@10 .1397 — `docs/RESULTS.md §7`). Còn mở (không chặn): ablation K=500.
- **Service** *(🟨 backend CLI xong)*: `recommend.py` serve đúng `cold_serving` (main list rerank warm + section "Anime mới" cosine), 3 nguồn user (dataset / live MAL / file). **Còn lại: FastAPI wrap + frontend** (`service/CLAUDE.md §5`).
