# RECONSTRUCT_PLAN.md

> **Trạng thái: CHƯA THỰC THI.** File này là spec để tái cấu trúc thư mục project.
> Khi quay lại làm, đi tuần tự Phase 1 → 2 → 3, mỗi phase có verify-gate riêng.
> Quy tắc xuyên suốt (theo CLAUDE.md): dời tối thiểu, không "tiện tay" sửa code không liên quan,
> không đụng `data/` và `cleaned-data/`.

---

## 1. Mục tiêu

Tách 1 project phẳng hiện tại thành **3 mảng build độc lập** — `retriever/`, `ranker/`, `service/` — để
3 thứ có thể làm **song song** trong lúc còn đang tune two-tower trên Colab.

**Chìa khoá song song hoá = `artifacts/` (firewall).** Ranker và service KHÔNG phụ thuộc retriever
*train xong*, chỉ phụ thuộc **interface output ổn định** trong `artifacts/`. Vì checkpoint
`retriever/checkpoints/best.pt` đã tồn tại, ngay sau reconstruct ta có thể xuất **artifacts provisional**
→ unblock cả ranker lẫn service ngay.

---

## 2. Quyết định đã chốt

| # | Quyết định | Hệ quả |
|---|---|---|
| 1 | **Giữ `data-sample/`** | Vì `data/`+`cleaned-data/` bị block cứng, 5 dòng sample là cách DUY NHẤT được phép xem 1 row cụ thể (tên cột, cách serialize list, null/0). `refresh_samples.py` cũng giữ. |
| 2 | **Một venv chung ở root** | 1 `requirements.txt` + 1 `venv/`. Cho phép import chéo → service **import thẳng** định nghĩa model của retriever (vd `UserTower`), KHÔNG cần `common/` hay ONNX. |

**Lưu ý "venv chung":** chỉ gỡ *rào đóng gói*, KHÔNG gỡ *rào dữ liệu*. Vẫn giữ `artifacts/` làm contract
để 3 track build song song trên file ổn định. Kỷ luật: **ranker/service chỉ import *định nghĩa model* +
đọc `artifacts/`; tuyệt đối không import code train của retriever.**

---

## 3. Cây thư mục đích

```
anime-recommender/
├── data/                 # raw (blocked) — GIỮ ở root
├── cleaned-data/         # cleaned (blocked) — input chung, read-only — GIỮ ở root
├── data-sample/          # GIỮ — van an toàn xem schema
├── artifacts/            # ★ FIREWALL: retriever GHI, ranker+service ĐỌC  (tạo rỗng)
├── docs/                 # overview + CLEANING.md + (TRAIN_DATA/MODEL tạm để đây — xem §8)
├── data_audit/           # ← scripts/data_audit/  (giữ nguyên codes/ + output/) + refresh_samples.py
│   ├── codes/            #   {details,profiles,ratings}_audit/*.py  (details ~24, profiles 2, ratings 2)
│   ├── output/           #   {details,profiles,ratings}_audit/*.txt (kết quả audit)
│   └── refresh_samples.py  # ← scripts/refresh_samples.py
│   #  (giữ tên underscore theo bạn; muốn đồng bộ hyphen như data-sample/ thì đổi → data-audit/)
├── requirements.txt      # 1 bộ chung (torch + lightgbm + fastapi…)
├── venv/                 # 1 venv chung — GIỮ ở root
├── CLAUDE.md
├── RECONSTRUCT_PLAN.md   # file này
│
├── retriever/
│   ├── src/              # ← model/*.py  (config, data, model, loss, metrics, train, __init__)
│   ├── baselines/        # ← model/baselines/  (_eval, content_based, itemknn, mf, popular, rand + .txt)
│   ├── data_prep/        # ← scripts/build_train_data/  (01..06 + 99_verify)
│   ├── train-data/       # ← train-data/  (artifact NỘI BỘ retriever; gitignored)
│   ├── checkpoints/      # ← model/checkpoints/  (best.pt; gitignored)
│   ├── train.ipynb       # ← model/train.ipynb
│   ├── export.py         # [LÀM SAU] best.pt → artifacts/   (đặt trong src/ vì import model/data)
│   └── docs/             # [TẠO SAU khi chẻ docs — §8]
│
├── ranker/
│   ├── src/              # [LÀM SAU] GBDT
│   └── docs/             # [TẠO SAU]
│
└── service/
    ├── backend/          # [LÀM SAU] FastAPI: top-K cosine + ranker + Jikan/MAL
    ├── frontend/         # [LÀM SAU]
    └── docs/             # [TẠO SAU]
```

---

## 4. Mapping nguồn → đích

| Hiện tại | Đích | Cách dời | Ghi chú |
|---|---|---|---|
| `model/{config,data,model,loss,metrics,train}.py`, `model/__init__.py` | `retriever/src/` | `git mv` | tracked |
| `model/baselines/` (`_eval`, content_based, itemknn, mf, popular, rand + `.txt`) | `retriever/baselines/` | **`mv`** thường | **untracked** (bạn vừa tạo) — giữ nguyên nhóm |
| `model/train.ipynb` | `retriever/train.ipynb` | `git mv` | tracked |
| `model/checkpoints/` | `retriever/checkpoints/` | `mv` thường | **gitignored** |
| `scripts/build_train_data/` | `retriever/data_prep/` | `git mv` | tracked |
| `train-data/` | `retriever/train-data/` | `mv` thường | **gitignored** |
| `scripts/data_audit/` (`codes/` + `output/`) | `data_audit/` | **`mv`** thường | **untracked** (bạn vừa tạo) — giữ nguyên codes/+output/ |
| `scripts/refresh_samples.py` | `data_audit/refresh_samples.py` | `git mv` | tracked |
| `scripts/` (rỗng sau dời) | xoá | `rmdir scripts` | |
| `data/`, `cleaned-data/`, `data-sample/`, `venv/`, `docs/` | **không đổi** | — | giữ ở root |

---

## 5. `artifacts/` — contract (firewall)

| File | Ghi | Đọc | Nội dung |
|---|---|---|---|
| `item_vectors.npy` | retriever | ranker, service | `[N, 128]` L2-norm, 1 dòng/anime |
| `item_index.parquet` | retriever | ranker, service | row → `anime_idx` → MAL id (đúng thứ tự vector) |
| `user_tower.pt` (+ spec) | retriever | service (, ranker) | encode `(history, gender, joined)` → U |
| `user_split.parquet` | retriever | ranker, eval | tập train/val/test user (DÙNG CHUNG — bắt buộc cho two-stage eval) |
| `ranker.txt` | **ranker** | service | model GBDT đã train |
| `CONTRACT.md` | retriever | tất cả | shape/dtype/version |

Nguồn của các artifact retriever lấy từ `retriever/train-data/`:
`item_vectors` ← chạy item-tower trên `item_features.parquet`; `item_index` ← `anime_id_map.parquet`;
`user_split` ← `_user_split.parquet`; `user_tower.pt` ← `checkpoints/best.pt` + `feature_spec.json`.
(`export.py` làm việc này — **viết sau**.)

---

## 6. Hai điểm kỹ thuật bắt buộc nhớ khi vá path (Phase 2)

**(A) Import trong `model/` là FLAT** — vd `import config as cfg_mod`, `import data as data_mod`,
`from model import TwoTower`. Tức code chạy với **CWD = thư mục chứa module**, không phải dạng package.
Hệ quả:
- **Không cần đổi dòng import** nếu giữ quy ước "chạy từ trong `retriever/src/`".
- **export.py phải nằm CÙNG `retriever/src/`** (vì cũng import flat `config`/`data`).
- **Baselines** bạn đã gom vào folder riêng (`model/baselines/`, có `_eval.py` dùng chung) → target
  `retriever/baselines/`. Vì khác thư mục `src/`, import của chúng cần `sys.path`/đổi import — việc này
  thuộc **phần vá path đã hoãn** (theo yêu cầu hiện tại).

**(B) `data_prep/01–06` đọc INPUT ở root nhưng GHI OUTPUT vào retriever** — mỗi script hiện có:
`SRC = ROOT / "cleaned-data" / ...` và `OUT = ROOT / "train-data"`. Sau khi dời:
- `cleaned-data/` vẫn ở **repo root**.
- `train-data/` chuyển vào **`retriever/train-data/`**.
→ Một biến `ROOT` không còn đúng cho cả hai. Phải tách **2 mỏ neo**: `REPO_ROOT` (cho `cleaned-data`) và
`RETRIEVER` (cho `train-data`). Đây là chỗ dễ sai nhất.

**Phụ:** `config.py` có `TRAIN_DATA = ROOT / "train-data"` (dùng qua `cfg.train_data`). Sau khi `config.py`
về `retriever/src/`, chỉnh `ROOT`/`TRAIN_DATA` để trỏ đúng `retriever/train-data/` (kiểm tra
`Path(__file__).parents[N]`). Đồng thời rà mọi path khác phái sinh từ `ROOT` trong `config.py` (vd
checkpoints/runs nếu có).

**Audits & refresh_samples (path nội bộ — HOÃN):** promote `scripts/data_audit/` → root `data_audit/` (giữ
`codes/`+`output/`, sâu thêm 1 cấp) làm `ROOT` trong các script audit + `refresh_samples.py` lệch độ sâu →
phải rà lại `Path(__file__).parents[N]` (`cleaned-data/`+`data-sample/` vẫn ở root). Theo yêu cầu hiện tại,
**để lại cho phần vá path sau**.

---

## 7. Phases thực thi

### Phase 1 — Dời file (không đổi hành vi)

```bash
# tạo khung
mkdir -p retriever/src artifacts ranker/src service/backend service/frontend

# code retriever core (tracked → git mv)
git mv model/config.py model/data.py model/model.py model/loss.py \
       model/metrics.py model/train.py model/__init__.py retriever/src/
git mv model/train.ipynb retriever/train.ipynb
git mv scripts/build_train_data retriever/data_prep      # để git tự tạo target (đừng mkdir trước)

# UNTRACKED (bạn vừa tạo, git chưa track) → mv thường, KHÔNG git mv
mv model/baselines retriever/baselines
mv scripts/data_audit data_audit                         # giữ nguyên codes/ + output/

# refresh_samples (tracked) → git mv vào data_audit/ vừa promote
git mv scripts/refresh_samples.py data_audit/refresh_samples.py
rmdir scripts                                            # rỗng sau khi dời

# gitignored → mv thường (KHÔNG git mv)
mv train-data retriever/train-data
mv model/checkpoints retriever/checkpoints
rmdir model 2>/dev/null || true                          # model/ rỗng sau khi dời hết
```
> ⚠️ `git mv scripts/build_train_data retriever/data_prep` — nếu `retriever/data_prep/` đã `mkdir` sẵn,
> git có thể dồn vào `retriever/data_prep/build_train_data/`. Cách an toàn: **đừng** mkdir `data_prep`
> trước, để `git mv` tự tạo; hoặc `git mv scripts/build_train_data/* retriever/data_prep/`.

**Sửa `.gitignore`:**
- `model/checkpoints/` → `retriever/checkpoints/`  **(bắt buộc** — pattern có slash giữa, neo theo root).
- `train-data/` → có thể để nguyên (trailing-slash, match mọi cấp nên vẫn ignore `retriever/train-data/`),
  hoặc đổi `retriever/train-data/` cho rõ ràng.
- `data/`, `cleaned-data/`, `venv/` — **không đổi**.

**Verify-gate Phase 1:** `git status` chỉ thấy rename (không mất file); `data/`+`cleaned-data/` nguyên vị trí.

### Phase 2 — Vá path (phần cần cẩn thận nhất)

1. `retriever/src/config.py`: chỉnh `ROOT`/`TRAIN_DATA` → `retriever/train-data/` (rà cả path khác từ `ROOT`).
2. `retriever/data_prep/01..06_*.py`: tách **2 mỏ neo** — `cleaned-data` ở repo-root, `train-data` ở retriever
   (xem §6-B). `99_verify.py` chỉ đọc `train-data` → chỉ cần trỏ đúng `retriever/train-data/`.
3. `data_audit/refresh_samples.py`: chỉnh `ROOT` theo vị trí mới (cleaned-data + data-sample vẫn ở root).
4. `data_audit/codes/*_audit/*.py`: giờ sâu thêm (`data_audit/codes/...`) → `ROOT = parents[N]` lệch, rà lại N
   (cleaned-data vẫn ở root). File `.txt` trong `data_audit/output/` không phải sửa.
5. `retriever/train.ipynb`: sửa bootstrap Colab (mount Drive `train-data` → `retriever/train-data`; chỗ
   `{CODE}/train-data`; thư mục chạy code). Đây là glue Colab, không phải logic.

**Verify-gate Phase 2 (chạy local, KHÔNG đụng data lớn):**
```bash
# smoke retriever (CWD = src vì import flat)
cd retriever/src && ../../venv/bin/python train.py --smoke && cd ../..
# verify artifacts train-data (chỉ đọc train-data)
venv/bin/python retriever/data_prep/99_verify.py
# 1 audit chạm cleaned-data (xác nhận path root còn đúng)
venv/bin/python data_audit/codes/details_audit/audit_type.py
# refresh sample (xác nhận ghi data-sample/ đúng)
venv/bin/python data_audit/refresh_samples.py
```
Cả 4 phải chạy không lỗi mới coi là Phase 2 xong.

### Phase 3 — Cập nhật `CLAUDE.md`

- Viết lại §0.5 (cây project) theo §3 ở trên.
- Cập nhật mọi path lệnh mẫu (vd smoke-test `model/train.py` → `retriever/src/train.py`; vị trí
  `scripts/build_train_data` → `retriever/data_prep`; `scripts/data_audit/` → `data_audit/`).
- **GIỮ NGUYÊN**: §0 (block `data/`+`cleaned-data/`) và quy tắc data-sample (vì đã quyết giữ).

**Verify-gate Phase 3:** đọc lại CLAUDE.md, mọi path nêu trong đó tồn tại thật trên cây mới.

---

## 8. Hoãn lại — KHÔNG làm trong reconstruct này

- **Chẻ docs**: TRAIN_DATA.md + MODEL.md → `retriever/docs/`, viết `ranker/docs`, `service/docs`,
  `docs/CONTRACT.md`. Theo yêu cầu, **để 3 .md hiện tại nguyên trong `docs/` root**; reorg docs là **bước
  riêng sau khi cấu trúc đã ổn**. (Vì vậy các `*/docs/` để trống lúc này.)
- **`retriever/export.py`** (code thật): best.pt → `artifacts/`.
- **`ranker/`** (GBDT) và **`service/`** (backend/frontend) — code thật.
- **Two-stage eval** harness (ghép retriever+ranker) — đặt trong `ranker/` hoặc `eval/` ở root, quyết sau.

---

## 9. Sau reconstruct: trình tự build song song

1. **Step 0 — chốt contract + provisional artifacts**: viết `docs/CONTRACT.md` + `retriever/export.py`,
   chạy export từ `best.pt` → `artifacts/` (provisional). **Đây là việc unblock mọi thứ.**
2. **Song song từ đây:**
   - *Retriever track*: tiếp tục tune trên Colab; xong chạy lại `export.py` ghi đè `artifacts/`.
   - *Ranker track*: feature pipeline (raw từ `cleaned-data/` + cosine từ `artifacts/`) → train GBDT →
     ghi `artifacts/ranker.txt`. Retriever chốt → trỏ lại artifacts mới + retrain (CPU, vài phút).
   - *Service track*: backend đọc `artifacts/` (vectors, user_tower, ranker, id map) + Jikan/MAL → 2 user
     flow; frontend. Ban đầu dùng ranker stub, cuối swap vào ranker thật.

---

## 10. Rủi ro & lưu ý

- **Gotcha #1 (2 mỏ neo §6-B):** `data_prep` dễ sai nhất — verify bằng `99_verify.py`.
- **Gotcha #2 (import flat §6-A):** giữ baselines/export trong `retriever/src/`, chạy với CWD=src.
- **`mv` thường vs `git mv`:** dùng `mv` thường cho (a) **gitignored** — `train-data/`, `model/checkpoints/`,
  `venv/`, `data/`, `cleaned-data/`; và (b) **untracked bạn vừa tạo** — `model/baselines/`, `scripts/data_audit/`.
  Phần **tracked** còn lại (`model/*.py`, `train.ipynb`, `build_train_data/`, `refresh_samples.py`) dùng `git mv` giữ history.
- **Rollback:** `git mv` → `git reset`/`git mv` ngược; phần `mv` thường → `mv` về chỗ cũ. An toàn vì
  Phase 1 không sửa nội dung file.
- **Không đụng `data/`+`cleaned-data/`** trong toàn bộ quá trình (chỉ dời, không đọc).
```
