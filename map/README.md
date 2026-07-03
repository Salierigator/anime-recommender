# map/ — bản đồ trực quan không gian Two-Tower (PUMAP 2D)

Công cụ **phân tích offline** (KHÔNG phải serving): chiếu output item-tower 128-d (`artifacts/
item_vectors.npy`, đã L2-norm, cosine = khoảng cách góc) xuống **2D** bằng **Parametric UMAP**
để quan sát cụm anime theo khán giả, và đặt **vector user** ("you are here") + top-K gợi ý lên map.

Một method duy nhất: **`pumap2d`** (Parametric UMAP). Đặt điểm mới = forward-pass NN (`.transform`),
deploy-able. Sphere 3D + genre-probe (`encode_genre`) đã bỏ.

## CHỐT hiển thị (2026-07-03)
Sau khảo sát nhiều phương án (harness `explore_viz.py` + `explore_viz_v2.py`, ảnh + đánh giá ở
`outputs/explore/` và `outputs/explore/sweep_v2/` — `EVAL.md`, `EVAL_v2.md`):
- **Projection:** `pumap2d` **n_neighbors=50, min_dist=0.8** (mềm/tròn, lãnh thổ liền mạch).
- **Clustering:** KMeans **k=28** ở 128-d. HDBSCAN loại (không gian liên tục → 2 cụm + ~44% noise).
- **Naming:** **log-odds-ratio + prior Dirichlet** ('fightin words', `cluster.py::_logodds_names`) — tag đặc
  trưng nhất, ít lặp/nhòe hơn lift & tf-idf. Cột `examples` (3 phim phổ biến/cụm) sẵn cho hover frontend.
- **Render:** **kde_boundary** = `viz.py --style territory` (tô theo cụm áp đảo + đường biên trắng). hexbin
  đẹp nhưng **thiếu ý nghĩa** → loại. Điểm rời + hover + you-are-here vẫn ở `--style points` (Plotly HTML).
- **Serving (2026-07-03):** `export_service.py` → **`artifacts/map/`** (points/clusters/encoder-npz/
  territory nền/meta — CONTRACT riêng tự sinh trong đó) → web serve `GET /api/map` + you-are-here
  **không cần TF** (weights encoder trích qua h5py, forward numpy ≡ keras verified 0 diff — `service/CLAUDE.md §5`).

## Firewall
- Chỉ ĐỌC `artifacts/` + `cleaned-data/details.csv` (sạch). Không sửa `artifacts/` — **ngoại lệ duy
  nhất**: `export_service.py` GHI vào đúng namespace `artifacts/map/` (export cho service; schema +
  sync rule ở `artifacts/map/CONTRACT.md` tự sinh).
- Tái dùng ranker user-encoder (`ranker/src/{pool,user_encode,features}`) để encode user — **không**
  load `best.pt`, **không** import `retriever/src`, **không** đụng `train-data/`.
- Output ghi `map/outputs/` (gitignored).

## pumap CHẠY TRÊN COLAB (không phải local)
pumap cần **TensorFlow**. Venv local = **Python 3.9.6 / macOS 26.3.1 arm64** → `import tensorflow`
(2.20) **abort cứng** (`mutex lock failed`); `umap/__init__` còn eager-import parametric_umap → TF
nên cả `import umap` cũng deadlock. ⇒ **bước projection + encode_user chạy trên `run_colab.ipynb`**.

Phân công:
| Chạy ở đâu | Script |
|---|---|
| **Local** (pandas/sklearn/plotly/h5py, không cần TF/umap) | `build_base.py`, `cluster.py`, `viz.py` (render từ coords Colab tải về), `export_service.py` (→ `artifacts/map/`) |
| **Colab** (`run_colab.ipynb`, TF) | `project.py` (fit pumap), `encode_user.py` (load reducer + transform — chỉ còn cho khảo sát offline; serving đã có đường numpy) |

Deps: `scikit-learn` + `plotly` đã pin ở `requirements.txt` (local). `umap-learn` + `tensorflow`
= **Colab-only**, notebook tự `pip install umap-learn` (đừng cài local — sẽ deadlock).

## Chạy

### Trên Colab (`run_colab.ipynb`)
Đặt `artifacts/` + `cleaned-data/details.csv` vào `MyDrive/map_data/` (xem cell đầu), set `BRANCH`
đúng branch đã push, rồi chạy tuần tự các cell: base → project pumap2d → cluster → (encode_user nếu
điền USERNAME) → viz → HTML ghi về `MyDrive/map_data/outputs/`. Kéo HTML về local mở browser.

### Local (sau khi có coords từ Colab trong `outputs/`)
```bash
python map/build_base.py                                  # join vector + metadata (1 lần)
python map/cluster.py --algo kmeans --k 28                # CHỐT: k=28, cluster 128-d + naming log-odds
python map/viz.py --method pumap2d --cluster kmeans --style territory --suffix demo   # CHỐT: bản đồ territory PNG
python map/viz.py --method pumap2d --color cluster --cluster kmeans \
    --overlay overlay_user_me                             # HTML tương tác + you-are-here (tuỳ chọn)
python map/export_service.py                              # → artifacts/map/ cho web (GET /api/map)
```

## Pipeline
`build_base` (vector+metadata) → `project` (pumap2d → coords + reducer) → `cluster` (128-d → nhãn +
tên cụm) → `encode_user` (user point + top-K neighbor) → `viz` (HTML 2D) → `export_service`
(→ `artifacts/map/` cho web). Cluster làm ở **128-d**, tô màu chéo lên projection — KHÔNG cluster
trên tọa độ 2D. ⚠ Vectors đổi (retriever re-export) ⇒ chạy lại CẢ chuỗi; service tự phát hiện lệch
(sha `item_vectors` trong map_meta.json) và tắt map cho tới khi export lại.

## Nhật ký findings (không gian xếp theo KHÁN GIẢ, không theo genre)
Đo trên 128-d giải thích vì sao map "lùm nhùm": không gian two-tower xếp theo **khán giả (co-watch)**,
không theo nhãn genre:
- mean-cosine-tới-tâm: franchise (Pokemon/Gundam/Dragon Ball/Conan…) **0.75–0.90**, genre rộng
  (Action .46, Comedy .57, Drama .53) **≈ baseline ngẫu nhiên .30** → smear khắp map; genre
  1-khán-giả (Hentai .87, Avant Garde .84, Boys Love .80) chặt + lệch hẳn 1 góc.
- Cluster 128-d: HDBSCAN ra rất ít cụm tách bạch (không gian liên tục, ~44% noise) → KMeans **k=28**;
  mỗi cụm tự đặt **tên theo log-odds-ratio** (tag đặc trưng: Isekai·Reincarnation, Mystery·Workplace,
  Suspense·Psychological, Award Winning·Music, Adult Cast·Sci-Fi…). Sweep K + so lift/tf-idf/log-odds:
  `outputs/explore/sweep_v2/EVAL_v2.md`.
- `build_base` kéo `themes_list` (Mecha/Isekai/Music…) — tín hiệu khán giả mạnh hơn genre rộng.
- `build_base` **loại hentai** khỏi map (demo SFW-only, khớp nsfw serving: genre Hentai / rating
  `Rx - Hentai`) → không còn cụm Hentai/Hentai(2); k cụm dồn hết cho SFW. Bỏ vì 2D chiếu nó sát rìa
  lục địa dù 128-d rất xa (vd One Piece↔Hentai cosine −0.37) — artifact khó chịu, mà demo cũng lọc.

Demographic (shounen/seinen…) bị loại làm trục màu chính: 71% rỗng.

~~TODO: Phase 2 — export base map (JSON/binary) cho frontend WebGL + API "you are here".~~ **XONG
2026-07-03**: `export_service.py` + `GET /api/map` + `meta.map_xy` trong `/api/recommend` (shape:
`service/API_CONTRACT.md`). TODO còn lại: **frontend view map** (scatter WebGL + territory.png nền +
LOD theo `popularity` + marker user).
