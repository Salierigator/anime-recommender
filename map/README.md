# map/ — bản đồ trực quan không gian Two-Tower (PUMAP 2D)

Công cụ **phân tích offline** (KHÔNG phải serving): chiếu output item-tower 128-d (`artifacts/
item_vectors.npy`, đã L2-norm, cosine = khoảng cách góc) xuống **2D** bằng **Parametric UMAP**
để quan sát cụm anime theo khán giả, và đặt **vector user** ("you are here") + top-K gợi ý lên map.

Một method duy nhất: **`pumap2d`** (Parametric UMAP). Đặt điểm mới = forward-pass NN (`.transform`),
deploy-able. Sphere 3D + genre-probe (`encode_genre`) đã bỏ.

## Firewall
- Chỉ ĐỌC `artifacts/` + `cleaned-data/details.csv` (sạch). Không sửa `artifacts/`.
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
| **Local** (pandas/sklearn/plotly, không cần TF/umap) | `build_base.py`, `cluster.py`, `viz.py` (render từ coords Colab tải về) |
| **Colab** (`run_colab.ipynb`, TF) | `project.py` (fit pumap), `encode_user.py` (load reducer + transform) |

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
python map/cluster.py --algo kmeans --k 24                # cluster 128-d + tự đặt tên cụm
python map/viz.py --method pumap2d --color cluster --cluster kmeans \
    --overlay overlay_user_me                             # render HTML (overlay tuỳ chọn)
```

## Pipeline
`build_base` (vector+metadata) → `project` (pumap2d → coords + reducer) → `cluster` (128-d → nhãn +
tên cụm) → `encode_user` (user point + top-K neighbor) → `viz` (HTML 2D). Cluster làm ở **128-d**,
tô màu chéo lên projection — KHÔNG cluster trên tọa độ 2D.

## Nhật ký findings (không gian xếp theo KHÁN GIẢ, không theo genre)
Đo trên 128-d giải thích vì sao map "lùm nhùm": không gian two-tower xếp theo **khán giả (co-watch)**,
không theo nhãn genre:
- mean-cosine-tới-tâm: franchise (Pokemon/Gundam/Dragon Ball/Conan…) **0.75–0.90**, genre rộng
  (Action .46, Comedy .57, Drama .53) **≈ baseline ngẫu nhiên .30** → smear khắp map; genre
  1-khán-giả (Hentai .87, Avant Garde .84, Boys Love .80) chặt + lệch hẳn 1 góc.
- Cluster 128-d: HDBSCAN ra rất ít cụm tách bạch (không gian liên tục) → KMeans k=24 để tô màu hợp
  lý hơn; mỗi cụm tự đặt **tên theo genre+theme lift cao** (Mecha·Space=Gundam, Strategy Game·
  Adventure=Pokemon/Digimon, Suspense·Mystery=Death Note/AoT/FMA, Hentai, Boys Love·Idols…).
- `build_base` kéo `themes_list` (Mecha/Isekai/Music…) — tín hiệu khán giả mạnh hơn genre rộng.

Demographic (shounen/seinen…) bị loại làm trục màu chính: 71% rỗng.

TODO: Phase 2 — export base map (JSON/binary) cho frontend WebGL + API "you are here".
