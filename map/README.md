# map/ — bản đồ trực quan không gian Two-Tower

Công cụ **phân tích offline** (KHÔNG phải serving): chiếu output item-tower 128-d (`artifacts/
item_vectors.npy`, đã L2-norm, cosine = khoảng cách góc) xuống 2D / mặt cầu để quan sát cluster
anime theo khán giả, và **đặt điểm mới** lên map có sẵn: vector user ("you are here") + top-K gợi ý,
centroid genre, và ItemTower-probe genre.

## Firewall
- Phần item-map + user-point: chỉ ĐỌC `artifacts/` + `cleaned-data/details.csv` (sạch).
- Phần genre-**probe**: load `retriever/checkpoints/best.pt` + import `retriever/src` — vượt ranh giới
  train/serve, **chỉ chấp nhận trong map/** vì là phân tích offline. KHÔNG đem pattern này vào `service/`.
- Không sửa `artifacts/`. Output ghi `map/outputs/` (gitignored).

## Phương pháp (ma trận)

| method (`--method`) | chiều | đặt điểm mới | vai trò |
|---|---|---|---|
| `umap2d` | 2D | `.transform` ✓ | **map chính 2D** |
| `umap_sphere` | sphere (haversine→xyz) | ✓ | **sphere native** |
| `pumap2d` (Parametric UMAP) | 2D | forward-pass ✓ | đặt user live (deploy-able); cần TensorFlow |
| `pca2d` / `pca_sphere` | 2D / sphere (3D→L2) | exact ✓ | baseline nhanh/chính xác |
| `densmap2d` | 2D | ✗ (không có `.transform`) | mật độ cục bộ (cụm chặt/loãng) — chỉ vẽ/cluster |
| `tsne2d` | 2D | ✗ | hình đẹp / cluster |
| `pacmap2d` | 2D | ✗ (transform cần basis) | cấu trúc toàn cục — chỉ vẽ/cluster |

**Đặt được điểm mới** (user/genre overlay): chỉ `umap2d`, `umap_sphere`, `pca2d`, `pca_sphere`,
`pumap2d` (lưu reducer). densMAP/t-SNE/PaCMAP = **plot-only** (không transform out-of-sample sạch);
densMAP KHÔNG có biến thể sphere (lib chỉ output euclidean) → dùng `umap_sphere`/`pca_sphere`.

Phân cụm làm ở **128-d** (`cluster.py`), tô màu chéo mọi projection — KHÔNG cluster trên tọa độ 2D.

### Chạy `pumap2d` (TensorFlow) trên Colab
Local mac crash TF → dùng `run_colab.ipynb`: đặt data vào `MyDrive/map_data/`
(`artifacts/`, `train-data/`, `checkpoints/`, `cleaned-data/` — xem cell đầu notebook), notebook
clone code từ GitHub, symlink data + `map/outputs/`→Drive, chạy full pipeline rồi ghi HTML thẳng
về `MyDrive/map_data/outputs/` (kéo về local mở browser).

## Chạy

```bash
# 0) base 1 lần (join vector + metadata genre)
venv/bin/python map/build_base.py

# 1) fit projection (mỗi method 1 lần) -> coords + reducer
venv/bin/python map/project.py --method umap2d
venv/bin/python map/project.py --method umap_sphere

# 2) cluster 128-d (tuỳ chọn, để tô màu)
venv/bin/python map/cluster.py --algo hdbscan

# 3) đặt điểm mới
venv/bin/python map/encode_user.py <username> --method umap2d         # user dataset
venv/bin/python map/encode_user.py --mal-ids ids.txt --name me --method umap_sphere
venv/bin/python map/encode_genre.py --method umap2d --genre Action     # centroid + probe
venv/bin/python map/encode_genre.py --method umap_sphere --all

# 4) render HTML tương tác
venv/bin/python map/viz.py --method umap2d --color primary_genre
venv/bin/python map/viz.py --method umap_sphere --color cluster --cluster hdbscan \
    --overlay overlay_user_me overlay_genre_Action
```

## Nhật ký findings (2026-06-16, verify lần đầu)

- **best.pt hiện tại = `final_syn` (synopsis dim64)** — ItemTower bật `use_synopsis`. Probe genre
  đi nhánh low-info (`no_synopsis` param) vì item tổng hợp không có synopsis.
- **base**: 22,821 anime real (1,142 cold), genre-join phủ 86%.
- **Đặt điểm chạy tốt**: `umap2d`/`umap_sphere` (transform drift ~0.12 — UMAP transform xấp xỉ,
  bình thường), `pca2d`/`pca_sphere` (drift 0.0, exact). encode_user: user 370-item → 15 neighbor OK.
- **Probe vs centroid** (`encode_genre --all`, cosine): đa số genre 0.80–0.92 (probe ≈ vùng thật),
  NHƯNG genre hay-đi-kèm/hiếm-đứng-một-mình thấp: Suspense .32, Award Winning .41, Ecchi .53,
  Girls Love .54 → "item chỉ có 1 genre" rơi off-manifold. ⇒ **centroid trung thực hơn cho bản đồ
  phủ sóng; probe chỉ minh hoạ "hướng genre"**.
- **Cluster 128-d**: HDBSCAN (min_cluster_size=150) ra **2 cụm + ~19.9k noise** → không gian
  two-tower khá liên tục, ÍT cụm mật-độ tách bạch; KMeans k=20 dùng để tô màu thì hợp lý hơn.
- **`pumap2d` KHÔNG chạy được local**: TensorFlow 2.20 abort lúc import trên máy này
  (`mutex lock failed` — C++ runtime, không catch được từ Python; py3.9/macOS cũ). Code path đúng,
  chạy được nơi TF import OK (Colab/Linux). Nếu bỏ pumap → gỡ `tensorflow` khỏi `requirements.txt`.

TODO khi xem HTML: chốt method đẹp nhất rồi prune libs thừa (nhất là `tensorflow`).
