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

# 2) cluster 128-d (+ tự đặt TÊN cụm theo genre/theme đặc trưng -> cluster_names_<algo>.parquet)
venv/bin/python map/cluster.py --algo kmeans --k 24

# 3) đặt điểm mới
venv/bin/python map/encode_user.py <username> --method umap2d         # user dataset
venv/bin/python map/encode_user.py --mal-ids ids.txt --name me --method umap_sphere
venv/bin/python map/encode_genre.py --method umap2d --genre Action     # centroid + probe
venv/bin/python map/encode_genre.py --method umap_sphere --all

# 4) render HTML tương tác (color=cluster -> tô + nhãn theo TÊN cụm; probe ẩn mặc định)
venv/bin/python map/viz.py --method umap2d --color cluster --cluster kmeans
venv/bin/python map/viz.py --method umap_sphere --color cluster --cluster kmeans \
    --overlay overlay_user_me        # thêm --show-probe nếu muốn xem probe
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

## Nhật ký findings (2026-06-17, Phase 1 — đọc-được hình)

Sau khi user xem HTML Colab (pumap2d + umap_sphere): genre rộng đè nhau, centroid genre dồn
cục, probe vô nghĩa, sphere trong suốt. Đo lại trên 128-d giải thích **không gian xếp theo
KHÁN GIẢ (co-watch), không theo nhãn genre**:
- mean-cosine-tới-tâm: franchise (Pokemon/Gundam/Dragon Ball/Conan…) **0.75–0.90**, genre rộng
  (Action .46, Comedy .57, Drama .53) **≈ baseline ngẫu nhiên .30** → smear khắp map; genre =
  1-khán-giả (Hentai .87, Avant Garde .84, Boys Love .80) chặt + nằm lệch hẳn 1 góc.
- centroid genre 2D dồn cục vì genre rộng phủ 74–97% bề rộng map → trung bình về tâm (toán học).
- probe co cụm (sphere: spread .47 vs centroid .98) → off-manifold, **ẩn mặc định** trong viz.

**Đổi để hình đọc được (thay 21 genre smear):**
- `build_base` kéo thêm `themes_list` (Mecha/Isekai/School/Music…) — tín hiệu khán giả mạnh hơn genre.
- `cluster.py` tự đặt TÊN cụm theo genre+theme **lift cao** (đặc trưng), không phải top-1 genre →
  k=24 ra tên có nghĩa: Mecha·Space (Gundam), Strategy Game·Adventure (Pokemon/Digimon),
  Suspense·Mystery (Death Note/AoT/FMA), Hentai, Boys Love·Idols, Harem·Ecchi, idol, Music…
  (cụm mainstream Comedy·Fantasy/Adventure·Comedy vẫn nhạt — đó là phần liên tục, không tách được).
- `viz.py`: `--color cluster` tô + ghi **nhãn tên cụm tại tâm cụm**; sphere thêm **vỏ cầu đục
  r=0.97 + marker opaque** → che bán cầu xa, hết "điểm mặt sau lòi qua mặt trước".
- Out HTML: `umap2d_named.html`, `pumap2d_named.html` (2D), `umap_sphere_named.html` (sphere).

Demographic (shounen/seinen…) bị loại làm trục màu chính: 71% rỗng. → genre vẫn là lựa chọn phụ.

TODO: Phase 2 — export base map (JSON/binary) cho frontend tự render WebGL + API "you are here"
(backend `reducer.transform` → toạ độ); chốt method rồi prune libs thừa (nhất là `tensorflow`).
