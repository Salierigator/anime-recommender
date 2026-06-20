"""project.py — fit 1 phương pháp giảm chiều trên item_vectors -> coords + reducer (CLI).

Đọc outputs/{base,vectors_real} (build_base.py trước). Ghi coords_<method>.parquet (hiển thị)
+ reducer_<method>.pkl (để encode_user/encode_genre đặt điểm mới qua .transform).

    venv/bin/python map/build_base.py                       # 1 lần
    venv/bin/python map/project.py --method umap2d
    venv/bin/python map/project.py --method umap_sphere
    venv/bin/python map/project.py --method pca_sphere

Methods:
  umap2d  umap_sphere  pca2d  pca_sphere  pumap2d   -> ĐẶT ĐIỂM MỚI được (lưu reducer .transform)
  densmap2d  tsne2d  pacmap2d                       -> CHỈ vẽ/cluster (không transform sạch out-of-sample)
  (densmap_sphere KHÔNG được — densMAP chỉ hỗ trợ output euclidean)
"""
from __future__ import annotations

import argparse
import time

import numpy as np

import _common as C

# chỉ các method này transform out-of-sample sạch -> lưu reducer cho encode_user/encode_genre.
# densmap (không có .transform), tsne (không transform), pacmap (transform cần basis, yếu) = plot-only.
PLACEMENT = {"umap2d", "umap_sphere", "pca2d", "pca_sphere", "pumap2d"}
UNSUPPORTED = {"densmap_sphere"}  # densMAP không hỗ trợ output_metric haversine


def fit(method: str, X: np.ndarray, n_neighbors: int, min_dist: float, seed: int):
    """Trả (reducer, emb_raw). emb_raw chưa hậu xử lý sphere (to_display_coords lo)."""
    if method in ("umap2d", "umap_sphere", "densmap2d", "densmap_sphere"):
        import umap
        sphere = method.endswith("sphere")
        reducer = umap.UMAP(
            n_components=2, metric="cosine", n_neighbors=n_neighbors, min_dist=min_dist,
            densmap=method.startswith("densmap"),
            output_metric="haversine" if sphere else "euclidean",
            random_state=seed, verbose=True,
        )
        return reducer, reducer.fit_transform(X)

    if method == "pumap2d":
        from umap.parametric_umap import ParametricUMAP
        reducer = ParametricUMAP(n_components=2, metric="cosine",
                                 n_neighbors=n_neighbors, min_dist=min_dist, verbose=True)
        return reducer, reducer.fit_transform(X)

    if method in ("pca2d", "pca_sphere"):
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=3 if method == "pca_sphere" else 2, random_state=seed)
        return reducer, reducer.fit_transform(X)

    if method == "tsne2d":
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, metric="cosine", init="pca", random_state=seed, verbose=1)
        return reducer, reducer.fit_transform(X)

    if method == "pacmap2d":
        import pacmap
        reducer = pacmap.PaCMAP(n_components=2, random_state=seed)
        # PaCMAP cần giữ data train để transform out-of-sample -> save_basis
        return reducer, reducer.fit_transform(X, init="pca")

    raise SystemExit(f"method không hỗ trợ: {method}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Fit projection -> coords + reducer")
    ap.add_argument("--method", required=True)
    ap.add_argument("--n-neighbors", type=int, default=15)
    ap.add_argument("--min-dist", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if args.method in UNSUPPORTED:
        raise SystemExit(f"method '{args.method}' không hỗ trợ (densMAP chỉ output euclidean — "
                         "dùng umap_sphere hoặc pca_sphere cho mặt cầu).")

    base, X = C.load_base()
    print(f"Fit {args.method} trên {X.shape} ...")
    t0 = time.time()
    reducer, emb_raw = fit(args.method, X, args.n_neighbors, args.min_dist, args.seed)
    emb = C.to_display_coords(args.method, emb_raw)
    dt = time.time() - t0

    cp = C.save_coords(args.method, base["anime_idx"].to_numpy(), emb)
    print(f"[{dt:.1f}s] coords -> {cp}  shape {emb.shape}")

    if args.method not in PLACEMENT:
        print(f"⚠ {args.method} CHỈ vẽ/cluster — không transform out-of-sample sạch, KHÔNG lưu "
              "reducer (encode_user/encode_genre không đặt điểm lên map này).")
        return

    rp = C.save_reducer(args.method, reducer)
    print(f"reducer -> {rp}")

    # sanity: transform lại vài item đã fit, so coord gốc (sphere/pca-sphere có hậu xử lý nên so xyz)
    if args.method != "pumap2d":          # pumap forward có sai số nhỏ, bỏ qua sanity nghiêm
        probe = X[:200]
        re = C.transform_to_coords(args.method, reducer, probe)
        drift = float(np.linalg.norm(re - emb[:200], axis=1).mean())
        print(f"sanity transform-lại 200 item: drift trung bình {drift:.4f} "
              f"(nhỏ = transform khớp fit)")


if __name__ == "__main__":
    main()
