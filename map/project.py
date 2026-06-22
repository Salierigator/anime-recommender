"""project.py — fit Parametric UMAP (pumap2d) trên item_vectors -> coords + reducer (CLI).

Đọc outputs/{base,vectors_real} (build_base.py trước). Ghi coords_pumap2d.parquet (hiển thị 2D)
+ reducer_pumap2d/ (ParametricUMAP.save -> encode_user đặt điểm mới qua .transform forward-pass).

pumap cần TensorFlow -> CHẠY TRÊN COLAB (run_colab.ipynb); local mac (py3.9/macOS26) TF abort.

    python map/build_base.py            # 1 lần
    python map/project.py               # fit pumap2d
"""
from __future__ import annotations

import argparse
import time

import numpy as np

import _common as C


def fit(X: np.ndarray, n_neighbors: int, min_dist: float):
    """Fit ParametricUMAP -> (reducer, emb 2D). umap import lazy (TF chỉ cần lúc fit)."""
    from umap.parametric_umap import ParametricUMAP
    reducer = ParametricUMAP(n_components=2, metric="cosine",
                             n_neighbors=n_neighbors, min_dist=min_dist, verbose=True)
    return reducer, reducer.fit_transform(X)


def main() -> None:
    ap = argparse.ArgumentParser(description="Fit pumap2d -> coords + reducer")
    ap.add_argument("--n-neighbors", type=int, default=15)
    ap.add_argument("--min-dist", type=float, default=0.1)
    args = ap.parse_args()

    base, X = C.load_base()
    print(f"Fit pumap2d trên {X.shape} ...")
    t0 = time.time()
    reducer, emb = fit(X, args.n_neighbors, args.min_dist)
    dt = time.time() - t0

    cp = C.save_coords("pumap2d", base["anime_idx"].to_numpy(), emb.astype(np.float32))
    print(f"[{dt:.1f}s] coords -> {cp}  shape {emb.shape}")
    rp = C.save_reducer("pumap2d", reducer)
    print(f"reducer -> {rp}")


if __name__ == "__main__":
    main()
