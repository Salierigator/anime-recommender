"""cluster.py — phân cụm trong KHÔNG GIAN 128-d (CLI).

Nguyên tắc: cluster ở 128-d (không gian thật), KHÔNG cluster trên tọa độ 2D (UMAP/t-SNE bịa
khoảng cách). Nhãn ra dùng tô màu chéo MỌI projection (viz.py --color cluster).

Vector đã L2-norm -> euclidean trên vector chuẩn hoá đơn điệu với cosine (spherical k-means xấp xỉ).

    venv/bin/python map/cluster.py --algo hdbscan
    venv/bin/python map/cluster.py --algo kmeans --k 20
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import _common as C


def main() -> None:
    ap = argparse.ArgumentParser(description="Cluster item vectors trong 128-d")
    ap.add_argument("--algo", choices=["hdbscan", "kmeans"], required=True)
    ap.add_argument("--k", type=int, default=20, help="số cụm (kmeans)")
    ap.add_argument("--min-cluster-size", type=int, default=150, help="hdbscan")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    base, X = C.load_base()
    print(f"Cluster {args.algo} trên {X.shape} (128-d) ...")

    if args.algo == "kmeans":
        from sklearn.cluster import KMeans
        labels = KMeans(n_clusters=args.k, random_state=args.seed, n_init="auto").fit_predict(X)
    else:
        from sklearn.cluster import HDBSCAN
        labels = HDBSCAN(min_cluster_size=args.min_cluster_size, metric="euclidean").fit_predict(X)

    n_clusters = len(set(labels) - {-1})
    n_noise = int((labels == -1).sum())
    df = pd.DataFrame({"anime_idx": base["anime_idx"].to_numpy().astype(np.int32),
                       "label": labels.astype(np.int32)})
    out = C.OUTPUTS / f"clusters_{args.algo}.parquet"
    df.to_parquet(out)
    print(f"-> {out}  ({n_clusters} cụm" + (f", {n_noise:,} noise" if n_noise else "") + ")")


if __name__ == "__main__":
    main()
