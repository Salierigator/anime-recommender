"""cluster.py — phân cụm trong KHÔNG GIAN 128-d (CLI).

Nguyên tắc: cluster ở 128-d (không gian thật), KHÔNG cluster trên tọa độ 2D (UMAP/t-SNE bịa
khoảng cách). Nhãn ra dùng tô màu chéo MỌI projection (viz.py --color cluster).

Vector đã L2-norm -> euclidean trên vector chuẩn hoá đơn điệu với cosine (spherical k-means xấp xỉ).

    venv/bin/python map/cluster.py --algo hdbscan
    venv/bin/python map/cluster.py --algo kmeans --k 20
"""
from __future__ import annotations

import argparse
from collections import Counter

import numpy as np
import pandas as pd

import _common as C


def name_clusters(base: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Đặt tên mỗi cụm theo genre+theme ĐẶC TRƯNG (lift = tần suất trong cụm / toàn cục), thay vì
    top-1 genre (genre rộng như Action/Comedy có mặt khắp nơi -> vô nghĩa). Tên = 2 tag lift cao
    nhất (xuất hiện ở >=15% cụm). Trả [label, name, size, examples] (examples = 3 title phổ biến)."""
    n_total = len(base)
    themes = base["themes_list"] if "themes_list" in base.columns else [[]] * n_total
    tagsets = [set(g) | set(t) for g, t in zip(base["genres_list"], themes)]
    global_df = Counter(t for s in tagsets for t in s)            # doc-freq toàn cục mỗi tag

    b = base.assign(_label=labels, _tags=tagsets)
    rows = []
    for lab, grp in b.groupby("_label"):
        n = len(grp)
        if lab == -1:
            name = "noise"
        else:
            cc = Counter(t for s in grp["_tags"] for t in s)
            scored = sorted(((cc[t] / n) / (global_df[t] / n_total), cc[t] / n, t)
                            for t in cc if cc[t] / n >= 0.15)       # lift, support>=15%
            name = "·".join(t for _, _, t in scored[-2:][::-1]) if scored \
                else grp["primary_genre"].mode().iloc[0]
        ex = " · ".join(grp.sort_values("popularity")["title"].head(3).astype(str).tolist())
        rows.append({"label": int(lab), "name": name, "size": int(n), "examples": ex})

    df = pd.DataFrame(rows).sort_values("size", ascending=False).reset_index(drop=True)
    seen: dict[str, int] = {}                                      # khử trùng tên (cùng top-2 tag)
    for i, nm in enumerate(df["name"]):
        if nm in seen:
            seen[nm] += 1
            df.loc[i, "name"] = f"{nm} ({seen[nm]})"
        else:
            seen[nm] = 1
    return df


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

    names = name_clusters(base, labels)
    names.to_parquet(C.OUTPUTS / f"cluster_names_{args.algo}.parquet")
    print(f"-> cluster_names_{args.algo}.parquet  (tên theo genre+theme đặc trưng)")
    print(names[["name", "size"]].head(12).to_string(index=False))


if __name__ == "__main__":
    main()
