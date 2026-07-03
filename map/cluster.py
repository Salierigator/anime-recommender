"""cluster.py — phân cụm trong KHÔNG GIAN 128-d (CLI).

Nguyên tắc: cluster ở 128-d (không gian thật), KHÔNG cluster trên tọa độ 2D (UMAP/t-SNE bịa
khoảng cách). Nhãn ra dùng tô màu chéo MỌI projection (viz.py --color cluster).

Vector đã L2-norm -> euclidean trên vector chuẩn hoá đơn điệu với cosine (spherical k-means xấp xỉ).

    venv/bin/python map/cluster.py --algo hdbscan
    venv/bin/python map/cluster.py --algo kmeans --k 28      # CHỐT: k=28 + naming log-odds
"""
from __future__ import annotations

import argparse
from collections import Counter

import numpy as np
import pandas as pd

import _common as C


def _logodds_names(base: pd.DataFrame, labels: np.ndarray,
                   a0: float = 100.0, min_support: float = 0.05, min_count: int = 3) -> dict:
    """Tên cụm = 2 tag genre+theme ĐẶC TRƯNG nhất theo LOG-ODDS-RATIO + prior Dirichlet (Monroe 2008
    'fightin words'): z = Δlog-odds / sqrt(var) — phân biệt cụm này với phần còn lại, kiểm soát base-rate
    + phương sai (tag hiếm không bị thổi phồng). Sạch/ít lặp hơn lift & tf-idf (đã so ở docs khảo sát map)."""
    n_total = len(base)
    themes = base["themes_list"] if "themes_list" in base.columns else [[]] * n_total
    tagsets = [set(g) | set(t) for g, t in zip(base["genres_list"], themes)]
    gc = Counter(t for s in tagsets for t in s)
    total = sum(gc.values())
    aw = {t: gc[t] / total * a0 for t in gc}                       # prior tỉ lệ tần suất nền
    out = {}
    for lab in sorted(set(labels.tolist()) - {-1}):
        idx = np.where(labels == lab)[0]
        n_cl = len(idx)
        yi = Counter(t for i in idx for t in tagsets[i])
        ni = sum(yi.values())
        n_rest = total - ni
        scored = []
        for t, c in yi.items():
            if c < min_count or c / n_cl < min_support:
                continue
            yiw, yrw = yi[t], gc[t] - yi[t]
            term_i = np.log(yiw + aw[t]) - np.log(ni + a0 - yiw - aw[t])
            term_r = np.log(yrw + aw[t]) - np.log(n_rest + a0 - yrw - aw[t])
            var = 1.0 / (yiw + aw[t]) + 1.0 / (yrw + aw[t])
            scored.append(((term_i - term_r) / np.sqrt(var), t))
        scored.sort(reverse=True)
        out[lab] = "·".join(t for _, t in scored[:2]) if scored \
            else base.iloc[idx]["primary_genre"].mode().iloc[0]
    return out


def name_clusters(base: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Đặt tên mỗi cụm bằng log-odds-ratio (xem `_logodds_names`). Trả [label, name, size, examples]
    (examples = 3 title phổ biến nhất — sẵn cho hover/tooltip frontend)."""
    names = _logodds_names(base, labels)
    b = base.assign(_label=labels)
    rows = []
    for lab, grp in b.groupby("_label"):
        name = "noise" if lab == -1 else names.get(lab, "?")
        ex = " · ".join(grp.sort_values("popularity")["title"].head(3).astype(str).tolist())
        rows.append({"label": int(lab), "name": name, "size": int(len(grp)), "examples": ex})

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
    ap.add_argument("--k", type=int, default=28, help="số cụm (kmeans) — CHỐT k=28 cho map territory")
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
