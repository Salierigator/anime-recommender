"""viz.py — render bản đồ tương tác (Plotly HTML): 2D scatter | sphere 3D (CLI).

Đọc outputs/base + coords_<method> (+ clusters_* nếu color=cluster) -> HTML self-contained
(mở browser xoay/zoom/hover). Tự nhận 2D/3D theo việc coords có cột z hay không.

    venv/bin/python map/viz.py --method umap2d --color primary_genre
    venv/bin/python map/viz.py --method umap_sphere --color cluster --cluster hdbscan \
        --overlay overlay_user_me overlay_genre_Action
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import _common as C


def load_plot_df(method: str, color: str, cluster_algo: str) -> tuple[pd.DataFrame, bool]:
    base = pd.read_parquet(C.OUTPUTS / "base.parquet")
    coords = C.load_coords(method)
    df = coords.merge(base, on="anime_idx", how="left")
    is3d = "z" in df.columns

    if color == "cluster":
        cl = pd.read_parquet(C.OUTPUTS / f"clusters_{cluster_algo}.parquet")
        df = df.merge(cl, on="anime_idx", how="left")
        df["cluster"] = df["label"].fillna(-1).astype(int).astype(str)
    elif color == "popularity":
        df["popularity"] = pd.to_numeric(df["popularity"], errors="coerce")
    return df, is3d


def add_overlays(fig, names: list[str], is3d: bool):
    import plotly.graph_objects as go

    # symbol hợp lệ cho CẢ Scatter (2D) lẫn Scatter3d: circle/cross/diamond/square/x
    style = {"user": ("diamond", 16, "black"), "neighbor": ("circle", 9, "#222"),
             "centroid": ("cross", 14, "black"), "probe": ("x", 14, "crimson")}
    for name in names:
        path = C.OUTPUTS / (name if name.endswith(".parquet") else f"{name}.parquet")
        if not path.exists():
            print(f"⚠ bỏ overlay (không thấy): {path}")
            continue
        ov = pd.read_parquet(path)
        for kind, grp in ov.groupby("kind"):
            sym, size, col = style.get(kind, ("circle", 12, "black"))
            text = grp["label"] if kind in ("user", "centroid", "probe") else None
            common = dict(name=kind, text=text, mode="markers+text" if text is not None else "markers",
                          textposition="top center",
                          marker=dict(size=size, symbol=sym, color=col,
                                      line=dict(width=1, color="white")))
            if is3d:
                fig.add_trace(go.Scatter3d(x=grp["x"], y=grp["y"], z=grp["z"], **common))
            else:
                fig.add_trace(go.Scatter(x=grp["x"], y=grp["y"], **common))


def main() -> None:
    import plotly.express as px

    ap = argparse.ArgumentParser(description="Render map -> HTML")
    ap.add_argument("--method", required=True)
    ap.add_argument("--color", choices=["primary_genre", "cluster", "popularity"],
                    default="primary_genre")
    ap.add_argument("--cluster", default="hdbscan", help="algo cho color=cluster (hdbscan|kmeans)")
    ap.add_argument("--overlay", nargs="*", default=[], help="file overlay trong outputs/ (không cần .parquet)")
    ap.add_argument("--suffix", default="", help="hậu tố tên file html")
    args = ap.parse_args()

    df, is3d = load_plot_df(args.method, args.color, args.cluster)
    color_col = {"primary_genre": "primary_genre", "cluster": "cluster",
                 "popularity": "popularity"}[args.color]
    hover = {"primary_genre": True, "x": False, "y": False}
    if is3d:
        hover["z"] = False

    kw = dict(color=color_col, hover_name="title", hover_data=hover, opacity=0.55,
              title=f"{args.method} · color={args.color} · {len(df):,} anime")
    if is3d:
        fig = px.scatter_3d(df, x="x", y="y", z="z", **kw)
    else:
        fig = px.scatter(df, x="x", y="y", render_mode="webgl", **kw)
    fig.update_traces(marker=dict(size=2.5))

    add_overlays(fig, args.overlay, is3d)
    fig.update_layout(legend=dict(itemsizing="constant"))

    out = C.OUTPUTS / f"{args.method}{('_' + args.suffix) if args.suffix else ''}.html"
    fig.write_html(out, include_plotlyjs=True, full_html=True)
    print(f"-> {out}  ({'3D sphere' if is3d else '2D'}, {len(df):,} điểm, overlay={args.overlay})")


if __name__ == "__main__":
    main()
