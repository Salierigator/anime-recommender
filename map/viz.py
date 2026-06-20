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
        df["label"] = df["label"].fillna(-1).astype(int)
        names_path = C.OUTPUTS / f"cluster_names_{cluster_algo}.parquet"
        if names_path.exists():                              # tô màu + legend theo TÊN cụm
            nm = pd.read_parquet(names_path)[["label", "name"]]
            df = df.merge(nm, on="label", how="left")
            df["cluster"] = df["name"].fillna("noise")
        else:
            df["cluster"] = df["label"].astype(str)
    elif color == "popularity":
        df["popularity"] = pd.to_numeric(df["popularity"], errors="coerce")
    return df, is3d


def add_sphere_shell(fig):
    """Mặt cầu mờ ĐỤC bán kính 0.97 dưới các điểm (r=1) -> che bán cầu xa, fix lỗi 'điểm mặt sau
    lòi qua mặt trước' (điểm phải opaque mới ghi depth đúng — xem main)."""
    import plotly.graph_objects as go

    u = np.linspace(0, 2 * np.pi, 60)
    v = np.linspace(0, np.pi, 30)
    xs = 0.97 * np.outer(np.cos(u), np.sin(v))
    ys = 0.97 * np.outer(np.sin(u), np.sin(v))
    zs = 0.97 * np.outer(np.ones_like(u), np.cos(v))
    fig.add_trace(go.Surface(x=xs, y=ys, z=zs, showscale=False, hoverinfo="skip",
                             colorscale=[[0, "#dfe3ea"], [1, "#dfe3ea"]], opacity=1.0,
                             lighting=dict(ambient=0.95, diffuse=0.1, specular=0.0)))


def add_cluster_labels(fig, df: pd.DataFrame, is3d: bool):
    """Ghi tên cụm tại tâm cụm (thay 21 nhãn genre dồn cục) -> đọc được map bằng chữ, không chỉ màu."""
    import plotly.graph_objects as go

    g = df.groupby("cluster")
    if is3d:
        c = np.stack([g["x"].mean(), g["y"].mean(), g["z"].mean()], axis=1)
        c = c / np.linalg.norm(c, axis=1, keepdims=True).clip(min=1e-9) * 1.04   # đẩy ra ngoài vỏ cầu
        fig.add_trace(go.Scatter3d(x=c[:, 0], y=c[:, 1], z=c[:, 2], mode="text",
                                   text=list(g.groups), name="cluster labels", hoverinfo="skip",
                                   textfont=dict(size=11, color="black")))
    else:
        fig.add_trace(go.Scatter(x=g["x"].median().values, y=g["y"].median().values, mode="text",
                                 text=list(g.groups), name="cluster labels", hoverinfo="skip",
                                 textfont=dict(size=12, color="black")))


def add_overlays(fig, names: list[str], is3d: bool, show_probe: bool = False):
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
            if kind == "probe" and not show_probe:        # probe off-manifold, mặc định ẩn
                continue
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
    ap.add_argument("--show-probe", action="store_true", help="hiện điểm probe (mặc định ẩn — off-manifold)")
    ap.add_argument("--suffix", default="", help="hậu tố tên file html")
    args = ap.parse_args()

    df, is3d = load_plot_df(args.method, args.color, args.cluster)
    color_col = {"primary_genre": "primary_genre", "cluster": "cluster",
                 "popularity": "popularity"}[args.color]
    hover = {"primary_genre": True, "x": False, "y": False}
    if is3d:
        hover["z"] = False

    # 3D: marker ĐỤC (opacity=1) mới ghi depth -> không bị điểm mặt sau lòi qua (kèm vỏ cầu mờ).
    kw = dict(color=color_col, hover_name="title", hover_data=hover, opacity=1.0 if is3d else 0.55,
              title=f"{args.method} · color={args.color} · {len(df):,} anime")
    if args.color == "cluster":
        kw["color_discrete_sequence"] = px.colors.qualitative.Dark24
    if is3d:
        fig = px.scatter_3d(df, x="x", y="y", z="z", **kw)
    else:
        fig = px.scatter(df, x="x", y="y", render_mode="webgl", **kw)
    fig.update_traces(marker=dict(size=2.0 if is3d else 2.5))     # trước khi add Surface/text

    if is3d:
        add_sphere_shell(fig)
    if args.color == "cluster":
        add_cluster_labels(fig, df, is3d)
    add_overlays(fig, args.overlay, is3d, show_probe=args.show_probe)
    fig.update_layout(legend=dict(itemsizing="constant"))

    out = C.OUTPUTS / f"{args.method}{('_' + args.suffix) if args.suffix else ''}.html"
    fig.write_html(out, include_plotlyjs=True, full_html=True)
    print(f"-> {out}  ({'3D sphere' if is3d else '2D'}, {len(df):,} điểm, overlay={args.overlay})")


if __name__ == "__main__":
    main()
