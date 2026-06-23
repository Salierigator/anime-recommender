"""viz.py — render bản đồ 2D tương tác (Plotly HTML, WebGL scatter) (CLI).

Đọc outputs/base + coords_<method> (+ clusters_* nếu color=cluster) -> HTML self-contained
(mở browser zoom/hover). Chạy local OK (plotly/pandas, không cần TF/umap).

    python map/viz.py --method pumap2d --color cluster --cluster kmeans
    python map/viz.py --method pumap2d --color cluster --cluster kmeans --overlay overlay_user_me
"""
from __future__ import annotations

import argparse

import pandas as pd

import _common as C


def load_plot_df(method: str, color: str, cluster_algo: str,
                 coords: pd.DataFrame | None = None) -> pd.DataFrame:
    base = pd.read_parquet(C.OUTPUTS / "base.parquet")
    if coords is None:                                       # sweep truyền coords RAM -> khỏi ghi file
        coords = C.load_coords(method)
    df = coords.merge(base, on="anime_idx", how="left")

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
    return df


def add_cluster_labels(fig, df: pd.DataFrame):
    """Ghi tên cụm tại tâm cụm (median x/y) -> đọc map bằng chữ, không chỉ màu."""
    import plotly.graph_objects as go

    g = df.groupby("cluster")
    fig.add_trace(go.Scatter(x=g["x"].median().values, y=g["y"].median().values, mode="text",
                             text=list(g.groups), name="cluster labels", hoverinfo="skip",
                             textfont=dict(size=12, color="black")))


def add_overlays(fig, names: list[str]):
    import plotly.graph_objects as go

    style = {"user": ("diamond", 16, "black"), "neighbor": ("circle", 9, "#222")}
    for name in names:
        path = C.OUTPUTS / (name if name.endswith(".parquet") else f"{name}.parquet")
        if not path.exists():
            print(f"⚠ bỏ overlay (không thấy): {path}")
            continue
        ov = pd.read_parquet(path)
        for kind, grp in ov.groupby("kind"):
            sym, size, col = style.get(kind, ("circle", 12, "black"))
            text = grp["label"] if kind == "user" else None
            fig.add_trace(go.Scatter(
                x=grp["x"], y=grp["y"], name=kind, text=text,
                mode="markers+text" if text is not None else "markers", textposition="top center",
                marker=dict(size=size, symbol=sym, color=col, line=dict(width=1, color="white"))))


def build_fig(df: pd.DataFrame, color: str, title: str):
    """Scatter 2D + nhãn cụm + layout (KHÔNG overlay). Dùng chung CLI + sweep notebook."""
    import plotly.express as px

    hover = {"primary_genre": True, "x": False, "y": False}
    kw = dict(color=color, hover_name="title", hover_data=hover, opacity=0.55, title=title)
    if color == "cluster":
        kw["color_discrete_sequence"] = px.colors.qualitative.Dark24
    fig = px.scatter(df, x="x", y="y", render_mode="webgl", **kw)
    fig.update_traces(marker=dict(size=2.5))                 # trước khi add text/overlay
    if color == "cluster":
        add_cluster_labels(fig, df)
    fig.update_layout(legend=dict(itemsizing="constant"))
    return fig


def main() -> None:
    ap = argparse.ArgumentParser(description="Render map 2D -> HTML")
    ap.add_argument("--method", default="pumap2d")
    ap.add_argument("--color", choices=["primary_genre", "cluster", "popularity"],
                    default="primary_genre")
    ap.add_argument("--cluster", default="kmeans", help="algo cho color=cluster (kmeans|hdbscan)")
    ap.add_argument("--overlay", nargs="*", default=[], help="file overlay trong outputs/ (không cần .parquet)")
    ap.add_argument("--suffix", default="", help="hậu tố tên file html")
    args = ap.parse_args()

    df = load_plot_df(args.method, args.color, args.cluster)
    fig = build_fig(df, args.color, f"{args.method} · color={args.color} · {len(df):,} anime")
    add_overlays(fig, args.overlay)

    out = C.OUTPUTS / f"{args.method}{('_' + args.suffix) if args.suffix else ''}.html"
    fig.write_html(out, include_plotlyjs=True, full_html=True)
    print(f"-> {out}  (2D, {len(df):,} điểm, overlay={args.overlay})")


if __name__ == "__main__":
    main()
