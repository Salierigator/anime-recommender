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


def kde_dominant(x, y, labels, uniq, bins: int = 480, sigma: float = 6.0, pad: float = 0.5):
    """Trường KDE theo cụm trên grid chung -> (ex, ey, total, dom): biên bins x/y, tổng mật độ,
    index cụm áp đảo mỗi cell. Dùng chung render_territory + map/export_service.py (nền web).
    Đổi bins nhớ scale sigma cùng tỉ lệ (sigma tính theo CELL) để giữ nguyên bandwidth vật lý."""
    import numpy as np
    from scipy.ndimage import gaussian_filter

    ex = np.linspace(x.min() - pad, x.max() + pad, bins + 1)
    ey = np.linspace(y.min() - pad, y.max() + pad, bins + 1)
    stack = np.zeros((len(uniq), bins, bins))
    for i, c in enumerate(uniq):
        m = labels == c
        H, _, _ = np.histogram2d(x[m], y[m], bins=[ex, ey])
        stack[i] = gaussian_filter(H.T, sigma)
    return ex, ey, stack.sum(0), stack.argmax(0)


def territory_rgba(uniq, total, dom):
    """RGBA [bins,bins,4] phần FILL của kde_boundary: cell tô màu cụm áp đảo (palette
    tab20+tab20b cố định theo label) + alpha theo mật độ (norm percentile-99)."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pal = np.array(list(plt.get_cmap("tab20").colors) + list(plt.get_cmap("tab20b").colors))[:, :3]
    img = np.zeros((*dom.shape, 4))
    for i in range(len(uniq)):
        img[dom == i, :3] = pal[uniq[i] % len(pal)]
    img[..., 3] = np.clip(total / np.percentile(total[total > 0], 99), 0, 1) ** 0.5 * 0.7
    return img


def render_territory(df: pd.DataFrame, out_path, label_top: int = 20,
                     bins: int = 480, sigma: float = 6.0):
    """CHỐT: bản đồ 'territory' kde_boundary (matplotlib PNG tĩnh) — mỗi cell tô theo cụm ÁP ĐẢO (KDE
    mật độ mỗi cụm) + đường biên trắng, nhãn = tên cụm (log-odds) ở tâm top-N cụm lớn nhất. df cần
    x, y, label, cluster (tên). KHÔNG tương tác (hover/zoom/you-are-here vẫn ở --style points / frontend)."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x, y, labels = df["x"].to_numpy(), df["y"].to_numpy(), df["label"].to_numpy()
    uniq = sorted(set(labels.tolist()) - {-1})
    ex, ey, total, dom = kde_dominant(x, y, labels, uniq, bins=bins, sigma=sigma)
    img = territory_rgba(uniq, total, dom)

    cx, cy = 0.5 * (ex[:-1] + ex[1:]), 0.5 * (ey[:-1] + ey[1:])
    Xg, Yg = np.meshgrid(cx, cy)
    fig, ax = plt.subplots(figsize=(12, 11))
    ax.set_facecolor("#0b1020")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    ax.imshow(img, origin="lower", extent=[ex[0], ex[-1], ey[0], ey[-1]], aspect="equal",
              interpolation="bilinear")
    thr = np.percentile(total[total > 0], 40)
    for i in range(len(uniq)):
        mask = ((dom == i) & (total > thr)).astype(float)
        if mask.sum() > 5:
            ax.contour(Xg, Yg, mask, levels=[0.5], colors="white", linewidths=0.7, alpha=0.8)

    from collections import Counter
    name_by_label = df.drop_duplicates("label").set_index("label")["cluster"]
    for lab in [l for l, _ in Counter(labels.tolist()).most_common() if l in uniq][:label_top]:
        m = labels == lab
        ax.text(np.median(x[m]), np.median(y[m]), str(name_by_label.get(lab, lab)), fontsize=9,
                weight="bold", ha="center", va="center", color="white",
                bbox=dict(boxstyle="round,pad=0.15", fc="black", alpha=0.35, ec="none"))
    fig.suptitle(f"pumap2d · territory (kde_boundary) · {len(uniq)} cụm · {len(df):,} anime",
                 fontsize=13, y=0.98)
    fig.savefig(out_path, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)


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
    ap.add_argument("--suffix", default="", help="hậu tố tên file output")
    ap.add_argument("--style", choices=["points", "territory"], default="points",
                    help="points = Plotly HTML tương tác (hover/zoom/overlay); "
                         "territory = CHỐT kde_boundary PNG tĩnh")
    ap.add_argument("--label-top", type=int, default=20, help="territory: số cụm lớn nhất được ghi nhãn")
    args = ap.parse_args()

    if args.style == "territory":                                # CHỐT: kde_boundary (k28 + log-odds)
        df = load_plot_df(args.method, "cluster", args.cluster)
        out = C.OUTPUTS / f"{args.method}_territory{('_' + args.suffix) if args.suffix else ''}.png"
        render_territory(df, out, label_top=args.label_top)
        print(f"-> {out}  (territory kde_boundary, {len(df):,} anime)")
        return

    df = load_plot_df(args.method, args.color, args.cluster)
    fig = build_fig(df, args.color, f"{args.method} · color={args.color} · {len(df):,} anime")
    add_overlays(fig, args.overlay)

    out = C.OUTPUTS / f"{args.method}{('_' + args.suffix) if args.suffix else ''}.html"
    fig.write_html(out, include_plotlyjs=True, full_html=True)
    print(f"-> {out}  (2D, {len(df):,} điểm, overlay={args.overlay})")


if __name__ == "__main__":
    main()
