"""explore_viz.py — harness KHÁM PHÁ clustering + naming + coloring cho map SFW (KHÔNG phải pipeline chốt).

Đọc artifacts có sẵn (coords_pumap2d + vectors_real + base) + join thêm cột từ details.csv, rồi render
NHIỀU biến thể ra map/outputs/explore/*.png để so sánh. KHÔNG sửa cluster.py/viz.py/build_base.py.
Toàn bộ chạy LOCAL (sklearn/scipy/matplotlib) — projection giữ nguyên n50_d0.8.

    venv/bin/python map/explore_viz.py

3 nhóm: clus_* (đổi phân cụm) · name_* (đổi cách đặt tên) · col_* (đổi cách tô/loang/ranh giới).
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.ndimage import gaussian_filter

import _common as C

OUT = C.OUTPUTS / "explore"
BINS = 480
SIGMA = 6.0
PALETTE = np.array(list(plt.get_cmap("tab20").colors) + list(plt.get_cmap("tab20b").colors))[:, :3]

# vocab "biome" cho col_knn_genre — theme cụ thể trước, genre rộng sau (ưu tiên tín hiệu khán giả mạnh)
BIOME_PRIORITY = [
    "Mecha", "Space", "Music", "Idols (Female)", "Idols (Male)", "Mahou Shoujo", "Isekai", "Harem",
    "CGDCT", "Iyashikei", "Military", "Historical", "Martial Arts", "Racing", "Vampire", "Parody",
    "Psychological", "Detective", "Samurai", "Team Sports", "Gore",
    "Boys Love", "Girls Love", "Avant Garde", "Sports", "Horror", "Mystery", "Ecchi", "Supernatural",
    "Romance", "Sci-Fi", "Fantasy", "Adventure", "Drama", "Slice of Life", "Action", "Comedy",
]


# ============================ LOAD ============================
def load_all():
    """df (base + coords) căn hàng theo base row; X = vectors_real cùng thứ tự. Join thêm details cols."""
    base, X = C.load_base()                                 # cùng thứ tự (build_base)
    coords = C.load_coords("pumap2d").set_index("anime_idx")
    xy = coords.reindex(base["anime_idx"].to_numpy())       # align theo base
    base = base.copy()
    base["x"] = xy["x"].to_numpy()
    base["y"] = xy["y"].to_numpy()

    det = pd.read_csv(C.CLEANED / "details.csv", usecols=["mal_id", "demographics", "studios", "source"])
    det["demographics_list"] = det["demographics"].map(C._parse_list)
    det["studios_list"] = det["studios"].map(C._parse_list)
    m = base.merge(det[["mal_id", "demographics_list", "studios_list", "source"]], on="mal_id", how="left")
    m["demographics_list"] = m["demographics_list"].apply(lambda v: v if isinstance(v, list) else [])
    m["studios_list"] = m["studios_list"].apply(lambda v: v if isinstance(v, list) else [])
    assert len(m) == len(base)
    print(f"load_all: {len(m):,} điểm, X {X.shape}")
    return m, X


# ============================ CLUSTERING ============================
def clus_kmeans128(X, k):
    from sklearn.cluster import KMeans
    return KMeans(n_clusters=k, random_state=42, n_init="auto").fit_predict(X), None


def clus_gmm128(X, k):
    from sklearn.mixture import GaussianMixture
    g = GaussianMixture(k, covariance_type="diag", random_state=42, max_iter=120).fit(X)
    resp = g.predict_proba(X)
    return resp.argmax(1), resp


def clus_agglo128(X, k):
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.neighbors import kneighbors_graph
    conn = kneighbors_graph(X, n_neighbors=15, include_self=False)
    return AgglomerativeClustering(n_clusters=k, connectivity=conn, linkage="ward").fit_predict(X), None


def clus_hdbscan128(X):
    from sklearn.cluster import HDBSCAN
    return HDBSCAN(min_cluster_size=150, metric="euclidean").fit_predict(X), None


def clus_kmeans2d(df, k):
    from sklearn.cluster import KMeans
    XY = df[["x", "y"]].to_numpy()
    return KMeans(n_clusters=k, random_state=42, n_init="auto").fit_predict(XY), None


def clus_hdbscan2d(df):
    from sklearn.cluster import HDBSCAN
    XY = df[["x", "y"]].to_numpy()
    return HDBSCAN(min_cluster_size=150, metric="euclidean").fit_predict(XY), None


# ============================ NAMING (label -> name) ============================
def _tagsets(df):
    return [set(g) | set(t) for g, t in zip(df["genres_list"], df["themes_list"])]


def _dedupe(names: dict) -> dict:
    seen: dict[str, int] = {}
    out = {}
    for lab, nm in names.items():
        if nm in seen:
            seen[nm] += 1
            out[lab] = f"{nm} ({seen[nm]})"
        else:
            seen[nm] = 1
            out[lab] = nm
    return out


def name_lift(df, labels):
    tags = _tagsets(df)
    N = len(df)
    gdf = Counter(t for s in tags for t in s)
    names = {}
    for lab in sorted(set(labels) - {-1}):
        idx = np.where(labels == lab)[0]
        n = len(idx)
        cc = Counter(t for i in idx for t in tags[i])
        scored = sorted(((cc[t] / n) / (gdf[t] / N), t) for t in cc if cc[t] / n >= 0.15)
        names[lab] = "·".join(t for _, t in scored[-2:][::-1]) if scored \
            else df.iloc[idx]["primary_genre"].mode().iloc[0]
    return _dedupe(names)


def name_tfidf(df, labels):
    tags = _tagsets(df)
    uniq = sorted(set(labels) - {-1})
    tf = {}
    for lab in uniq:
        idx = np.where(labels == lab)[0]
        cc = Counter(t for i in idx for t in tags[i])
        tf[lab] = {t: cc[t] / len(idx) for t in cc}
    dfc = Counter(t for lab in uniq for t, v in tf[lab].items() if v >= 0.10)
    K = len(uniq)
    names = {}
    for lab in uniq:
        idf = {t: np.log((K + 1) / (dfc[t] + 1)) + 1 for t in tf[lab]}
        scored = sorted(((tf[lab][t] * idf[t], t) for t in tf[lab]), reverse=True)
        names[lab] = "·".join(t for _, t in scored[:2]) if scored else "?"
    return _dedupe(names)


def name_exemplar(df, labels):
    names = {}
    for lab in sorted(set(labels) - {-1}):
        g = df[labels == lab].sort_values("popularity")
        t = str(g["title"].iloc[0])
        names[lab] = t if len(t) <= 26 else t[:24] + "…"
    return _dedupe(names)


def name_demo_theme(df, labels):
    tags_theme = list(df["themes_list"])
    N = len(df)
    gdf = Counter(t for s in tags_theme for t in s)
    names = {}
    for lab in sorted(set(labels) - {-1}):
        idx = np.where(labels == lab)[0]
        n = len(idx)
        demos = [d for i in idx for d in df.iloc[i]["demographics_list"]]
        demo = Counter(demos).most_common(1)[0][0] if demos else ""
        cc = Counter(t for i in idx for t in tags_theme[i])
        scored = sorted(((cc[t] / n) / (gdf[t] / N), t) for t in cc if cc[t] / n >= 0.12)
        theme = scored[-1][1] if scored else df.iloc[idx]["primary_genre"].mode().iloc[0]
        names[lab] = f"{demo} · {theme}" if demo else theme
    return _dedupe(names)


# ============================ RENDER HELPERS ============================
def _fig(dark=True):
    fig, ax = plt.subplots(figsize=(12, 11))
    ax.set_facecolor("#0b1020" if dark else "white")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    return fig, ax


def _labels_text(ax, df, labels, names, color="white"):
    for lab, nm in names.items():
        m = labels == lab
        ax.text(df["x"][m].median(), df["y"][m].median(), nm, fontsize=9, weight="bold",
                ha="center", va="center", color=color,
                bbox=dict(boxstyle="round,pad=0.15", fc="black", alpha=0.35, ec="none"))


def _density_stack(x, y, labels, uniq):
    pad = 0.5
    ex = np.linspace(x.min() - pad, x.max() + pad, BINS + 1)
    ey = np.linspace(y.min() - pad, y.max() + pad, BINS + 1)
    stack = np.zeros((len(uniq), BINS, BINS))
    for i, c in enumerate(uniq):
        m = labels == c
        H, _, _ = np.histogram2d(x[m], y[m], bins=[ex, ey])
        stack[i] = gaussian_filter(H.T, SIGMA)
    extent = [ex[0], ex[-1], ey[0], ey[-1]]
    cx = 0.5 * (ex[:-1] + ex[1:])
    cy = 0.5 * (ey[:-1] + ey[1:])
    return stack, extent, np.meshgrid(cx, cy)


def _dom_rgba(stack, uniq, gamma=0.6):
    total = stack.sum(0)
    dom = stack.argmax(0)
    img = np.zeros((BINS, BINS, 4))
    for i in range(len(uniq)):
        img[dom == i, :3] = PALETTE[uniq[i] % len(PALETTE)]
    hi = np.percentile(total[total > 0], 99)
    a = np.clip(total / hi, 0, 1) ** gamma
    img[..., 3] = a
    return img, total, dom


def _save(fig, name, title):
    fig.suptitle(title, fontsize=13, y=0.98)
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"{name}.png"
    fig.savefig(p, dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  ->", p.name)
    return p


# ============================ COLORING RENDERS ============================
def render_kde_fill(df, labels, names, name, title):
    uniq = sorted(set(labels) - {-1})
    stack, extent, _ = _density_stack(df["x"].to_numpy(), df["y"].to_numpy(), labels, uniq)
    img, _, _ = _dom_rgba(stack, uniq)
    fig, ax = _fig(dark=True)
    ax.imshow(img, origin="lower", extent=extent, aspect="equal", interpolation="bilinear")
    _labels_text(ax, df, labels, names)
    return _save(fig, name, title)


def render_kde_boundary(df, labels, names, name, title):
    uniq = sorted(set(labels) - {-1})
    stack, extent, (Xg, Yg) = _density_stack(df["x"].to_numpy(), df["y"].to_numpy(), labels, uniq)
    img, total, dom = _dom_rgba(stack, uniq, gamma=0.5)
    img[..., 3] *= 0.7
    fig, ax = _fig(dark=True)
    ax.imshow(img, origin="lower", extent=extent, aspect="equal", interpolation="bilinear")
    thr = np.percentile(total[total > 0], 40)
    for i in range(len(uniq)):
        mask = ((dom == i) & (total > thr)).astype(float)
        if mask.sum() > 5:
            ax.contour(Xg, Yg, mask, levels=[0.5], colors="white", linewidths=0.7, alpha=0.8)
    _labels_text(ax, df, labels, names)
    return _save(fig, name, title)


def render_hardscatter(df, labels, names, name, title):
    fig, ax = _fig(dark=False)
    for lab in sorted(set(labels) - {-1}):
        m = labels == lab
        ax.scatter(df["x"][m], df["y"][m], s=3, color=PALETTE[lab % len(PALETTE)], alpha=0.6, linewidths=0)
    _labels_text(ax, df, labels, names, color="black")
    return _save(fig, name, title)


def render_gmm_blend(df, resp, names, name, title):
    labels = resp.argmax(1)
    pal = PALETTE[: resp.shape[1]]
    rgb = np.clip(resp @ pal, 0, 1)
    fig, ax = _fig(dark=True)
    ax.scatter(df["x"], df["y"], s=5, c=rgb, alpha=0.75, linewidths=0)
    _labels_text(ax, df, labels, names)
    return _save(fig, name, title)


def render_nebula(df, labels, names, name, title):
    x, y = df["x"].to_numpy(), df["y"].to_numpy()
    pad = 0.5
    ex = np.linspace(x.min() - pad, x.max() + pad, BINS + 1)
    ey = np.linspace(y.min() - pad, y.max() + pad, BINS + 1)
    H, _, _ = np.histogram2d(x, y, bins=[ex, ey])
    Hs = gaussian_filter(H.T, SIGMA * 0.8)
    fig, ax = _fig(dark=True)
    ax.imshow(Hs, origin="lower", extent=[ex[0], ex[-1], ey[0], ey[-1]], cmap="magma",
              aspect="equal", interpolation="bilinear")
    _labels_text(ax, df, labels, names)
    return _save(fig, name, title)


def render_position(df, labels, names, name, title):
    x, y = df["x"].to_numpy(), df["y"].to_numpy()
    cx, cy = x.mean(), y.mean()
    ang = np.arctan2(y - cy, x - cx)
    rad = np.hypot(x - cx, y - cy)
    rad = rad / rad.max()
    hue = (ang + np.pi) / (2 * np.pi)
    rgb = mcolors.hsv_to_rgb(np.stack([hue, 0.35 + 0.55 * rad, np.full_like(hue, 0.95)], 1))
    fig, ax = _fig(dark=True)
    ax.scatter(x, y, s=4, c=rgb, alpha=0.75, linewidths=0)
    _labels_text(ax, df, labels, names)
    return _save(fig, name, title)


def _biome(df):
    prio = {t: i for i, t in enumerate(BIOME_PRIORITY)}
    out = np.full(len(df), -1)
    for r, (g, t) in enumerate(zip(df["genres_list"], df["themes_list"])):
        best, bi = None, 1e9
        for tag in set(g) | set(t):
            if tag in prio and prio[tag] < bi:
                bi, best = prio[tag], tag
        out[r] = BIOME_PRIORITY.index(best) if best else -1
    return out


def render_knn_genre(df, labels_unused, names_unused, name, title):
    biome = _biome(df)
    uniq = sorted(set(biome.tolist()) - {-1})
    stack, extent, _ = _density_stack(df["x"].to_numpy(), df["y"].to_numpy(), biome, uniq)
    img, _, _ = _dom_rgba(stack, uniq)
    names = {b: BIOME_PRIORITY[b] for b in uniq}
    fig, ax = _fig(dark=True)
    ax.imshow(img, origin="lower", extent=extent, aspect="equal", interpolation="bilinear")
    _labels_text(ax, df, biome, names)
    return _save(fig, name, title)


def render_voronoi(df, labels, names, name, title):
    uniq = sorted(set(labels) - {-1})
    cent = np.array([[df["x"][labels == c].median(), df["y"][labels == c].median()] for c in uniq])
    x, y = df["x"].to_numpy(), df["y"].to_numpy()
    pad = 0.5
    ex = np.linspace(x.min() - pad, x.max() + pad, BINS)
    ey = np.linspace(y.min() - pad, y.max() + pad, BINS)
    Xg, Yg = np.meshgrid(ex, ey)
    P = np.stack([Xg.ravel(), Yg.ravel()], 1)
    d = np.linalg.norm(P[:, None, :] - cent[None, :, :], axis=2)
    dom = d.argmin(1).reshape(BINS, BINS)
    # clip theo mật độ để không tô ra vùng trống
    H, _, _ = np.histogram2d(x, y, bins=[np.append(ex, ex[-1] + (ex[1] - ex[0])),
                                          np.append(ey, ey[-1] + (ey[1] - ey[0]))])
    total = gaussian_filter(H.T, SIGMA)
    img = np.zeros((BINS, BINS, 4))
    for i in range(len(uniq)):
        img[dom == i, :3] = PALETTE[uniq[i] % len(PALETTE)]
    img[..., 3] = (total > np.percentile(total[total > 0], 30)) * 0.85
    fig, ax = _fig(dark=True)
    ax.imshow(img, origin="lower", extent=[ex[0], ex[-1], ey[0], ey[-1]], aspect="equal",
              interpolation="nearest")
    _labels_text(ax, df, labels, names)
    return _save(fig, name, title)


def render_hexbin(df, labels, names, name, title):
    uniq = sorted(set(labels) - {-1})
    remap = {c: i for i, c in enumerate(uniq)}
    valid = np.isin(labels, uniq)
    cidx = np.array([remap.get(l, 0) for l in labels[valid]])
    cmap = mcolors.ListedColormap([PALETTE[c % len(PALETTE)] for c in uniq])
    fig, ax = _fig(dark=True)
    ax.hexbin(df["x"].to_numpy()[valid], df["y"].to_numpy()[valid], C=cidx, gridsize=46,
              reduce_C_function=lambda a: Counter(a).most_common(1)[0][0],
              cmap=cmap, vmin=0, vmax=len(uniq) - 1, mincnt=1, linewidths=0.2, edgecolors="#0b1020")
    _labels_text(ax, df, labels, names)
    return _save(fig, name, title)


def render_contour_topo(df, labels, names, name, title):
    x, y = df["x"].to_numpy(), df["y"].to_numpy()
    pad = 0.5
    ex = np.linspace(x.min() - pad, x.max() + pad, BINS + 1)
    ey = np.linspace(y.min() - pad, y.max() + pad, BINS + 1)
    H, _, _ = np.histogram2d(x, y, bins=[ex, ey])
    Hs = gaussian_filter(H.T, SIGMA)
    cx = 0.5 * (ex[:-1] + ex[1:])
    cy = 0.5 * (ey[:-1] + ey[1:])
    Xg, Yg = np.meshgrid(cx, cy)
    fig, ax = _fig(dark=False)
    ax.contourf(Xg, Yg, Hs, levels=12, cmap="YlGnBu")
    ax.contour(Xg, Yg, Hs, levels=12, colors="#2a4d69", linewidths=0.4, alpha=0.6)
    _labels_text(ax, df, labels, names, color="black")
    return _save(fig, name, title)


# ============================ MONTAGE ============================
def montage(paths, out_name, cols, title):
    from PIL import Image
    imgs = [Image.open(p).convert("RGB") for p in paths]
    w = 900
    thumbs = [im.resize((w, int(im.height * w / im.width))) for im in imgs]
    rows = (len(thumbs) + cols - 1) // cols
    ch = max(t.height for t in thumbs)
    canvas = Image.new("RGB", (cols * w, rows * ch), "white")
    for i, t in enumerate(thumbs):
        canvas.paste(t, ((i % cols) * w, (i // cols) * ch))
    p = OUT / f"{out_name}.png"
    canvas.save(p)
    print(f"montage {out_name}: {len(paths)} ảnh -> {p.name}")
    return p


# ============================ MAIN ============================
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df, X = load_all()

    # ---------- GROUP A: clustering (coloring=kde_fill, naming=lift) ----------
    print("\n== GROUP A: clustering ==")
    clusterings = {}
    for k in (12, 16, 20, 24, 32):
        clusterings[f"kmeans128_k{k}"] = clus_kmeans128(X, k)
    clusterings["gmm128_k20"] = clus_gmm128(X, 20)
    clusterings["agglo128_k20"] = clus_agglo128(X, 20)
    clusterings["hdbscan128"] = clus_hdbscan128(X)
    clusterings["kmeans2d_k20"] = clus_kmeans2d(df, 20)
    clusterings["hdbscan2d"] = clus_hdbscan2d(df)

    a_paths = []
    for key, (labels, _resp) in clusterings.items():
        nc = len(set(labels) - {-1})
        print(f"{key}: {nc} cụm" + (f", {(labels == -1).sum()} noise" if -1 in labels else ""))
        names = name_lift(df, labels)
        a_paths.append(render_kde_fill(df, labels, names, f"clus_{key}",
                                       f"clustering={key} ({nc} cụm) · name=lift · color=kde_fill"))

    # base clustering dùng chung cho group B & C
    base_labels, _ = clusterings["kmeans128_k20"]
    _, gmm_resp = clusterings["gmm128_k20"]

    # ---------- GROUP B: naming (clustering=kmeans128_k20, coloring=kde_fill) ----------
    print("\n== GROUP B: naming ==")
    b_paths = []
    for nm_key, fn in [("lift", name_lift), ("tfidf", name_tfidf),
                       ("exemplar", name_exemplar), ("demo_theme", name_demo_theme)]:
        names = fn(df, base_labels)
        b_paths.append(render_kde_fill(df, base_labels, names, f"name_{nm_key}",
                                       f"clustering=kmeans128_k20 · name={nm_key} · color=kde_fill"))

    # ---------- GROUP C: coloring (clustering=kmeans128_k20, naming=lift) ----------
    print("\n== GROUP C: coloring ==")
    names = name_lift(df, base_labels)
    c_paths = []
    c_paths.append(render_hardscatter(df, base_labels, names, "col_hardscatter",
                                      "color=hardscatter (baseline lốm đốm)"))
    c_paths.append(render_gmm_blend(df, gmm_resp, name_lift(df, gmm_resp.argmax(1)), "col_gmm_blend",
                                    "color=gmm_blend (watercolor mềm, không ranh giới)"))
    c_paths.append(render_kde_fill(df, base_labels, names, "col_kde_fill",
                                   "color=kde_fill (lãnh thổ contiguous, biên mềm)"))
    c_paths.append(render_kde_boundary(df, base_labels, names, "col_kde_boundary",
                                       "color=kde_boundary (lãnh thổ + đường biên)"))
    c_paths.append(render_nebula(df, base_labels, names, "col_nebula",
                                 "color=nebula (mật độ magma, chỉ nhãn)"))
    c_paths.append(render_position(df, base_labels, names, "col_position",
                                   "color=position (gradient địa lý theo vị trí 2D)"))
    c_paths.append(render_knn_genre(df, base_labels, names, "col_knn_genre",
                                    "color=knn_genre (lãnh thổ theo biome genre/theme)"))
    c_paths.append(render_voronoi(df, base_labels, names, "col_voronoi",
                                  "color=voronoi (đa giác lãnh thổ cứng)"))
    c_paths.append(render_hexbin(df, base_labels, names, "col_hexbin",
                                 "color=hexbin (ô lục giác cụm trội)"))
    c_paths.append(render_contour_topo(df, base_labels, names, "col_contour_topo",
                                       "color=contour_topo (bản đồ địa hình isoline)"))

    # ---------- MONTAGES ----------
    print("\n== MONTAGE ==")
    montage(a_paths, "montage_A_clustering", 3, "A")
    montage(b_paths, "montage_B_naming", 2, "B")
    montage(c_paths, "montage_C_coloring", 3, "C")
    print(f"\nXONG. Xem {OUT}/")


if __name__ == "__main__":
    main()
