"""explore_viz_v2.py — sweep vòng 2 (chốt từ explore_viz.py): sweep K cao + so naming.

User đã chọn: naming=tfidf (thích), color=kde_boundary + hexbin. Còn phân vân K → sweep K cao để chọn.
Script này TÁI DÙNG lõi của explore_viz.py (load/cluster/render/montage), chỉ thêm:
  - K-sweep kmeans128 K ∈ {20,24,28,32,40,48,60}, render mỗi K bằng kde_boundary + hexbin.
  - So 3 naming: tfidf (baseline) · hybrid (tfidf + phim tiêu biểu) · logodds (fightin'-words, đặc trưng nhất).
Output -> map/outputs/explore/sweep_v2/ (folder riêng). Local, projection giữ nguyên n50_d0.8.

    venv/bin/python map/explore_viz_v2.py
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import explore_viz as E  # tái dùng loaders + clus_* + name_tfidf + render_* + montage

# palette rộng hơn (tới 60 màu) cho K cao — render đọc E.PALETTE global nên set đè là đủ
E.PALETTE = np.array(list(plt.get_cmap("tab20").colors) + list(plt.get_cmap("tab20b").colors)
                     + list(plt.get_cmap("tab20c").colors))[:, :3]
E.OUT = E.C.OUTPUTS / "explore" / "sweep_v2"

KS = [20, 24, 28, 32, 40, 48, 60]
K_NAMING = 24            # K cố định khi so 3 cách naming
LABEL_TOP = 18           # chỉ ghi nhãn top-N cụm lớn nhất (tránh chữ đè khi K cao)


def top_names(labels, names, n=LABEL_TOP):
    """Giữ nhãn cho n cụm lớn nhất → map K cao vẫn đọc được."""
    order = [lab for lab, _ in Counter(labels).most_common() if lab in names]
    return {lab: names[lab] for lab in order[:n]}


# ---------------- naming mới ----------------
def name_hybrid(df, labels):
    """Danh mục tfidf + 1 phim tiêu biểu (phổ biến nhất cụm) — 2 dòng."""
    tf = E.name_tfidf(df, labels)
    out = {}
    for lab in sorted(set(labels) - {-1}):
        title = str(df[labels == lab].sort_values("popularity")["title"].iloc[0])
        title = title if len(title) <= 22 else title[:20] + "…"
        out[lab] = f"{tf[lab]}\n{title}"
    return out


def name_logodds(df, labels, a0=100.0, min_support=0.05, min_count=3):
    """Tag đặc trưng theo log-odds-ratio + prior Dirichlet (Monroe 2008 'fightin words').

    z = delta_logodds / sqrt(var) — chọn tag phân biệt cụm này với phần còn lại, kiểm soát base-rate +
    phương sai (tag hiếm không bị thổi phồng). Lấy top-2 z, có lọc support tối thiểu trong cụm."""
    tags = E._tagsets(df)
    gc = Counter(t for s in tags for t in s)                # count toàn cục
    total = sum(gc.values())
    aw = {t: gc[t] / total * a0 for t in gc}                # prior tỉ lệ tần suất nền
    names = {}
    for lab in sorted(set(labels) - {-1}):
        idx = np.where(labels == lab)[0]
        n_cluster = len(idx)
        yi = Counter(t for i in idx for t in tags[i])
        ni = sum(yi.values())
        n_rest = total - ni
        scored = []
        for t, c in yi.items():
            if c < min_count or c / n_cluster < min_support:
                continue
            yiw, yrw = yi[t], gc[t] - yi[t]
            term_i = np.log(yiw + aw[t]) - np.log(ni + a0 - yiw - aw[t])
            term_r = np.log(yrw + aw[t]) - np.log(n_rest + a0 - yrw - aw[t])
            var = 1.0 / (yiw + aw[t]) + 1.0 / (yrw + aw[t])
            scored.append((( term_i - term_r) / np.sqrt(var), t))
        scored.sort(reverse=True)
        names[lab] = "·".join(t for _, t in scored[:2]) if scored \
            else df.iloc[idx]["primary_genre"].mode().iloc[0]
    return E._dedupe(names)


def main():
    E.OUT.mkdir(parents=True, exist_ok=True)
    df, X = E.load_all()

    # ---------- K-SWEEP (naming=tfidf; kde_boundary + hexbin) ----------
    print("== K-SWEEP kmeans128 (tfidf) ==")
    kb_paths, hx_paths = [], []
    labels_cache = {}
    for k in KS:
        labels, _ = E.clus_kmeans128(X, k)
        labels_cache[k] = labels
        names = top_names(labels, E.name_tfidf(df, labels))
        print(f"k{k}: {len(set(labels))} cụm")
        kb_paths.append(E.render_kde_boundary(df, labels, names, f"kb_k{k}",
                                              f"kmeans128 k{k} · tfidf · kde_boundary"))
        hx_paths.append(E.render_hexbin(df, labels, names, f"hex_k{k}",
                                        f"kmeans128 k{k} · tfidf · hexbin"))
    E.montage(kb_paths, "montage_Ksweep_kde_boundary", 3, "")
    E.montage(hx_paths, "montage_Ksweep_hexbin", 3, "")

    # ---------- NAMING (K=K_NAMING; kde_boundary) ----------
    print(f"\n== NAMING so sánh (k{K_NAMING}) ==")
    labels = labels_cache[K_NAMING] if K_NAMING in labels_cache else E.clus_kmeans128(X, K_NAMING)[0]
    nm_paths = []
    for key, names in [("tfidf", E.name_tfidf(df, labels)),
                       ("hybrid", name_hybrid(df, labels)),
                       ("logodds", name_logodds(df, labels))]:
        nm_paths.append(E.render_kde_boundary(df, labels, top_names(labels, names),
                                              f"name_{key}_k{K_NAMING}",
                                              f"kmeans128 k{K_NAMING} · name={key} · kde_boundary"))
    E.montage(nm_paths, "montage_naming_v2", 3, "")
    print(f"\nXONG. Xem {E.OUT}/")


if __name__ == "__main__":
    main()
