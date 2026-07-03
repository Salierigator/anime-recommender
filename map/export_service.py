"""export_service.py — export map artifacts cho service (CLI, chạy LOCAL, không cần TF).

Đọc map/outputs/ (đã chạy đủ: build_base → project pumap2d (Colab) → cluster kmeans k=28)
-> ghi `map/outputs/service/` (export cho web — service ĐỌC folder này; xem CONTRACT.md sinh kèm):
  map_points.parquet   mal_id/title/x/y/label/popularity/is_cold — 21k điểm SFW cho GET /api/map
  map_clusters.parquet label/name/size/examples/cx/cy — 28 cụm (nhãn + hover)
  pumap_encoder.npz    weights encoder ParametricUMAP (MLP 128→100→100→100→2 relu) trích từ
                       encoder.keras (zip chứa h5, đọc bằng h5py) → service forward numpy,
                       "you are here" KHÔNG cần TensorFlow
  territory.png        nền kde_boundary sạch (fill + biên, KHÔNG nhãn/title) cho frontend overlay
  map_meta.json        n_points/k/extent/sha256(item_vectors)/encoder — service check sync

Verify BẮT BUỘC trước khi ghi: forward numpy(vectors_real) ≈ coords đã fit (pumap fit_transform
trả đúng output encoder trên tập train) — lệch > 1e-3 nghĩa là trích weights sai → abort.

    venv/bin/python map/export_service.py
"""
from __future__ import annotations

import hashlib
import io
import json
import zipfile
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import _common as C
import viz

MAP_DIR = C.OUTPUTS / "service"
TERR_BINS, TERR_SIGMA = 960, 12.0     # = look chốt 480/6.0 (viz.render_territory) scale 2× pixel


def extract_encoder_weights() -> list[tuple[np.ndarray, np.ndarray]]:
    """encoder.keras (zip) -> [(W,b)] theo thứ tự layer. Assert chain 128 → ... → 2."""
    import h5py

    z = zipfile.ZipFile(C.reducer_path("pumap2d") / "encoder.keras")
    layers = []
    with h5py.File(io.BytesIO(z.read("model.weights.h5")), "r") as f:
        for name in ("dense", "dense_1", "dense_2", "dense_3"):
            layers.append((f[f"layers/{name}/vars/0"][()], f[f"layers/{name}/vars/1"][()]))
    assert layers[0][0].shape[0] == 128 and layers[-1][0].shape[1] == 2, \
        [w.shape for w, _ in layers]
    for (w1, _), (w2, _) in zip(layers, layers[1:]):
        assert w1.shape[1] == w2.shape[0], "layer shapes không nối nhau"
    return layers


def forward(layers, x: np.ndarray) -> np.ndarray:
    """Forward MLP relu×3 + linear cuối — bản numpy của encoder.predict."""
    h = x.astype(np.float32)
    for w, b in layers[:-1]:
        h = np.maximum(h @ w + b, 0.0)
    w, b = layers[-1]
    return h @ w + b


def render_territory_asset(pts: pd.DataFrame) -> tuple[list[float], str]:
    """PNG nền kde_boundary KHÔNG chữ (fill + biên trắng pixel, alpha rìa → 0) — frontend
    overlay điểm bằng extent trả về. Cùng công thức viz.render_territory (kde_dominant +
    territory_rgba), biên vẽ pixel-diff thay contour để khỏi dựng figure matplotlib."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x, y, labels = pts["x"].to_numpy(), pts["y"].to_numpy(), pts["label"].to_numpy()
    uniq = sorted(set(labels.tolist()))
    ex, ey, total, dom = viz.kde_dominant(x, y, labels, uniq, bins=TERR_BINS, sigma=TERR_SIGMA)
    img = viz.territory_rgba(uniq, total, dom)

    thr = np.percentile(total[total > 0], 40)
    fld = np.where(total > thr, dom + 1, 0)                # 0 = "biển" dưới ngưỡng mật độ
    edge = np.zeros(fld.shape, dtype=bool)                 # biên đánh dấu CẢ 2 phía -> nét liền
    dv = fld[1:, :] != fld[:-1, :]
    edge[1:, :] |= dv
    edge[:-1, :] |= dv
    dh = fld[:, 1:] != fld[:, :-1]
    edge[:, 1:] |= dh
    edge[:, :-1] |= dh
    img[edge, :3] = img[edge, :3] * 0.2 + 0.8              # biên trắng mờ như contour chốt
    img[edge, 3] = np.maximum(img[edge, 3], 0.8)

    out = MAP_DIR / "territory.png"
    plt.imsave(out, np.clip(img, 0.0, 1.0), origin="lower")
    return [float(ex[0]), float(ex[-1]), float(ey[0]), float(ey[-1])], out.name


def main() -> None:
    base = pd.read_parquet(C.OUTPUTS / "base.parquet")
    vectors = np.load(C.OUTPUTS / "vectors_real.npy")
    coords = pd.read_parquet(C.OUTPUTS / "coords_pumap2d.parquet")
    clus = pd.read_parquet(C.OUTPUTS / "clusters_kmeans.parquet")
    names = pd.read_parquet(C.OUTPUTS / "cluster_names_kmeans.parquet")
    assert (coords["anime_idx"].to_numpy() == base["anime_idx"].to_numpy()).all(), \
        "coords lệch hàng base — chạy lại build_base + project"

    # 1) encoder npz + verify forward ≈ coords (gate trích weights đúng)
    layers = extract_encoder_weights()
    diff = np.abs(forward(layers, vectors) - coords[["x", "y"]].to_numpy())
    print(f"verify encoder: max|forward - coords| = {diff.max():.2e} (mean {diff.mean():.2e})")
    assert diff.max() < 1e-3, "forward numpy lệch coords — trích weights sai, KHÔNG ghi export"

    MAP_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(MAP_DIR / "pumap_encoder.npz",
             **{f"W{i}": w for i, (w, _) in enumerate(layers)},
             **{f"b{i}": b for i, (_, b) in enumerate(layers)})

    # 2) points: coords + metadata hiển thị (mal_id để join với recs; popularity cho LOD frontend)
    pts = coords.merge(base[["anime_idx", "mal_id", "title", "popularity", "is_cold"]],
                       on="anime_idx").merge(clus, on="anime_idx")
    assert len(pts) == len(base) and pts["label"].notna().all()
    pts["title"] = pts["title"].fillna(pts["mal_id"].astype(str))
    pts["popularity"] = pd.to_numeric(pts["popularity"], errors="coerce") \
        .fillna(999_999).astype(np.int32)
    pts = pts.astype({"x": np.float32, "y": np.float32, "label": np.int16})
    pts[["mal_id", "title", "x", "y", "label", "popularity", "is_cold"]] \
        .to_parquet(MAP_DIR / "map_points.parquet")

    # 3) clusters: tên log-odds + centroid (median member — chỗ frontend đặt nhãn)
    cent = pts.groupby("label")[["x", "y"]].median().astype(np.float32) \
        .rename(columns={"x": "cx", "y": "cy"}).reset_index()
    names.merge(cent, on="label").to_parquet(MAP_DIR / "map_clusters.parquet")

    # 4) nền territory + meta (sha item_vectors = fingerprint chống map lệch phiên bản vectors)
    extent, terr_name = render_territory_asset(pts)
    sha = hashlib.sha256((C.ARTIFACTS / "item_vectors.npy").read_bytes()).hexdigest()
    meta = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "method": "pumap2d", "k": int(pts["label"].nunique()), "n_points": int(len(pts)),
        "sfw_only": True, "extent": extent,
        "territory": {"file": terr_name, "bins": TERR_BINS, "sigma": TERR_SIGMA,
                      "bg": "#0b1020"},
        "encoder": {"file": "pumap_encoder.npz",
                    "arch": "128->100->100->100->2 relu (ParametricUMAP encoder)",
                    "max_recon_diff": float(diff.max())},
        "item_vectors_sha256": sha,
    }
    (MAP_DIR / "map_meta.json").write_text(json.dumps(meta, indent=2))

    (MAP_DIR / "CONTRACT.md").write_text(f"""# map/outputs/service/ — CONTRACT

Tự sinh bởi `map/export_service.py` — không sửa tay; re-run export để cập nhật.
Đây là output của map/ dành cho web (`artifacts/` thuần retriever/ranker/service, map KHÔNG
đụng). Consumer: service (`app/ml/anime_map.py` → `GET /api/map`) — chỉ ĐỌC folder này.

- Generated: {meta['generated']}
- Nguồn: `map/outputs/` (pipeline: build_base → project pumap2d n50/d0.8 (Colab) → cluster
  kmeans k={meta['k']} + naming log-odds → export_service). Chốt hiển thị: `map/README.md`.
- `artifacts/item_vectors.npy` sha256: `{sha[:16]}…` (đầy đủ trong map_meta.json). Service so
  khớp lúc load — LỆCH nghĩa là retriever đã re-export mà map chưa re-fit → service TẮT map
  (degrade), recommend vẫn chạy bình thường.

## Files

### `map_points.parquet` — {meta['n_points']:,} anime (SFW-only, loại hentai khớp nsfw serving)
- `mal_id int64` (join với recs/posters), `title str`, `x/y float32` (toạ độ pumap2d),
  `label int16` (cụm KMeans k={meta['k']}), `popularity int32` (rank MAL, NHỎ = phổ biến;
  999999 = thiếu — dùng cho LOD frontend), `is_cold bool` (item cold, kênh "anime mới").

### `map_clusters.parquet` — {meta['k']} cụm
- `label, name` (log-odds 2-tag), `size`, `examples` (3 title phổ biến, phân cách " · "),
  `cx/cy float32` (median toạ độ member — chỗ đặt nhãn text).

### `pumap_encoder.npz` — encoder ParametricUMAP, MLP `{meta['encoder']['arch']}`
- key `W0..W3` [in,out] + `b0..b3`. Forward numpy: `h=relu(x@Wi+bi)` ×3 → `xy=h@W3+b3`
  (= `encoder.predict`, verified max diff {meta['encoder']['max_recon_diff']:.1e} trên toàn catalog SFW).
- Input: vector 128-d L2-norm CÙNG không gian two-tower — item vector HOẶC user vector U
  (từ user_tower) → "you are here" không cần TensorFlow.

### `territory.png` — nền bản đồ kde_boundary (fill cụm áp đảo + biên trắng, KHÔNG chữ)
- {TERR_BINS}×{TERR_BINS}, lưu `origin="lower"` ⇒ mép DƯỚI ảnh = y min. Overlay: map (x,y)
  tuyến tính vào `extent` = [x0, x1, y0, y1] trong map_meta.json. Alpha rìa → 0 (đặt trên nền
  tối `{meta['territory']['bg']}` như bản chốt).

### `map_meta.json` — generated/k/n_points/extent/territory/encoder/sha — service + frontend đọc.

## Sync (BẮT BUỘC đọc khi retriever re-export)
Coords + encoder là HÀM của `item_vectors.npy`. Vectors đổi ⇒ chạy lại từ đầu:
`build_base` → `project` (Colab) → `cluster --algo kmeans --k {meta['k']}` → `export_service`.
Chỉ đổi cluster/naming (vectors giữ nguyên) ⇒ chỉ cần `cluster` + `export_service`.
""")

    print(f"-> {MAP_DIR}/ : map_points ({len(pts):,}) + map_clusters ({len(names)}) "
          f"+ pumap_encoder.npz + {terr_name} + map_meta.json + CONTRACT.md")


if __name__ == "__main__":
    main()
