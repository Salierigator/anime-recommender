"""encode_genre.py — đặt điểm genre lên bản đồ (CLI): centroid + ItemTower probe.

Stack RETRIEVER (TÁCH process khỏi encode_user.py — tránh đụng 'config.py' 2 mảng). Phần probe
load retriever/checkpoints/best.pt + code model retriever (vượt ranh giới train/serve — chấp nhận
vì map/ là công cụ phân tích offline, không phải serving path).

  - centroid: mean item_vectors của anime có genre G, L2-norm (artifacts-only, vị trí THẬT).
  - probe: ItemTower forward trên item tổng hợp (chỉ bật genre G, id->OOV) — "hướng genre thuần".

    venv/bin/python map/encode_genre.py --method umap2d --genre Action
    venv/bin/python map/encode_genre.py --method umap_sphere --all

Ghi outputs/overlay_genre_<G>.parquet (hoặc overlay_genre_all.parquet) -> viz.py --overlay.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
import torch

import _common as C

sys.path.insert(0, str(C.ROOT / "retriever"))
import export as rexport                  # noqa: E402  (chèn retriever/src vào path + import config/data/model)
from data import ItemTable                # noqa: E402
from model import ItemBatch               # noqa: E402


def load_model(device: str = "cpu"):
    ckpt_path = C.ROOT / "retriever" / "checkpoints" / "best.pt"
    print(f"Loading {ckpt_path} ...")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model, spec = rexport.build_model(ckpt, device)   # tái dùng cfg-shim + reconcile của export
    return model, spec


@torch.no_grad()
def probe_vector(model, spec, genre: str, device: str = "cpu") -> np.ndarray:
    """ItemTower forward trên item tổng hợp: chỉ bật genre G (+present), mọi feature khác neutral,
    id->OOV (content-only). Trả vector [128] L2-norm."""
    gf = spec["item_features"]["genres"]
    gmap = gf["map"]
    if genre not in gmap:
        raise SystemExit(f"genre '{genre}' không có trong vocab. Có: {sorted(gmap)}")
    genres = torch.zeros(1, gf["width"])
    genres[0, gmap[genre]] = 1.0
    genres[0, gf["present_index"]] = 1.0                  # cờ "có genre" (khớp encode_multihot 01)
    themes = torch.zeros(1, spec["item_features"]["themes"]["width"])
    studios = torch.zeros(1, 1, dtype=torch.long)          # [[0]] = empty-token (học được)
    cat = {col: torch.zeros(1, dtype=torch.long) for col in ItemTable.CAT_COLS}  # 0 = PAD/neutral
    # synopsis: item tổng hợp không có synopsis -> nhánh low-info (dùng param no_synopsis học được).
    syn = syn_low = None
    if getattr(model.item_tower, "use_synopsis", False):
        raw_dim = model.item_table.synopsis_emb.shape[1]
        syn = torch.zeros(1, raw_dim, device=device)
        syn_low = torch.tensor([True], device=device)
    batch = ItemBatch({k: v.to(device) for k, v in cat.items()},
                      genres.to(device), themes.to(device), studios.to(device), syn, syn_low)
    id_idx = torch.tensor([spec["special_idx"]["OOV"]], device=device)
    return model.item_tower(batch, id_idx).cpu().numpy()[0]


def centroid_vector(base: pd.DataFrame, X: np.ndarray, genre: str) -> np.ndarray | None:
    """Mean item_vectors của anime có genre G (re-norm). None nếu không anime nào."""
    mask = base["genres_list"].map(lambda L: genre in L).to_numpy()
    if not mask.any():
        return None
    c = X[mask].mean(axis=0)
    return (c / np.linalg.norm(c).clip(min=1e-9)).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description="Đặt centroid + probe genre lên map")
    ap.add_argument("--method", required=True)
    ap.add_argument("--genre", help="1 genre (vd Action)")
    ap.add_argument("--all", action="store_true", help="mọi genre trong vocab")
    args = ap.parse_args()

    base, X = C.load_base()
    model, spec = load_model()
    reducer = C.load_reducer(args.method)
    has_z = "z" in C.load_coords(args.method).columns

    genres = sorted(spec["item_features"]["genres"]["map"]) if args.all \
        else ([args.genre] if args.genre else None)
    if genres is None:
        raise SystemExit("Cần --genre G hoặc --all.")

    vecs, meta_rows = [], []           # gom tất cả vector rồi transform 1 lần
    for g in genres:
        cen = centroid_vector(base, X, g)
        prb = probe_vector(model, spec, g)
        if cen is not None:
            cos = float(cen @ prb)
            print(f"{g:<14} cosine(probe, centroid) = {cos:.3f}  (cao = probe trỏ đúng vùng genre)")
            vecs.append(cen); meta_rows.append(("centroid", g))
        else:
            print(f"{g:<14} (không anime nào — bỏ centroid)")
        vecs.append(prb); meta_rows.append(("probe", g))

    coords = C.transform_to_coords(args.method, reducer, np.asarray(vecs, np.float32))
    rows = []
    for (kind, g), c in zip(meta_rows, coords):
        r = {"kind": kind, "label": f"{g} ({kind})", "anime_idx": -1,
             "x": float(c[0]), "y": float(c[1])}
        if has_z:
            r["z"] = float(c[2])
        rows.append(r)

    tag = "all" if args.all else args.genre
    out = C.OUTPUTS / f"overlay_genre_{tag}.parquet"
    pd.DataFrame(rows).to_parquet(out)
    print(f"-> {out}  ({len(rows)} điểm)")


if __name__ == "__main__":
    main()
