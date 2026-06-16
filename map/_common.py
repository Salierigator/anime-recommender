"""_common.py — helper dùng chung cho map/ (KHÔNG phải CLI).

Firewall: phần item-map + IO chỉ ĐỌC `artifacts/` + `cleaned-data/details.csv`. Genre vocab
lấy từ `retriever/train-data/feature_spec.json` (file nhỏ, đọc được). Không sửa artifacts.

Quy ước:
- item_vectors.npy[i] == anime_idx i (0=PAD, 1=OOV neutral, real >=2). Map chỉ vẽ real anime
  (mal_id != -1); cờ is_cold đánh dấu row content-only (id->OOV lúc export).
- Coords lưu parquet (anime_idx, x, y[, z]); reducer lưu method-aware (joblib | ParametricUMAP.save).
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"
CLEANED = ROOT / "cleaned-data"
TRAIN_DATA = ROOT / "retriever" / "train-data"
OUTPUTS = Path(__file__).resolve().parent / "outputs"

DETAIL_COLS = ["mal_id", "title", "type", "genres", "studios", "popularity", "start_date"]


def feature_spec() -> dict:
    with open(TRAIN_DATA / "feature_spec.json") as f:
        return json.load(f)


def genre_vocab() -> dict:
    """{genre_name: index} (21 genre) từ feature_spec — vocab chuẩn của ItemTower."""
    return feature_spec()["item_features"]["genres"]["map"]


def _parse_list(v):
    """'[]'/NaN -> []; "['A','B']" -> ['A','B']. Khớp parse_list ở data_prep/01."""
    if pd.isna(v):
        return []
    try:
        out = ast.literal_eval(v)
        return out if isinstance(out, list) else []
    except (ValueError, SyntaxError):
        return []


def build_base_table() -> tuple[pd.DataFrame, np.ndarray]:
    """Join item_vectors + item_index + details -> (base_df, vectors_real) căn hàng theo row.

    base_df cột: anime_idx, mal_id, title, type, primary_genre, genres_list, popularity,
    start_year, is_cold. Loại PAD/OOV (mal_id == -1)."""
    import pyarrow.parquet as pq

    vectors = np.load(ARTIFACTS / "item_vectors.npy")
    idx = pq.read_table(ARTIFACTS / "item_index.parquet").to_pandas()
    idx = idx[idx["mal_id"] != -1].reset_index(drop=True)            # bỏ PAD/OOV

    det = pd.read_csv(CLEANED / "details.csv", usecols=DETAIL_COLS)
    base = idx.merge(det, on="mal_id", how="left").reset_index(drop=True)

    base["genres_list"] = base["genres"].map(_parse_list)
    base["primary_genre"] = base["genres_list"].map(lambda L: L[0] if L else "Unknown")
    base["start_year"] = pd.to_datetime(base["start_date"], errors="coerce").dt.year
    base = base.drop(columns=["genres", "studios", "start_date"])

    vectors_real = vectors[base["anime_idx"].to_numpy()]
    return base, vectors_real.astype(np.float32)


def load_base() -> tuple[pd.DataFrame, np.ndarray]:
    """Đọc base đã build (build_base.py). base_df + vectors_real căn hàng theo row."""
    base = pd.read_parquet(OUTPUTS / "base.parquet")
    vectors = np.load(OUTPUTS / "vectors_real.npy")
    return base, vectors


def lonlat_to_xyz(emb: np.ndarray) -> np.ndarray:
    """UMAP output_metric='haversine' -> [n,2] (θ polar, φ azimuth) -> [n,3] đơn vị mặt cầu.
    Quy ước khớp ví dụ sphere của UMAP: x=sinθcosφ, y=sinθsinφ, z=cosθ."""
    theta, phi = emb[:, 0], emb[:, 1]
    return np.stack([np.sin(theta) * np.cos(phi),
                     np.sin(theta) * np.sin(phi),
                     np.cos(theta)], axis=1).astype(np.float32)


# ---- coords IO ----
def coords_path(method: str) -> Path:
    return OUTPUTS / f"coords_{method}.parquet"


def save_coords(method: str, anime_idx: np.ndarray, emb: np.ndarray) -> Path:
    """emb [n,2] hoặc [n,3]. Ghi parquet (anime_idx, x, y[, z])."""
    cols = {"anime_idx": anime_idx.astype(np.int32), "x": emb[:, 0], "y": emb[:, 1]}
    if emb.shape[1] == 3:
        cols["z"] = emb[:, 2]
    df = pd.DataFrame(cols)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    df.to_parquet(coords_path(method))
    return coords_path(method)


def load_coords(method: str) -> pd.DataFrame:
    return pd.read_parquet(coords_path(method))


# ---- reducer IO (method-aware: ParametricUMAP có keras model riêng) ----
def reducer_path(method: str) -> Path:
    return OUTPUTS / (f"reducer_{method}" if method == "pumap" else f"reducer_{method}.pkl")


def save_reducer(method: str, reducer) -> Path:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    p = reducer_path(method)
    if method == "pumap":
        reducer.save(str(p))                       # ParametricUMAP.save -> thư mục (keras + pickle)
    else:
        import joblib
        joblib.dump(reducer, p)
    return p


def load_reducer(method: str):
    p = reducer_path(method)
    if not p.exists():
        raise FileNotFoundError(f"Chưa fit reducer cho '{method}' — chạy project.py --method {method} trước ({p})")
    if method == "pumap":
        from umap.parametric_umap import load_ParametricUMAP
        return load_ParametricUMAP(str(p))
    import joblib
    return joblib.load(p)


def to_display_coords(method: str, emb: np.ndarray) -> np.ndarray:
    """Hậu xử lý output reducer -> coords hiển thị (dùng chung fit và transform để nhất quán).
    - haversine sphere (umap/densmap): [n,2] lon/lat -> xyz mặt cầu.
    - pca_sphere: [n,3] -> L2-norm lên mặt cầu.
    - còn lại: giữ nguyên (2D)."""
    if method in ("umap_sphere", "densmap_sphere"):
        return lonlat_to_xyz(emb)
    if method == "pca_sphere":
        return (emb / np.linalg.norm(emb, axis=1, keepdims=True).clip(min=1e-9)).astype(np.float32)
    return emb.astype(np.float32)


def transform_to_coords(method: str, reducer, vecs: np.ndarray) -> np.ndarray:
    """Chiếu vector mới [m,128] qua reducer đã fit -> coords hiển thị (cùng hậu xử lý lúc fit)."""
    return to_display_coords(method, reducer.transform(vecs))
