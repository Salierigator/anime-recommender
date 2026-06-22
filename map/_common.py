"""_common.py — helper dùng chung cho map/ (KHÔNG phải CLI).

Firewall: chỉ ĐỌC `artifacts/` + `cleaned-data/details.csv`. Không sửa artifacts.

Quy ước:
- item_vectors.npy[i] == anime_idx i (0=PAD, 1=OOV neutral, real >=2). Map chỉ vẽ real anime
  (mal_id != -1); cờ is_cold đánh dấu row content-only (id->OOV lúc export).
- Coords lưu parquet (anime_idx, x, y) 2D; reducer = ParametricUMAP (pumap2d) lưu thư mục.
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"
CLEANED = ROOT / "cleaned-data"
OUTPUTS = Path(__file__).resolve().parent / "outputs"

DETAIL_COLS = ["mal_id", "title", "type", "genres", "themes", "studios", "popularity", "start_date"]


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

    base_df cột: anime_idx, mal_id, title, type, primary_genre, genres_list, themes_list,
    popularity, start_year, is_cold. Loại PAD/OOV (mal_id == -1)."""
    import pyarrow.parquet as pq

    vectors = np.load(ARTIFACTS / "item_vectors.npy")
    idx = pq.read_table(ARTIFACTS / "item_index.parquet").to_pandas()
    idx = idx[idx["mal_id"] != -1].reset_index(drop=True)            # bỏ PAD/OOV

    det = pd.read_csv(CLEANED / "details.csv", usecols=DETAIL_COLS)
    base = idx.merge(det, on="mal_id", how="left").reset_index(drop=True)

    base["genres_list"] = base["genres"].map(_parse_list)
    base["themes_list"] = base["themes"].map(_parse_list)
    base["primary_genre"] = base["genres_list"].map(lambda L: L[0] if L else "Unknown")
    base["start_year"] = pd.to_datetime(base["start_date"], errors="coerce").dt.year
    base = base.drop(columns=["genres", "themes", "studios", "start_date"])

    vectors_real = vectors[base["anime_idx"].to_numpy()]
    return base, vectors_real.astype(np.float32)


def load_base() -> tuple[pd.DataFrame, np.ndarray]:
    """Đọc base đã build (build_base.py). base_df + vectors_real căn hàng theo row."""
    base = pd.read_parquet(OUTPUTS / "base.parquet")
    vectors = np.load(OUTPUTS / "vectors_real.npy")
    return base, vectors


# ---- coords IO (2D) ----
def coords_path(method: str) -> Path:
    return OUTPUTS / f"coords_{method}.parquet"


def save_coords(method: str, anime_idx: np.ndarray, emb: np.ndarray) -> Path:
    """emb [n,2]. Ghi parquet (anime_idx, x, y)."""
    df = pd.DataFrame({"anime_idx": anime_idx.astype(np.int32), "x": emb[:, 0], "y": emb[:, 1]})
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    df.to_parquet(coords_path(method))
    return coords_path(method)


def load_coords(method: str) -> pd.DataFrame:
    return pd.read_parquet(coords_path(method))


# ---- reducer IO (ParametricUMAP: keras model + pickle trong 1 thư mục) ----
def reducer_path(method: str) -> Path:
    return OUTPUTS / f"reducer_{method}"


def save_reducer(method: str, reducer) -> Path:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    p = reducer_path(method)
    p.mkdir(parents=True, exist_ok=True)           # umap save() KHÔNG tự tạo dir -> phải tạo trước
    reducer.save(str(p))                           # ParametricUMAP.save -> ghi encoder.keras + model.pkl
    return p


def load_reducer(method: str):
    p = reducer_path(method)
    if not p.exists():
        raise FileNotFoundError(f"Chưa fit reducer '{method}' — chạy project.py trước ({p})")
    from umap.parametric_umap import load_ParametricUMAP
    return load_ParametricUMAP(str(p))


def transform_to_coords(method: str, reducer, vecs: np.ndarray) -> np.ndarray:
    """Chiếu vector mới [m,128] qua reducer pumap đã fit -> coords 2D (forward-pass)."""
    return reducer.transform(vecs).astype(np.float32)
