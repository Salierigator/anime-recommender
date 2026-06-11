"""features.py — bảng feature item (cleaned-data/details.csv) + lắp ma trận feature ranker.

Dùng chung build_train/build_eval/serve để feature train/eval/serve KHỚP nhau. Deterministic
từ details.csv (cùng input → cùng code categorical); ranker_meta.json lưu thứ tự feature +
index categorical cho service.

Cold policy (no-leak): row `is_cold` (item_index.parquet) bị ép mất các stat trưởng-thành
(mal_score/scored_by/members/favorites/popularity/rank → impute-as-missing + flag) — anime
mới lúc serve không có stats; content (genres/themes/type/...) + episodes/recency giữ nguyên
(metadata công bố từ đầu). Áp NGAY TRONG ItemFeatures.load nên mọi đường dùng đều nhất quán.

Chỉ ĐỌC details.csv (pandas, ~22.8k dòng) + item_index.parquet (firewall-clean).
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

# Thứ tự cột feature CỐ ĐỊNH — cross/user tính per-(user,cand) ở caller (pool/build),
# item numeric/categorical gather theo anime_idx từ ItemFeatures.
FEATURE_NAMES = [
    # cross (user × item)
    "cos_uv", "pool_rank", "hist_cos_max", "hist_cos_mean", "hist_cos_top5_mean",
    "genre_aff", "theme_aff", "genre_overlap", "score_gap",
    # user
    "u_n_rated", "u_mean_score", "u_std_score", "u_account_age", "support_len",
    # item numeric
    "mal_score", "mal_score_missing", "log_scored_by", "log_members", "log_favorites",
    "popularity", "rank", "rank_missing", "episodes", "recency_years",
    # item categorical (LightGBM native)
    "type_code", "source_code", "rating_code", "demo_code", "era_code",
]
CAT_COLS = ["type_code", "source_code", "rating_code", "demo_code", "era_code"]
N_CROSS = 14                                          # số cột đầu tính ở caller
ITEM_COLS = FEATURE_NAMES[N_CROSS:]                   # cột gather theo anime_idx
REF_YEAR = 2024                                       # mốc tính recency / account_age


def _parse_list(s) -> list:
    if not isinstance(s, str) or not s.startswith("["):
        return []
    try:
        return list(ast.literal_eval(s))
    except (ValueError, SyntaxError):
        return []


def _codes(series: pd.Series) -> np.ndarray:
    """Categorical → int code (sorted, NaN/missing → 0). Deterministic từ details.csv."""
    cat = series.astype("category")
    return (cat.cat.codes.to_numpy() + 1).astype(np.int32)   # +1: -1(NaN)→0, hạng đầu→1


def _multihot(lists: list[list], vocab: list[str]) -> np.ndarray:
    pos = {v: i for i, v in enumerate(vocab)}
    out = np.zeros((len(lists), len(vocab)), dtype=np.float32)
    for r, tags in enumerate(lists):
        for t in tags:
            j = pos.get(t)
            if j is not None:
                out[r, j] = 1.0
    return out


class ItemFeatures:
    """Mảng feature item index theo anime_idx (row==anime_idx, khớp item_vectors)."""

    def __init__(self, item: dict, genres: np.ndarray, themes: np.ndarray, is_cold: np.ndarray):
        self.item = item                              # name -> [N] array
        self.genres = genres                          # [N, G] multihot
        self.themes = themes                          # [N, T] multihot
        self.is_cold = is_cold                        # [N] bool

    @classmethod
    def load(cls, artifacts: Path, cleaned: Path) -> "ItemFeatures":
        idx = pl.read_parquet(artifacts / "item_index.parquet").to_pandas()  # anime_idx, mal_id, is_cold
        det = pd.read_csv(cleaned / "details.csv", usecols=[
            "mal_id", "type", "source", "rating", "demographics", "score", "scored_by",
            "members", "favorites", "popularity", "rank", "episodes", "start_date",
            "genres", "themes"])
        df = idx.merge(det, on="mal_id", how="left").sort_values("anime_idx").reset_index(drop=True)
        cold = df["is_cold"].to_numpy(bool)

        # cold policy: stat trưởng-thành → missing TRƯỚC khi impute (median tính trên warm)
        df.loc[cold, ["score", "popularity", "rank"]] = np.nan
        df.loc[cold, ["scored_by", "members", "favorites"]] = 0

        year = pd.to_datetime(df["start_date"], errors="coerce", utc=True).dt.year
        era = pd.cut(year, bins=[-np.inf, 1989, 1999, 2009, 2017, np.inf], labels=False)

        item = {
            "mal_score": df["score"].fillna(df["score"].median()).to_numpy(np.float32),
            "mal_score_missing": df["score"].isna().to_numpy(np.float32),
            "log_scored_by": np.log1p(df["scored_by"].fillna(0).to_numpy(np.float32)),
            "log_members": np.log1p(df["members"].fillna(0).to_numpy(np.float32)),
            "log_favorites": np.log1p(df["favorites"].fillna(0).to_numpy(np.float32)),
            "popularity": df["popularity"].fillna(df["popularity"].median()).to_numpy(np.float32),
            "rank": df["rank"].fillna(df["rank"].median()).to_numpy(np.float32),
            "rank_missing": df["rank"].isna().to_numpy(np.float32),
            "episodes": df["episodes"].fillna(df["episodes"].median()).to_numpy(np.float32),
            "recency_years": (REF_YEAR - year.fillna(year.median())).to_numpy(np.float32),
            "type_code": _codes(df["type"]),
            "source_code": _codes(df["source"]),
            "rating_code": _codes(df["rating"]),
            "demo_code": _codes(df["demographics"].where(df["demographics"] != "[]")),
            "era_code": np.nan_to_num(era.to_numpy(np.float32), nan=-1).astype(np.int32) + 1,
        }
        gl = [_parse_list(s) for s in df["genres"]]
        tl = [_parse_list(s) for s in df["themes"]]
        gvocab = sorted({t for tags in gl for t in tags})
        tvocab = sorted({t for tags in tl for t in tags})
        return cls(item, _multihot(gl, gvocab), _multihot(tl, tvocab), cold)

    def affinity(self, cand_idx: np.ndarray, g_pref: np.ndarray, t_pref: np.ndarray):
        """genre/theme affinity của candidate với user-pref (mean multihot history).
        cand_idx [.., C], pref [.., G/T] — broadcast theo batch."""
        gc, tc = self.genres[cand_idx], self.themes[cand_idx]              # [.., C, G/T]
        genre_aff = np.einsum("...cg,...g->...c", gc, g_pref)
        theme_aff = np.einsum("...ct,...t->...c", tc, t_pref)
        genre_overlap = np.einsum("...cg,...g->...c", gc, (g_pref > 0).astype(np.float32))
        return (genre_aff.astype(np.float32), theme_aff.astype(np.float32),
                genre_overlap.astype(np.float32))


def build_frame(itemfeat: ItemFeatures, cand_idx: np.ndarray, cross: dict) -> pd.DataFrame:
    """Lắp DataFrame[FEATURE_NAMES]: cross/user (cross dict, flat [n]) + item gather theo cand_idx."""
    data = {name: cross[name] for name in FEATURE_NAMES[:N_CROSS]}
    for name in ITEM_COLS:
        data[name] = itemfeat.item[name][cand_idx]
    return pd.DataFrame(data)[FEATURE_NAMES]
