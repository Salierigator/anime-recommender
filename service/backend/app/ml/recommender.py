"""recommender.py — Recommender serving core (model inference). Moved từ recommend.py.

Wrap/consume bởi: CLI (service/backend/recommend.py) và app/services/real_service.py.
Đọc-only artifacts + cleaned-data (firewall service/CLAUDE.md §0).

⚠ THỨ TỰ IMPORT load-bearing (giữ nguyên — service/CLAUDE.md §4):
  (1) torch TRƯỚC lightgbm (2 OpenMP runtime → segfault mac);
  (2) features/pool TRƯỚC user_encode (user_encode chèn retriever/src vào sys.path[0], cũng có
      config.py → import sau nó lấy NHẦM config retriever).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

ROOT = Path(__file__).resolve().parents[4]                 # anime recommender/
sys.path.insert(0, str(ROOT / "ranker" / "src"))
import torch                                               # noqa: E402  (TRƯỚC lightgbm)
# features/pool TRƯỚC user_encode: cache module `config` của ranker vào sys.modules
# trước khi user_encode chèn retriever/src (cũng có config.py) vào sys.path[0]
from features import REF_YEAR, FEATURE_NAMES, ItemFeatures, build_frame, _parse_list  # noqa: E402
from pool import (cross_features, encode_users, topk_pool,                  # noqa: E402
                  user_stats_from_support)
from user_encode import ARTIFACTS, encode_gender_joined, load_user_encoder  # noqa: E402
import lightgbm as lgb                                     # noqa: E402  (SAU torch)

CLEANED = ROOT / "cleaned-data"


class Recommender:
    """Load artifacts 1 lần (đắt ~5s) → recommend() rẻ, gọi lại được nhiều lần."""

    def __init__(self, device: str = "cpu"):
        self.enc, self.meta = load_user_encoder(device)
        self.cap = self.meta.get("eval_history_cap", 1024)
        self.V = self.enc.item_cache.numpy()
        self.itemfeat = ItemFeatures.load(ARTIFACTS, CLEANED)
        self.cold_idx = np.flatnonzero(self.itemfeat.is_cold)
        self.warm_idx = np.flatnonzero(~self.itemfeat.is_cold)
        self.booster = lgb.Booster(model_file=str(ARTIFACTS / "ranker.txt"))
        assert self.booster.feature_name() == FEATURE_NAMES, "ranker.txt lệch FEATURE_NAMES"
        rmeta = json.loads((ARTIFACTS / "ranker_meta.json").read_text())
        self.alpha = float(rmeta["blend_alpha"])
        self.k_retrieve = int(rmeta["k_retrieve"])
        self.newest = len(self.meta["user_features"]["joined"]["bins"]) - 2

        idx = pl.read_parquet(ARTIFACTS / "item_index.parquet")
        self.idx2mal = idx["mal_id"].to_numpy()
        real = idx.filter(pl.col("anime_idx") >= 2)
        self.mal2idx = dict(zip(real["mal_id"].to_list(), real["anime_idx"].to_list()))
        det = pd.read_csv(CLEANED / "details.csv",
                          usecols=["mal_id", "title", "type", "score", "start_date",
                                   "genres", "themes", "studios", "rating"])
        det["year"] = pd.to_datetime(det["start_date"], errors="coerce", utc=True).dt.year
        for col in ("genres", "themes", "studios"):        # str -> list[str] (cho filter client)
            det[col] = det[col].map(_parse_list)
        self.detail = det.set_index("mal_id")[["title", "type", "score", "year",
                                               "genres", "themes", "studios"]]
        # SFW: anime_idx của item hentai (genre 'Hentai' HOẶC rating 'Rx - Hentai')
        is_hentai = (det["rating"] == "Rx - Hentai") | \
            det["genres"].map(lambda g: "Hentai" in g)
        self.nsfw_idx = np.asarray(
            [self.mal2idx[m] for m in det.loc[is_hentai, "mal_id"] if m in self.mal2idx],
            dtype=np.int64)

    # ---- nguồn history ----
    def user_from_dataset(self, username: str) -> dict | None:
        """username trong user_split → history/gender/joined từ users_history (offline)."""
        row = pl.scan_parquet(ARTIFACTS / "user_split.parquet") \
            .filter(pl.col("username") == username).collect()
        if row.height == 0:
            return None
        uidx = int(row["user_idx"][0])
        uh = pl.scan_parquet(ARTIFACTS / "users_history.parquet") \
            .filter(pl.col("user_idx") == uidx).collect()
        prof = pd.read_csv(CLEANED / "profiles.csv", usecols=["username", "joined"])
        yr = pd.to_datetime(
            prof.loc[prof["username"] == username, "joined"], errors="coerce").dt.year
        hist = np.asarray(uh["history_ids"][0].to_list(), dtype=np.int64)
        scores = np.asarray(uh["history_scores"][0].to_list(), dtype=np.int64)
        hard_neg = np.asarray(uh["hard_neg_ids"][0].to_list(), dtype=np.int64)
        return {
            "hist_idx": hist, "hist_score": scores,
            "seen": np.union1d(hist, hard_neg),
            "gender_id": int(uh["gender_id"][0]), "joined_bucket": int(uh["joined_bucket"][0]),
            "age": float(REF_YEAR - yr.iloc[0]) if len(yr) and pd.notna(yr.iloc[0]) else np.nan,
            "split": str(row["split"][0]), "source": "dataset",
        }

    def user_from_animelist(self, animelist: list, profile: dict | None,
                            username: str) -> dict:
        """JSON live MAL v2 → positives (status∈{completed,watching} & score∉[1,4] — khớp
        prep retriever), FULL sort score desc; seen = mọi anime map được (mọi status)."""
        seen, pos = set(), {}
        for e in animelist:
            ai = self.mal2idx.get((e.get("node") or {}).get("id"))
            if ai is None:                                 # ngoài corpus 22.8k
                continue
            seen.add(ai)
            ls = e.get("list_status") or {}
            score = ls.get("score") or 0
            if ls.get("status") in ("completed", "watching") and score not in (1, 2, 3, 4):
                pos[ai] = max(score, pos.get(ai, 0))
        items = sorted(pos.items(), key=lambda kv: (-kv[1], kv[0]))
        gender_id, joined_bucket, age = 0, self.newest, np.nan
        if profile:
            df = pd.DataFrame([{"username": username, "gender": profile.get("gender"),
                                "joined": profile.get("joined")}])
            enc = encode_gender_joined(df, self.meta).iloc[0]
            gender_id, joined_bucket = int(enc["gender_id"]), int(enc["joined_bucket"])
            yr = pd.to_datetime(profile.get("joined"), errors="coerce", utc=True)
            age = float(REF_YEAR - yr.year) if pd.notna(yr) else np.nan
        return {
            "hist_idx": np.asarray([a for a, _ in items], dtype=np.int64),
            "hist_score": np.asarray([s for _, s in items], dtype=np.int64),
            "seen": np.asarray(sorted(seen), dtype=np.int64),
            "gender_id": gender_id, "joined_bucket": joined_bucket, "age": age,
            "split": "-", "source": "live",
        }

    def user_from_mal_ids(self, ids: list[int]) -> dict:
        """List mal_id → giả completed chưa chấm (score 0 vẫn là positive hợp lệ)."""
        hist = np.asarray(sorted({self.mal2idx[i] for i in ids if i in self.mal2idx}),
                          dtype=np.int64)
        return {"hist_idx": hist, "hist_score": np.zeros(len(hist), dtype=np.int64),
                "seen": hist, "gender_id": 0, "joined_bucket": self.newest,
                "age": np.nan, "split": "-", "source": "mal_ids"}

    # ---- end-to-end ----
    def recommend(self, user: dict, top_k: int = 20, cold_k: int = 10,
                  anchor_mal_id: int | None = None, sfw: bool = True) -> dict:
        U = encode_users(self.enc, [user["hist_idx"]], [user["hist_score"]],
                         np.asarray([user["gender_id"]]), np.asarray([user["joined_bucket"]]),
                         self.cap)
        # SFW: union item hentai vào mask → rớt khỏi cả pool warm lẫn cold trước khi rank
        block = np.union1d(user["seen"], self.nsfw_idx) if sfw else user["seen"]
        # serve có thể lấy pool sâu hơn k_retrieve (feature item bất biến theo depth: pool_rank
        # là hạng cosine tuyệt đối) → client lo filter/slice; top vẫn khớp điểm vận hành k_retrieve
        depth = max(self.k_retrieve, top_k)
        if anchor_mal_id is None:
            mask = [block]
            cold_query = U
            # [Gợi ý] warm-only → rerank LightGBM (cold_serving: cold KHÔNG qua ranker)
            cand, cos = topk_pool(U, self.enc.item_cache, mask, depth,
                                  cold_idx=self.cold_idx)
        else:
            # "giống X": pool theo anchor, nhưng cos_uv tính lại = user-item (ranker đúng phân phối)
            aidx = self.mal2idx.get(int(anchor_mal_id))
            if aidx is None:
                raise KeyError(anchor_mal_id)
            cold_query = self.enc.item_cache[aidx:aidx + 1]    # [1, d] = vector của X
            mask = [np.union1d(block, [aidx])]                 # loại chính X
            cand, _ = topk_pool(cold_query, self.enc.item_cache, mask, depth,
                                cold_idx=self.cold_idx)
            cos = (U @ self.enc.item_cache[torch.from_numpy(cand[0])].t()).numpy()  # [1, k]
        stats = user_stats_from_support([user["hist_score"]],
                                        np.asarray([user["age"]], dtype=np.float64))
        cross = cross_features(self.V, self.itemfeat, cand, cos, [user["hist_idx"]], stats)
        X = build_frame(self.itemfeat, cand.ravel(), cross)
        pred = self.booster.predict(X)
        if self.alpha < 1.0:                               # α=1 → sort thẳng theo pred
            from metrics import blend
            score = blend(cos.ravel().astype(np.float64), pred,
                          np.array([0, cand.shape[1]]), self.alpha)
        else:
            score = pred
        order = np.argsort(-score)[:top_k]
        main = [self._row(int(cand[0, j]), pred=float(pred[j]), cos=float(cos[0, j]))
                for j in order]
        # [Anime mới] cold theo cosine (mask warm + seen); anchor mode → cold giống X
        ccand, ccos = topk_pool(cold_query, self.enc.item_cache, mask, cold_k,
                                cold_idx=self.warm_idx)
        cold = [self._row(int(a), cos=float(c)) for a, c in zip(ccand[0], ccos[0])]
        return {"main": main, "cold": cold}

    def _row(self, anime_idx: int, **scores) -> dict:
        mal_id = int(self.idx2mal[anime_idx])
        d = self.detail.loc[mal_id] if mal_id in self.detail.index else None
        return {
            "mal_id": mal_id,
            "title": str(d["title"]) if d is not None else "?",
            "type": str(d["type"]) if d is not None and pd.notna(d["type"]) else "?",
            "year": int(d["year"]) if d is not None and pd.notna(d["year"]) else None,
            "mal_score": round(float(d["score"]), 2)
                         if d is not None and pd.notna(d["score"]) else None,
            "genres": list(d["genres"]) if d is not None else [],
            "themes": list(d["themes"]) if d is not None else [],
            "studios": list(d["studios"]) if d is not None else [],
            **{k: round(v, 4) for k, v in scores.items()},
        }
