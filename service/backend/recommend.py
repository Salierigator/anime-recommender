"""recommend.py — luồng inference end-to-end của service (firewall-clean, no UI).

username → (animelist + profile JSON live) → encode U → retrieve top-200 cosine → rerank
LightGBM + blend cosine → gợi ý top-K. Mirror khối per-user của ranker/src/eval.py:107-133
(collect_records) + score_config, KHÁC: input JSON live (không stream ratings.csv), 1 user,
KHÔNG hold-out query (serve thật → dùng toàn bộ positive làm history).

TÁI SỬ DỤNG (không viết lại) 2 module ranker "DÙNG CHUNG với service":
  - user_encode.py: load_user_encoder, encode_gender_joined (tự import UserTower từ retriever).
  - features.py:    ItemFeatures, build_frame, FEATURE_NAMES, K_RETRIEVE, REF_YEAR.
ĐỌC artifacts/ + cleaned-data/details.csv (qua ItemFeatures + map title). Live qua mal_api.

⚠️ import torch (user_encode) TRƯỚC lightgbm — 2 OpenMP runtime → segfault trên mac.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]                 # anime recommender/
sys.path.insert(0, str(ROOT / "ranker" / "src"))           # import module shared của ranker
import torch                                               # noqa: E402  (TRƯỚC lightgbm)
from user_encode import ARTIFACTS, encode_gender_joined, load_user_encoder   # noqa: E402
from features import REF_YEAR, ItemFeatures, K_RETRIEVE, FEATURE_NAMES, build_frame  # noqa: E402
import lightgbm as lgb                                     # noqa: E402  (SAU torch)

CLEANED = ARTIFACTS.parent / "cleaned-data"
K_HISTORY = 30                                             # khớp data_prep/05 + build_dataset
AGE_FALLBACK = 8.0                                         # u_account_age khi thiếu 'joined' (no batch median)


def rank_norm(x: np.ndarray) -> np.ndarray:
    """Thứ hạng chuẩn hoá [0,1] (lớn→1). Copy eval.py:40-44 (blend cần cùng công thức)."""
    r = np.empty(len(x), dtype=np.float64)
    r[np.argsort(x)] = np.arange(len(x))
    return r / max(len(x) - 1, 1)


class Recommender:
    """Load artifacts 1 lần (đắt) → recommend() rẻ, gọi lại mọi request."""

    def __init__(self, device: str = "cpu"):
        self.enc, self.meta = load_user_encoder(device)
        self.V = self.enc.item_cache.numpy()                          # [N,128] L2-norm (row==anime_idx)
        self.itemfeat = ItemFeatures.load(ARTIFACTS, CLEANED)
        self.booster = lgb.Booster(model_file=str(ARTIFACTS / "ranker.txt"))
        rmeta = json.loads((ARTIFACTS / "ranker_meta.json").read_text())
        self.alpha = float(rmeta["blend_alpha"])
        self.newest = len(self.meta["user_features"]["joined"]["bins"]) - 2   # joined_bucket mặc định

        idx = pd.read_parquet(ARTIFACTS / "item_index.parquet")       # anime_idx, mal_id
        real = idx[idx["anime_idx"] >= 2]                             # bỏ PAD(0)/OOV(1) (mal_id=-1)
        self.mal2idx = dict(zip(real["mal_id"].tolist(), real["anime_idx"].tolist()))
        self.idx2mal = dict(zip(idx["anime_idx"].tolist(), idx["mal_id"].tolist()))
        det = pd.read_csv(CLEANED / "details.csv", usecols=["mal_id", "title", "score"])
        self.id2title = dict(zip(det["mal_id"].tolist(), det["title"].tolist()))
        self.id2score = dict(zip(det["mal_id"].tolist(), det["score"].tolist()))   # điểm TB MAL (NaN nếu thiếu)

    # ---- user-side feature engineering (từ JSON live) ----
    def build_history(self, animelist: list) -> dict:
        """animelist (MAL v2) → history top-30 + seen + pref + user stats.

        positive = `status=='completed' & score∉[1,4]` (khớp build_dataset.py:63). seen = mọi
        anime map được (mọi status) → loại khỏi retrieval. history = positive top-30 (score desc,
        anime_idx asc tie-break deterministic)."""
        seen: set[int] = set()
        pos: dict[int, int] = {}                                      # anime_idx -> score max
        for e in animelist:
            ai = self.mal2idx.get((e.get("node") or {}).get("id"))
            if ai is None:                                            # anime ngoài corpus 22.8k
                continue
            seen.add(ai)
            ls = e.get("list_status") or {}
            score = ls.get("score") or 0
            if ls.get("status") == "completed" and score not in (1, 2, 3, 4):
                if score > pos.get(ai, -1):
                    pos[ai] = score
        items = sorted(pos.items(), key=lambda kv: (-kv[1], kv[0]))[:K_HISTORY]
        hist_idx = np.array([ai for ai, _ in items], dtype=np.int64)
        hist_score = np.array([sc for _, sc in items], dtype=np.int64)
        if len(hist_idx):
            g_pref = self.itemfeat.genres[hist_idx].mean(0)
            t_pref = self.itemfeat.themes[hist_idx].mean(0)
        else:                                                        # cold: history rỗng → h_empty lo
            g_pref = np.zeros(self.itemfeat.genres.shape[1], np.float32)
            t_pref = np.zeros(self.itemfeat.themes.shape[1], np.float32)
        rated = hist_score[hist_score >= 1]
        return {
            "hist_idx": hist_idx, "hist_score": hist_score, "seen": seen,
            "g_pref": g_pref, "t_pref": t_pref, "n_positive": len(pos),
            "u_n_rated": float(len(rated)),
            "u_mean_score": float(rated.mean()) if len(rated) else 0.0,
            "u_std_score": float(rated.std()) if len(rated) else 0.0,
        }

    def build_user_feats(self, profile: dict | None, username: str):
        """Jikan profile → (gender_id, joined_bucket, u_account_age). Profile None → mặc định."""
        if not profile:
            return 0, self.newest, AGE_FALLBACK
        df = pd.DataFrame([{"username": username,
                            "gender": profile.get("gender"), "joined": profile.get("joined")}])
        enc = encode_gender_joined(df, self.meta).iloc[0]
        yr = pd.to_datetime(profile.get("joined"), errors="coerce", utc=True)
        u_age = float(REF_YEAR - yr.year) if pd.notna(yr) else AGE_FALLBACK
        return int(enc["gender_id"]), int(enc["joined_bucket"]), u_age

    # ---- end-to-end ----
    def recommend(self, animelist: list, profile: dict | None,
                  top_k: int = 20, username: str = "user") -> tuple[list[dict], dict]:
        """→ (recs top-K, info debug). FastAPI sau chỉ cần recs ([0])."""
        h = self.build_history(animelist)
        gender_id, joined_bucket, u_age = self.build_user_feats(profile, username)
        hist_idx = h["hist_idx"]

        pad = lambda a: np.pad(a, (0, K_HISTORY - len(a)))[:K_HISTORY]
        hid = torch.tensor(pad(hist_idx)[None, :], dtype=torch.long)
        hsc = torch.tensor(pad(h["hist_score"])[None, :], dtype=torch.long)
        gid = torch.tensor([gender_id], dtype=torch.long)
        jb = torch.tensor([joined_bucket], dtype=torch.long)
        u = self.enc.encode(hid, hid != 0, hsc, gid, jb).numpy()[0]   # [128] L2-norm
        cos = self.V @ u                                              # [N] = cos_uv toàn item

        masked = cos.copy()
        if h["seen"]:
            masked[list(h["seen"])] = -np.inf
        masked[:2] = -np.inf                                          # PAD/OOV
        cand = np.argpartition(masked, -K_RETRIEVE)[-K_RETRIEVE:]     # top-200 (chưa sort)

        if len(hist_idx):
            sims = self.V[cand] @ self.V[hist_idx].T                  # [200, h]
            hist_cos_max, hist_cos_mean = sims.max(1), sims.mean(1)
        else:
            hist_cos_max = hist_cos_mean = np.zeros(len(cand), np.float32)
        ga, ta, go = self.itemfeat.affinity(cand, h["g_pref"], h["t_pref"])
        n = len(cand)
        cos_cand = cos[cand]
        cross = {
            "cos_uv": cos_cand, "hist_cos_max": hist_cos_max, "hist_cos_mean": hist_cos_mean,
            "genre_aff": ga, "theme_aff": ta, "genre_overlap": go,
            "u_n_rated": np.full(n, h["u_n_rated"]), "u_mean_score": np.full(n, h["u_mean_score"]),
            "u_std_score": np.full(n, h["u_std_score"]), "u_account_age": np.full(n, u_age),
        }
        X = build_frame(self.itemfeat, cand, cross)
        pred = self.booster.predict(X)
        blend = (1 - self.alpha) * rank_norm(cos_cand) + self.alpha * rank_norm(pred)

        order = np.argsort(-blend)[:top_k]
        recs = []
        for i, j in enumerate(order):
            ai = int(cand[j])
            mal_id = int(self.idx2mal.get(ai, -1))
            sc = self.id2score.get(mal_id)
            recs.append({
                "rank": i + 1, "anime_idx": ai, "mal_id": mal_id,
                "title": self.id2title.get(mal_id, "?"),
                "score": None if sc is None or np.isnan(sc) else round(float(sc), 2),   # điểm TB MAL
                "blend_score": float(blend[j]), "cos_uv": float(cos_cand[j]), "ranker_pred": float(pred[j]),
            })

        info = {
            "n_entries": len(animelist), "n_seen": len(h["seen"]),
            "n_positive": h["n_positive"], "n_history": int(len(hist_idx)),
            "cold": bool(len(hist_idx) == 0),
            "gender_id": gender_id, "joined_bucket": joined_bucket,
            "u_n_rated": h["u_n_rated"], "u_mean_score": h["u_mean_score"],
            "u_std_score": h["u_std_score"], "u_account_age": u_age,
            "alpha": self.alpha, "k_retrieve": int(K_RETRIEVE),
            "history": [{"mal_id": int(self.idx2mal.get(int(ai), -1)),
                         "title": self.id2title.get(self.idx2mal.get(int(ai)), "?"), "score": int(sc)}
                        for ai, sc in zip(hist_idx, h["hist_score"])],
        }
        return recs, info
