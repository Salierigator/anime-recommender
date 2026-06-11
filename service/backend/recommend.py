"""recommend.py — CLI inference end-to-end: username → 2 section gợi ý (firewall-clean, no UI).

    venv/bin/python service/backend/recommend.py <username> [--top-k 20] [--cold-k 10]
                                                 [--live] [--no-cache] [--dump]
    venv/bin/python service/backend/recommend.py --mal-ids service/backend/dummy_mal_ids.txt

Nguồn history (theo thứ tự):
  1. username có trong dataset → artifacts/users_history.parquet (offline, không cần API;
     user val/test: history = support — gợi ý có thể trúng query held-out, đó là điểm tốt).
  2. --live hoặc username ngoài dataset → MAL v2 animelist + Jikan profile (mal_api.py,
     cần MAL_CLIENT_ID trong service/.env); cache JSON ở backend/cache/.
  3. --mal-ids <file>: list mal_id (1 id/dòng) giả làm list completed chưa chấm điểm.

Serving theo `ranker_meta.json::cold_serving` (xem docs/RANKER.md §7-8):
  U (UserEncoder) → cosine full catalog, mask seen∪hard_neg →
  [Gợi ý]    top-k_retrieve WARM-only → 29 feature (đúng code path pool.cross_features
             của train/eval) → LightGBM → blend α (α=1 → sort theo pred) → top-K
  [Anime mới] item is_cold theo cosine (KHÔNG qua ranker — α=1 dìm cold)

TÁI SỬ DỤNG module ranker "DÙNG CHUNG service" (KHÔNG viết lại — tránh drift):
user_encode (encoder), features (ItemFeatures/build_frame), pool (encode_users/topk_pool/
cross_features/user_stats_from_support). ⚠ import torch TRƯỚC lightgbm (OpenMP segfault mac).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

ROOT = Path(__file__).resolve().parents[2]                 # anime recommender/
sys.path.insert(0, str(ROOT / "ranker" / "src"))
import torch                                               # noqa: E402  (TRƯỚC lightgbm)
# features/pool TRƯỚC user_encode: cache module `config` của ranker vào sys.modules
# trước khi user_encode chèn retriever/src (cũng có config.py) vào sys.path[0]
from features import REF_YEAR, FEATURE_NAMES, ItemFeatures, build_frame     # noqa: E402
from pool import (cross_features, encode_users, topk_pool,                  # noqa: E402
                  user_stats_from_support)
from user_encode import ARTIFACTS, encode_gender_joined, load_user_encoder  # noqa: E402
import lightgbm as lgb                                     # noqa: E402  (SAU torch)

CLEANED = ROOT / "cleaned-data"
CACHE = Path(__file__).resolve().parent / "cache"


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
                          usecols=["mal_id", "title", "type", "score", "start_date"])
        det["year"] = pd.to_datetime(det["start_date"], errors="coerce", utc=True).dt.year
        self.detail = det.set_index("mal_id")[["title", "type", "score", "year"]]

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
    def recommend(self, user: dict, top_k: int = 20, cold_k: int = 10) -> dict:
        U = encode_users(self.enc, [user["hist_idx"]], [user["hist_score"]],
                         np.asarray([user["gender_id"]]), np.asarray([user["joined_bucket"]]),
                         self.cap)
        mask = [user["seen"]]
        # [Gợi ý] warm-only → rerank LightGBM (cold_serving: cold KHÔNG qua ranker)
        cand, cos = topk_pool(U, self.enc.item_cache, mask, self.k_retrieve,
                              cold_idx=self.cold_idx)
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
        # [Anime mới] cold theo cosine (mask warm + seen)
        ccand, ccos = topk_pool(U, self.enc.item_cache, mask, cold_k,
                                cold_idx=self.warm_idx)
        cold = [self._row(int(a), cos=float(c)) for a, c in zip(ccand[0], ccos[0])]
        return {"main": main, "cold": cold}

    def _row(self, anime_idx: int, **scores) -> dict:
        mal_id = int(self.idx2mal[anime_idx])
        d = self.detail.loc[mal_id] if mal_id in self.detail.index else None
        return {
            "mal_id": mal_id,
            "title": str(d["title"]) if d is not None else "?",
            "type": str(d["type"]) if d is not None else "?",
            "year": int(d["year"]) if d is not None and pd.notna(d["year"]) else None,
            "mal_score": round(float(d["score"]), 2)
                         if d is not None and pd.notna(d["score"]) else None,
            **{k: round(v, 4) for k, v in scores.items()},
        }


def fetch_live(username: str, no_cache: bool):
    """MAL v2 animelist + Jikan profile, cache JSON ở backend/cache/."""
    import mal_api
    CACHE.mkdir(exist_ok=True)
    al_p, pr_p = CACHE / f"{username}_animelist.json", CACHE / f"{username}_profile.json"
    if not no_cache and al_p.exists() and pr_p.exists():
        print(f"[cache] {al_p.name} + {pr_p.name}")
        return json.loads(al_p.read_text()), json.loads(pr_p.read_text())
    print(f"[api] fetch animelist + profile '{username}' ...")
    animelist = mal_api.get_user_anime_list(username)
    profile = mal_api.get_user_profile(username)
    al_p.write_text(json.dumps(animelist, ensure_ascii=False))
    pr_p.write_text(json.dumps(profile or {}, ensure_ascii=False))
    return animelist, profile


def print_table(title: str, rows: list[dict]) -> None:
    print(f"\n── {title} " + "─" * max(0, 76 - len(title)))
    print(f"{'#':>3}  {'title':<46} {'year':>5} {'type':<7} {'MAL':>5}"
          f"{'pred':>8} {'cos':>7}")
    for i, r in enumerate(rows, 1):
        print(f"{i:>3}  {(r['title'] or '?')[:46]:<46} {r['year'] or '-':>5} "
              f"{r['type']:<7} {r['mal_score'] if r['mal_score'] is not None else '-':>5}"
              f"{r.get('pred', ''):>8} {r.get('cos', ''):>7}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Gợi ý anime cho 1 user (CLI).")
    ap.add_argument("username", nargs="?", help="MAL username (dataset hoặc live)")
    ap.add_argument("--mal-ids", type=Path, help="file mal_id (1 id/dòng) thay cho username")
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--cold-k", type=int, default=10)
    ap.add_argument("--live", action="store_true", help="ép fetch API dù username có trong dataset")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--dump", action="store_true", help="dump JSON ra backend/cache/")
    args = ap.parse_args()
    if not args.username and not args.mal_ids:
        ap.error("cần username hoặc --mal-ids")

    t0 = time.time()
    rec = Recommender()
    print(f"[load] Recommender sẵn sàng ({time.time() - t0:.1f}s)")

    if args.mal_ids:
        ids = [int(l) for l in args.mal_ids.read_text().split() if l.strip()]
        user, name = rec.user_from_mal_ids(ids), args.mal_ids.stem
        print(f"[user] {len(ids)} mal_id → {len(user['hist_idx'])} item trong corpus")
    else:
        name = args.username
        user = None if args.live else rec.user_from_dataset(name)
        if user is None:
            animelist, profile = fetch_live(name, args.no_cache)
            if not animelist:
                print("[!] animelist rỗng (list private / user không tồn tại / thiếu "
                      "MAL_CLIENT_ID trong service/.env)")
            user = rec.user_from_animelist(animelist or [], profile, name)

    n_hist = len(user["hist_idx"])
    print(f"[user] '{name}' source={user['source']} split={user['split']} "
          f"history={n_hist} seen(masked)={len(user['seen'])}"
          + ("   [!] COLD START → gợi ý thiên phổ biến" if n_hist == 0 else ""))
    top = [rec._row(int(a)) | {"score": int(s)}
           for a, s in zip(user["hist_idx"][:5], user["hist_score"][:5])]
    for r in top:
        print(f"        hist: {r['title'][:40]:<40} (score {r['score']})")

    t1 = time.time()
    out = rec.recommend(user, top_k=args.top_k, cold_k=args.cold_k)
    print(f"[infer] {time.time() - t1:.2f}s  (α={rec.alpha}, K={rec.k_retrieve})")
    print_table(f"Gợi ý cho bạn (rerank LightGBM, top {args.top_k})", out["main"])
    print_table(f"Anime mới cho bạn (cold theo retriever, top {args.cold_k})", out["cold"])

    if args.dump:
        CACHE.mkdir(exist_ok=True)
        p = CACHE / f"{name}_recs.json"
        p.write_text(json.dumps({"user": name, **out}, ensure_ascii=False, indent=2,
                                default=str))
        print(f"\n[dump] {p}")


if __name__ == "__main__":
    main()
