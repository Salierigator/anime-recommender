"""pool.py — lõi dựng candidate pool DÙNG CHUNG build_train/build_eval (train↔serve consistency):
encode U (UserEncoder từ artifacts, history cap = eval_history_cap) → cosine full catalog →
mask → top-D + cross features. Một đường code duy nhất nên phân phối candidate lúc train
khớp serve.

Import torch TRƯỚC lightgbm ở mọi file downstream (segfault 2 OpenMP runtime trên mac).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import torch

import config
from features import ItemFeatures, build_frame

REF_YEAR = 2024


class UsersHistory:
    """artifacts/users_history.parquet — ragged (values+offsets, không pad full RAM).
    History FULL sort (score desc, tie asc) → prefix = top-by-score. Eval user: list này
    chính là support của retriever (đã trừ query + cold H)."""

    def __init__(self, artifacts: Path = config.ARTIFACTS):
        t = pq.read_table(artifacts / "users_history.parquet")
        uidx = t.column("user_idx").to_numpy()
        self.num_users = len(uidx)
        assert (uidx == np.arange(self.num_users)).all(), "users_history phải dense 0..U"
        self.gender_id = t.column("gender_id").to_numpy().astype(np.int64)
        self.joined_bucket = t.column("joined_bucket").to_numpy().astype(np.int64)
        hcol = t.column("history_ids").combine_chunks()
        self.hist_vals = hcol.values.to_numpy(zero_copy_only=False).astype(np.int32)
        self.hist_offs = hcol.offsets.to_numpy().astype(np.int64)
        scol = t.column("history_scores").combine_chunks()
        self.hist_scores = scol.values.to_numpy(zero_copy_only=False).astype(np.int8)
        ncol = t.column("hard_neg_ids").combine_chunks()
        self.hn_vals = ncol.values.to_numpy(zero_copy_only=False).astype(np.int32)
        self.hn_offs = ncol.offsets.to_numpy().astype(np.int64)

    def history(self, u: int) -> tuple[np.ndarray, np.ndarray]:
        s, e = self.hist_offs[u], self.hist_offs[u + 1]
        return self.hist_vals[s:e], self.hist_scores[s:e]

    def hard_neg(self, u: int) -> np.ndarray:
        return self.hn_vals[self.hn_offs[u]:self.hn_offs[u + 1]]

    def history_lens(self) -> np.ndarray:
        return self.hist_offs[1:] - self.hist_offs[:-1]


def load_eval_seen(artifacts: Path = config.ARTIFACTS) -> dict[int, np.ndarray]:
    t = pq.read_table(artifacts / "eval_seen.parquet")
    uids = t.column("user_idx").to_numpy()
    col = t.column("seen_ids").combine_chunks()
    vals = col.values.to_numpy(zero_copy_only=False).astype(np.int64)
    offs = col.offsets.to_numpy().astype(np.int64)
    return {int(u): vals[offs[i]:offs[i + 1]] for i, u in enumerate(uids)}


def load_queries(split: str, artifacts: Path = config.ARTIFACTS
                 ) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """eval_queries_{split}.parquet -> (queries, query_scores): {user_idx: array anime_idx} +
    {user_idx: array score} song song (giữ thứ tự file). score dùng chấm liked-metric."""
    t = pl.read_parquet(artifacts / f"eval_queries_{split}.parquet")
    qa: dict[int, list] = {}
    qs: dict[int, list] = {}
    for u, a, s in zip(t["user_idx"].to_numpy(), t["anime_idx"].to_numpy(), t["score"].to_numpy()):
        qa.setdefault(int(u), []).append(int(a))
        qs.setdefault(int(u), []).append(int(s))
    return ({u: np.asarray(v, dtype=np.int64) for u, v in qa.items()},
            {u: np.asarray(v, dtype=np.int64) for u, v in qs.items()})


def account_age_by_user(artifacts: Path = config.ARTIFACTS,
                        cleaned: Path = config.CLEANED) -> dict[int, float]:
    """user_idx -> account age (REF_YEAR - năm joined; NaN nếu thiếu) qua user_split + profiles."""
    split = pl.read_parquet(artifacts / "user_split.parquet")
    prof = pd.read_csv(cleaned / "profiles.csv", usecols=["username", "joined"])
    yr = pd.to_datetime(prof["joined"], errors="coerce").dt.year
    age = dict(zip(prof["username"], (REF_YEAR - yr).astype(float)))
    return {int(u): age.get(name, float("nan"))
            for name, u in zip(split["username"], split["user_idx"])}


def _pad_lists(lists: list[np.ndarray], cap: int, dtype=np.int64):
    """list ragged -> (mat [n, W] pad 0, mask [n, W]); W = min(max len, cap), >=1."""
    W = max(min(max((len(x) for x in lists), default=1), cap), 1)
    mat = np.zeros((len(lists), W), dtype=dtype)
    mask = np.zeros((len(lists), W), dtype=bool)
    for i, x in enumerate(lists):
        v = np.asarray(x)[:W]
        mat[i, : len(v)] = v
        mask[i, : len(v)] = True
    return mat, mask


@torch.no_grad()
def encode_users(enc, hist_ids: list[np.ndarray], hist_scores: list[np.ndarray],
                 gender: np.ndarray, joined: np.ndarray, cap: int) -> torch.Tensor:
    """U [n, d] từ history prefix (đã sort score desc) — khớp eval path retriever."""
    ids, mask = _pad_lists(hist_ids, cap)
    sc, _ = _pad_lists(hist_scores, cap)
    return enc.encode(torch.from_numpy(ids), torch.from_numpy(mask), torch.from_numpy(sc),
                      torch.from_numpy(gender), torch.from_numpy(joined))


@torch.no_grad()
def topk_pool(U: torch.Tensor, item_cache: torch.Tensor, mask_lists: list[np.ndarray],
              depth: int, cold_idx: np.ndarray | None = None):
    """scores = U @ V.T; -inf cho PAD/OOV + mask per-user (+ toàn bộ cold nếu cold_idx);
    top-depth sort desc. Trả (cand [n,depth] i64, cos [n,depth] f32)."""
    scores = U @ item_cache.t()                                   # [n, N]
    scores[:, :2] = float("-inf")
    if cold_idx is not None:
        scores[:, torch.from_numpy(cold_idx)] = float("-inf")
    mpad, mmask = _pad_lists(mask_lists, cap=10**9)
    mpad[~mmask] = 0                                              # pad -> PAD row (vốn -inf)
    scores.scatter_(1, torch.from_numpy(mpad), float("-inf"))
    top = torch.topk(scores, depth, dim=1)                        # sorted desc
    return top.indices.numpy().astype(np.int64), top.values.numpy().astype(np.float32)


def cross_features(V: np.ndarray, itemfeat: ItemFeatures, cand: np.ndarray, cos: np.ndarray,
                   hist_feat: list[np.ndarray], user_stats: dict[str, np.ndarray]) -> dict:
    """Cross features [n, D] -> dict name -> flat [n*D] (thứ tự FEATURE_NAMES[:N_CROSS]).
    hist_feat = prefix HIST_FEAT_CAP của support (top-by-score) — chỉ cho FEATURE."""
    n, D = cand.shape
    hp, hm = _pad_lists(hist_feat, config.HIST_FEAT_CAP)          # [n, H]
    Vc = torch.from_numpy(V[cand])                                # [n, D, d]
    Vh = torch.from_numpy(V[hp]) * torch.from_numpy(hm)[..., None]  # pad -> 0 vec
    sims = torch.bmm(Vc, Vh.transpose(1, 2))                      # [n, D, H]
    sims.masked_fill_(~torch.from_numpy(hm)[:, None, :], float("-inf"))
    hist_max = sims.max(dim=2).values
    hist_max = torch.where(torch.isfinite(hist_max), hist_max, torch.zeros(()))
    top5 = sims.topk(min(5, sims.shape[2]), dim=2).values
    lens = torch.from_numpy(hm.sum(1, dtype=np.float32))          # [n]
    top5_mean = torch.where(torch.isfinite(top5), top5, torch.zeros(())).sum(2) \
        / torch.minimum(lens, torch.tensor(5.0)).clamp(min=1)[:, None]
    hist_mean = torch.where(torch.isfinite(sims), sims, torch.zeros(())).sum(2) \
        / lens.clamp(min=1)[:, None]

    counts = hm.sum(1, keepdims=True).astype(np.float32).clip(min=1)
    g_pref = (itemfeat.genres[hp] * hm[..., None]).sum(1) / counts   # [n, G]
    t_pref = (itemfeat.themes[hp] * hm[..., None]).sum(1) / counts
    ga, ta, go = itemfeat.affinity(cand, g_pref, t_pref)             # [n, D]

    mal = itemfeat.item["mal_score"][cand]                           # [n, D]
    rep = lambda a: np.repeat(a.astype(np.float32), D)
    return {
        "cos_uv": cos.ravel(),
        "pool_rank": np.tile(np.arange(D, dtype=np.float32), n),
        "hist_cos_max": hist_max.numpy().ravel(),
        "hist_cos_mean": hist_mean.numpy().ravel(),
        "hist_cos_top5_mean": top5_mean.numpy().ravel(),
        "genre_aff": ga.ravel(), "theme_aff": ta.ravel(), "genre_overlap": go.ravel(),
        "score_gap": (mal - user_stats["u_mean_score"][:, None]).astype(np.float32).ravel(),
        "u_n_rated": rep(user_stats["u_n_rated"]),
        "u_mean_score": rep(user_stats["u_mean_score"]),
        "u_std_score": rep(user_stats["u_std_score"]),
        "u_account_age": rep(user_stats["u_account_age"]),
        "support_len": rep(user_stats["support_len"]),
    }


def user_stats_from_support(supp_scores: list[np.ndarray], ages: np.ndarray) -> dict:
    """Stats user từ score support (rated = score>=1; score 0 = completed không chấm)."""
    n_rated, mean_s, std_s, slen = [], [], [], []
    for sc in supp_scores:
        rated = sc[sc >= 1]
        n_rated.append(float(len(rated)))
        mean_s.append(float(rated.mean()) if len(rated) else 0.0)
        std_s.append(float(rated.std()) if len(rated) else 0.0)
        slen.append(float(len(sc)))
    return {
        "u_n_rated": np.asarray(n_rated, np.float32),
        "u_mean_score": np.asarray(mean_s, np.float32),
        "u_std_score": np.asarray(std_s, np.float32),
        "u_account_age": ages.astype(np.float32),
        "support_len": np.asarray(slen, np.float32),
    }


class PoolWriter:
    """Ghi pool parquet streaming theo chunk (tránh giữ 20M row trong RAM) + side table user."""

    def __init__(self, pool_path: Path, users_path: Path):
        pool_path.parent.mkdir(parents=True, exist_ok=True)
        self.pool_path, self.users_path = pool_path, users_path
        self.writer = None
        self.users_rows = {"qid": [], "user_idx": [], "r_total": [], "r_liked": [],
                           "U": [], "hist_top64": []}
        self.next_qid = 0

    def add_chunk(self, uids: np.ndarray, cand: np.ndarray, labels: np.ndarray,
                  frame: pd.DataFrame, U: np.ndarray, hist_top64: list[np.ndarray],
                  r_total: np.ndarray, label_liked: np.ndarray | None = None,
                  r_liked: np.ndarray | None = None, keep: np.ndarray | None = None):
        """cand/labels/label_liked [n, D]; frame = build_frame flat [n*D]; keep = bool [n] (lọc group).
        label_liked = candidate ∈ liked-query (binary); r_liked [n] = #liked query/user (cả ngoài pool).
        None (train pool — không chấm liked) -> ghi cột zeros để schema users/pool đồng nhất."""
        n, D = cand.shape
        if label_liked is None:
            label_liked = np.zeros_like(labels)
        if r_liked is None:
            r_liked = np.zeros(n, dtype=np.int64)
        if keep is None:
            keep = np.ones(n, dtype=bool)
        rows = np.repeat(keep, D)
        qid_local = np.cumsum(keep) - 1                            # dense trong chunk
        qid = (self.next_qid + qid_local).astype(np.int64)

        out = frame[rows].reset_index(drop=True)
        out.insert(0, "qid", np.repeat(qid, D)[rows])
        out.insert(1, "user_idx", np.repeat(uids.astype(np.int64), D)[rows])
        out.insert(2, "anime_idx", cand.ravel()[rows])
        out["label"] = labels.ravel()[rows]
        out["label_liked"] = label_liked.ravel()[rows]
        table = pa.Table.from_pandas(out, preserve_index=False)
        if self.writer is None:
            self.writer = pq.ParquetWriter(self.pool_path, table.schema, compression="zstd")
        self.writer.write_table(table)

        for i in np.flatnonzero(keep):
            self.users_rows["qid"].append(self.next_qid + int(qid_local[i]))
            self.users_rows["user_idx"].append(int(uids[i]))
            self.users_rows["r_total"].append(int(r_total[i]))
            self.users_rows["r_liked"].append(int(r_liked[i]))
            self.users_rows["U"].append(U[i].astype(np.float32))
            self.users_rows["hist_top64"].append(hist_top64[i][:config.HIST_TOP64].astype(np.int32))
        self.next_qid += int(keep.sum())

    def close(self):
        if self.writer is not None:
            self.writer.close()
        pl.DataFrame({
            "qid": np.asarray(self.users_rows["qid"], np.int64),
            "user_idx": np.asarray(self.users_rows["user_idx"], np.int64),
            "r_total": np.asarray(self.users_rows["r_total"], np.int32),
            "r_liked": np.asarray(self.users_rows["r_liked"], np.int32),
            "U": [u.tolist() for u in self.users_rows["U"]],
            "hist_top64": [h.tolist() for h in self.users_rows["hist_top64"]],
        }).write_parquet(self.users_path)
        return self.next_qid
