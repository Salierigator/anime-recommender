"""user_encode.py — dựng user-vector U từ artifacts (firewall-clean, DÙNG CHUNG với service).

Replicate đúng encoder user-side của retriever, KHÔNG đụng train-data:
- history lấy từ `artifacts/users_history.parquet` (FULL, sort score desc, tie asc — prefix =
  top-by-score; eval user = support đã trừ query+H). Module này nhận sẵn history_ids/scores.
- gender/joined encode theo map nằm trong `user_tower.pt` (khớp data_prep/04) —
  `encode_gender_joined` chỉ cần cho user MỚI lúc serve (user có sẵn dùng users_history).
- U = UserTower(pool_history(history qua item_vectors), gender, joined), đã L2-norm.

Chỉ ĐỌC `artifacts/` + import ĐỊNH NGHĨA model (UserTower) — không import code train, không train-data.
pool_history/_attn_pool copy nguyên từ retriever/src/model.py:185-210 (không dựng được TwoTower vì
cần ItemTable → train-data blocked).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent          # ranker/
ARTIFACTS = ROOT.parent / "artifacts"
RETRIEVER_SRC = ROOT.parent / "retriever" / "src"
sys.path.insert(0, str(RETRIEVER_SRC))
from model import UserTower, _masked_mean, _masked_weighted_mean   # noqa: E402  (chỉ định nghĩa model)


class UserEncoder(nn.Module):
    """Mirror nhánh user của TwoTower: pool history (mean/attn ±score) ⊕ gender ⊕ joined → U[d]."""

    def __init__(self, meta: dict, item_vectors: np.ndarray):
        super().__init__()
        self.d = meta["d"]
        self.history_pool = meta["history_pool"]
        self.score_pool = meta["score_pool"]
        self.user_tower = UserTower({"user_features": meta["user_features"]}, meta["d"], meta["mlp_hidden"])
        if self.history_pool == "attn":
            self.attn_key = nn.Linear(self.d, self.d, bias=False)
            self.attn_query = nn.Parameter(torch.zeros(self.d))
        if self.score_pool == "learned":
            self.score_weight = nn.Embedding(11, 1)
        # item_cache = item_vectors (row==anime_idx); persistent=False để không lọt vào load_state_dict
        self.register_buffer("item_cache", torch.from_numpy(item_vectors).float(), persistent=False)

    def _attn_pool(self, vecs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        logits = (self.attn_key(vecs) @ self.attn_query) / self.d ** 0.5
        logits = logits.masked_fill(~mask, float("-inf"))
        logits = logits.masked_fill(~mask.any(dim=-1, keepdim=True), 0.0)
        attn = torch.softmax(logits, dim=-1)
        return (attn.unsqueeze(-1) * vecs).sum(dim=-2)

    def pool_history(self, history_ids, history_mask, history_scores=None) -> torch.Tensor:
        vecs = self.item_cache[history_ids]                    # [B, L, d]
        if self.history_pool == "attn":
            pooled = self._attn_pool(vecs, history_mask)
        elif self.score_pool == "none" or history_scores is None:
            pooled = _masked_mean(vecs, history_mask)
        else:
            if self.score_pool == "learned":
                w = F.softplus(self.score_weight(history_scores).squeeze(-1))
            else:
                w = history_scores.to(vecs.dtype).clamp(min=1.0)
            pooled = _masked_weighted_mean(vecs, history_mask, w)
        empty = ~history_mask.any(dim=-1)
        return torch.where(empty.unsqueeze(-1), self.user_tower.h_empty.to(pooled.dtype), pooled)

    @torch.no_grad()
    def encode(self, history_ids, history_mask, history_scores, gender_id, joined_bucket) -> torch.Tensor:
        pooled = self.pool_history(history_ids, history_mask, history_scores)
        return self.user_tower(pooled, gender_id, joined_bucket)


def load_user_encoder(device: str = "cpu", artifacts: Path = ARTIFACTS):
    """Load user_tower.pt + item_vectors.npy → (UserEncoder eval, meta)."""
    meta = torch.load(artifacts / "user_tower.pt", map_location="cpu", weights_only=False)
    item_vectors = np.load(artifacts / "item_vectors.npy")
    enc = UserEncoder(meta, item_vectors)
    enc.load_state_dict(meta["state_dict"], strict=True)
    return enc.eval().to(device), meta


def encode_gender_joined(prof: pd.DataFrame, meta: dict) -> pd.DataFrame:
    """prof[username, gender, joined] → thêm gender_id, joined_bucket (khớp data_prep/04).

    gender: map từ meta (else 0=OOV). joined: year → bucket theo COHORT_BINS trong meta;
    NULL/unparseable → cohort mới nhất (id = n_bucket-1)."""
    gmap = meta["user_features"]["gender"]["map"]
    bins = [float(b) for b in meta["user_features"]["joined"]["bins"]]   # [-inf,2012,...,inf]
    newest = len(bins) - 2                                               # id cohort mới nhất (n_bucket-1)

    out = prof.copy()
    out["gender_id"] = out["gender"].map(
        lambda v: gmap.get(v, 0) if pd.notna(v) else 0).astype("int64")
    jyear = pd.to_datetime(out["joined"], errors="coerce").dt.year
    jcode = pd.cut(jyear, bins=bins, labels=False, right=True)           # 0..n_bucket-1
    out["joined_bucket"] = jcode.fillna(newest).astype("int64")
    return out[["username", "gender_id", "joined_bucket"]]
