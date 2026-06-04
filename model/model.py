"""Kiến trúc Two-Tower (xem plan.md). Module importable.

ItemTower: 9 feature item -> concat -> MLP -> L2-norm -> [*, d].
UserTower: pooled-history (mean cached item-vec, rỗng -> h_empty learned) ⊕ gender ⊕ joined
           -> MLP -> L2-norm -> [B, d].
TwoTower: gói 2 tower + item-vec cache cho nhánh history (refresh dày, detach).

User-id embedding: DROP ở v1 (cold-by-user hold-out trọn user).
"""
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


def _mlp(in_dim: int, hidden: List[int], out_dim: int) -> nn.Sequential:
    layers = []
    prev = in_dim
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.ReLU()]
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


def _masked_mean(vecs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """vecs [*, L, D], mask [*, L] bool -> mean theo L bỏ vị trí mask. Rỗng -> 0 (caller xử)."""
    m = mask.unsqueeze(-1).to(vecs.dtype)
    summed = (vecs * m).sum(dim=-2)
    cnt = m.sum(dim=-2).clamp(min=1.0)
    return summed / cnt


class ItemTower(nn.Module):
    """Encode feature item -> vector d (L2-normalized)."""

    CAT_FEATS = [
        ("type", "type_id"), ("source", "source_id"), ("rating", "rating_id"),
        ("demographics", "demographics_id"), ("start_year", "startyear_bucket"),
        ("episodes", "episodes_bucket"),
    ]

    def __init__(self, spec: dict, d: int, genres_proj: int, themes_proj: int, hidden: List[int],
                 num_items: int, id_dim: int, use_item_id: bool):
        super().__init__()
        item = spec["item_features"]
        self.cat_emb = nn.ModuleDict()
        cat_dim = 0
        for name, _col in self.CAT_FEATS:
            s = item[name]
            self.cat_emb[f"cat_{name}"] = nn.Embedding(s["vocab"], s["dim"], padding_idx=0)
            cat_dim += s["dim"]

        gw = item["genres"]["width"]
        tw = item["themes"]["width"]
        self.genres_proj = nn.Linear(gw, genres_proj)
        self.themes_proj = nn.Linear(tw, themes_proj)

        st = item["studios"]
        # KHÔNG padding_idx: emb[0] = token "studio rỗng" (empty_id=0) phải HỌC được
        # (≈28% anime rỗng). Pad cấu trúc khử bằng mask trong forward, không bằng padding_idx.
        self.studio_emb = nn.Embedding(st["vocab"], st["dim"])
        studio_dim = st["dim"]

        # anime-id embedding (collaborative residual). PAD=0 -> zeros (no grad); OOV(1) & real(2..) học.
        self.use_item_id = use_item_id
        in_dim = cat_dim + genres_proj + themes_proj + studio_dim
        if use_item_id:
            self.id_emb = nn.Embedding(num_items, id_dim, padding_idx=0)
            in_dim += id_dim
        self.mlp = _mlp(in_dim, hidden, d)

    def forward(self, items: "ItemBatch", id_idx: torch.Tensor) -> torch.Tensor:
        parts = [self.cat_emb[f"cat_{name}"](items.cat[col]) for name, col in self.CAT_FEATS]
        parts.append(self.genres_proj(items.genres))
        parts.append(self.themes_proj(items.themes))
        # studios: avg-pool embedding. id 0 vừa là pad cấu trúc vừa là empty_id.
        # Invariant (01_item_features.py:88-97): id 0 CHỈ xuất hiện ở list [0] standalone;
        # list non-empty luôn chứa id>=1. Nên row toàn 0 = item rỗng -> bật emb[0] (học được),
        # còn 0 lẻ trong row non-empty = pad -> mask đi.
        st_vec = self.studio_emb(items.studios)                 # [*, S, dim]
        st_mask = items.studios != 0                            # [*, S] pad & empty đều 0
        empty = ~st_mask.any(dim=-1)                            # [*] row toàn 0 = item rỗng
        st_mask[..., 0] = st_mask[..., 0] | empty               # empty -> dùng emb[0]
        parts.append(_masked_mean(st_vec, st_mask))
        if self.use_item_id:
            parts.append(self.id_emb(id_idx))                   # id_idx có thể đã dropout->OOV
        x = torch.cat(parts, dim=-1)
        return F.normalize(self.mlp(x), dim=-1)


class UserTower(nn.Module):
    """pooled-history ⊕ gender ⊕ joined -> MLP -> vector d (L2-normalized)."""

    def __init__(self, spec: dict, d: int, hidden: List[int]):
        super().__init__()
        uf = spec["user_features"]
        self.gender_emb = nn.Embedding(uf["gender"]["vocab"], uf["gender"]["dim"], padding_idx=0)
        self.joined_emb = nn.Embedding(uf["joined"]["vocab"], uf["joined"]["dim"])
        self.h_empty = nn.Parameter(torch.zeros(d))             # learned, thay pooling khi rỗng
        nn.init.normal_(self.h_empty, std=0.02)

        in_dim = d + uf["gender"]["dim"] + uf["joined"]["dim"]
        self.mlp = _mlp(in_dim, hidden, d)

    def forward(self, hist_pooled, gender_id, joined_bucket) -> torch.Tensor:
        x = torch.cat([hist_pooled, self.gender_emb(gender_id), self.joined_emb(joined_bucket)], dim=-1)
        return F.normalize(self.mlp(x), dim=-1)


class ItemBatch:
    """Gom feature tensor 1 batch anime_idx (gather từ ItemTable) cho ItemTower.forward."""

    def __init__(self, cat: Dict[str, torch.Tensor], genres, themes, studios):
        self.cat = cat
        self.genres = genres
        self.themes = themes
        self.studios = studios


class TwoTower(nn.Module):
    def __init__(self, spec: dict, cfg, item_table):
        super().__init__()
        self.d = cfg.d
        self.use_item_id = cfg.use_item_id
        self.id_dropout = cfg.id_dropout
        self.oov_idx = spec["special_idx"]["OOV"]
        self.item_tower = ItemTower(spec, cfg.d, cfg.genres_proj, cfg.themes_proj, cfg.mlp_hidden,
                                    item_table.num_items, cfg.id_dim, cfg.use_item_id)
        self.user_tower = UserTower(spec, cfg.d, cfg.mlp_hidden)
        self.item_table = item_table                            # tensors trên cùng device
        self.register_buffer("item_cache", torch.zeros(item_table.num_items, cfg.d), persistent=False)

    # --- item side ---
    def _gather(self, idx: torch.Tensor) -> ItemBatch:
        it = self.item_table
        cat = {col: it.cat[col][idx] for col in it.CAT_COLS}
        return ItemBatch(cat, it.genres[idx], it.themes[idx], it.studios[idx])

    def encode_items(self, idx: torch.Tensor) -> torch.Tensor:
        """Encode 1 batch anime_idx (có grad). idx shape bất kỳ -> [*, d].
        id-dropout: lúc train mask id->OOV (content vẫn real) -> mô phỏng anime mới + backoff."""
        shape = idx.shape
        flat = idx.reshape(-1)
        id_flat = flat
        if self.use_item_id and self.training and self.id_dropout > 0:
            drop = (torch.rand_like(flat, dtype=torch.float) < self.id_dropout) & (flat >= 2)
            id_flat = torch.where(drop, torch.full_like(flat, self.oov_idx), flat)
        vec = self.item_tower(self._gather(flat), id_flat)
        return vec.reshape(*shape, self.d)

    @torch.no_grad()
    def refresh_item_cache(self, chunk: int = 8192):
        self.eval()
        N = self.item_table.num_items
        for s in range(0, N, chunk):
            e = min(s + chunk, N)
            idx = torch.arange(s, e, device=self.item_cache.device)
            self.item_cache[s:e] = self.item_tower(self._gather(idx), idx)  # real id (warm)
        self.train()

    # --- user side ---
    def pool_history(self, history_ids, history_mask) -> torch.Tensor:
        """Mean cached item-vec theo history (detach, no grad qua history path).
        Row mask rỗng -> h_empty."""
        vecs = self.item_cache[history_ids].detach()           # [B, L, d]
        pooled = _masked_mean(vecs, history_mask)              # [B, d]
        empty = ~history_mask.any(dim=-1)                       # [B]
        pooled = torch.where(empty.unsqueeze(-1), self.user_tower.h_empty.to(pooled.dtype), pooled)
        return pooled

    def encode_users(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        pooled = self.pool_history(batch["history_ids"], batch["history_mask"])
        return self.user_tower(pooled, batch["gender_id"], batch["joined_bucket"])

    def forward(self, batch: Dict[str, torch.Tensor]):
        U = self.encode_users(batch)                           # [B, d]
        V_pos = self.encode_items(batch["pos"])                # [B, d] (grad)
        V_hn = self.encode_items(batch["hardneg_ids"])         # [B, m, d] (grad)
        return U, V_pos, V_hn
