"""Load artifacts train-data/ + Dataset + collate (xem docs/TRAIN_DATA.md §6).

Tất cả artifact nhỏ -> load 1 lần vào RAM. Collate build batch tensor vectorized:
history (gỡ anchor + dropout), hard-neg per-user (sample m + mask), gender/joined.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset


def load_feature_spec(train_data: Path) -> dict:
    with open(train_data / "feature_spec.json") as f:
        return json.load(f)


def load_logq(train_data: Path) -> torch.Tensor:
    """logq.npy dense theo anime_idx (PAD/OOV/non-candidate = -inf)."""
    arr = np.load(train_data / "logq.npy")
    return torch.from_numpy(arr.astype(np.float32))


def _pad_lists(col: List, width: int, dtype=np.int32) -> np.ndarray:
    """List[List[int]] -> ma trận [n, width] pad 0 (cắt nếu dài hơn width)."""
    out = np.zeros((len(col), width), dtype=dtype)
    for i, lst in enumerate(col):
        if lst is None:
            continue
        v = np.asarray(lst, dtype=dtype)[:width]
        out[i, : len(v)] = v
    return out


class ItemTable:
    """Feature item theo anime_idx (row 0=PAD,1=OOV neutral). Tensor để gather ở tower."""

    CAT_COLS = [
        "type_id", "source_id", "rating_id", "demographics_id",
        "startyear_bucket", "episodes_bucket",
    ]

    def __init__(self, train_data: Path):
        t = pq.read_table(train_data / "item_features.parquet")
        d = t.to_pydict()
        self.num_items = len(d["anime_idx"])
        assert d["anime_idx"] == list(range(self.num_items)), "item_features phải dense 0..N"

        self.cat = {c: torch.tensor(d[c], dtype=torch.long) for c in self.CAT_COLS}
        self.genres = torch.from_numpy(
            np.asarray([list(x) for x in d["genres_multihot"]], dtype=np.float32))
        self.themes = torch.from_numpy(
            np.asarray([list(x) for x in d["themes_multihot"]], dtype=np.float32))
        studio_lists = [list(x) if x is not None else [] for x in d["studio_ids"]]
        max_studios = max((len(x) for x in studio_lists), default=1) or 1
        self.studios = torch.from_numpy(_pad_lists(studio_lists, max_studios)).long()  # [N,S] pad 0

    def to(self, device):
        self.cat = {k: v.to(device) for k, v in self.cat.items()}
        self.genres = self.genres.to(device)
        self.themes = self.themes.to(device)
        self.studios = self.studios.to(device)
        return self


class UserTable:
    """Feature + history + hard_neg theo user_idx. Padded array để collate gather nhanh."""

    def __init__(self, train_data: Path, k_history: int, hard_neg_cap: int):
        t = pq.read_table(train_data / "users.parquet")
        d = t.to_pydict()
        self.num_users = len(d["user_idx"])
        assert d["user_idx"] == list(range(self.num_users)), "users phải dense 0..U"

        self.split = np.asarray(d["split"], dtype=object)
        self.gender_id = np.asarray(d["gender_id"], dtype=np.int64)
        self.joined_bucket = np.asarray(d["joined_bucket"], dtype=np.int64)

        self.history_pad = _pad_lists(d["history_ids"], k_history)            # [U,30] pad 0
        self.hardneg_pad = _pad_lists(d["hard_neg_ids"], hard_neg_cap)        # [U,64] pad 0
        self.hardneg_len = np.asarray(
            [min(len(x) if x is not None else 0, hard_neg_cap) for x in d["hard_neg_ids"]],
            dtype=np.int64,
        )


class ExamplesDataset(Dataset):
    """(user_idx, anime_idx) positive cho 1 split. subset: chỉ lấy N đầu (smoke)."""

    def __init__(self, train_data: Path, split: str, subset: int = None):
        t = pq.read_table(train_data / "examples" / f"split={split}" / "part-0.parquet")
        self.user_idx = t.column("user_idx").to_numpy()
        self.anime_idx = t.column("anime_idx").to_numpy()
        if subset is not None:
            self.user_idx = self.user_idx[:subset]
            self.anime_idx = self.anime_idx[:subset]

    def __len__(self):
        return len(self.user_idx)

    def __getitem__(self, i):
        return int(self.user_idx[i]), int(self.anime_idx[i])


class Collate:
    """Build batch tensor vectorized: history (gỡ anchor + dropout), hard-neg per-user
    (sample m phân biệt + mask), gender/joined.

    Là class (không phải closure) để PICKLE được -> num_workers>0 chạy trên cả macOS spawn
    lẫn Linux fork. RNG seed lazily theo torch.initial_seed() (riêng mỗi worker & mỗi epoch,
    suy ra tất định từ torch seed) -> randomness độc lập + reproducible.
    """

    def __init__(self, users: UserTable, hist_dropout: float, m_hardneg: int):
        self.history_pad = users.history_pad
        self.hardneg_pad = users.hardneg_pad
        self.hardneg_len = users.hardneg_len
        self.gender_id = users.gender_id
        self.joined_bucket = users.joined_bucket
        self.col_idx = np.arange(users.hardneg_pad.shape[1])
        self.hist_dropout = hist_dropout
        self.m_hardneg = m_hardneg
        self._rng = None                                          # None lúc pickle; tạo trong worker

    def __call__(self, batch) -> Dict[str, torch.Tensor]:
        if self._rng is None:
            self._rng = np.random.default_rng(torch.initial_seed())
        rng = self._rng
        u = np.fromiter((b[0] for b in batch), dtype=np.int64, count=len(batch))
        pos = np.fromiter((b[1] for b in batch), dtype=np.int64, count=len(batch))
        B = len(batch)

        # history: gỡ anchor (pos) + bỏ pad
        hist = self.history_pad[u]                                 # [B,30]
        hist_mask = (hist != 0) & (hist != pos[:, None])
        # history dropout: bỏ toàn bộ history của 1 phần example -> ép h_empty
        if self.hist_dropout > 0:
            drop = rng.random(B) < self.hist_dropout
            hist_mask[drop] = False

        # hard-neg: sample m item PHÂN BIỆT từ pool của chính user (without-replacement).
        # Thiếu (lens < m) -> phần dư là PAD + mask False (đừng bịa hard-neg).
        lens = self.hardneg_len[u]                                 # [B]
        hn_pool = self.hardneg_pad[u]                              # [B, cap]
        keys = rng.random((B, hn_pool.shape[1]))
        keys[self.col_idx[None, :] >= lens[:, None]] = np.inf     # cột pad -> sort xuống cuối
        chosen = np.argsort(keys, axis=1)[:, :self.m_hardneg]     # [B,m] m cột valid random
        hn_ids = np.take_along_axis(hn_pool, chosen, axis=1)      # [B,m]
        hn_mask = np.arange(self.m_hardneg)[None, :] < lens[:, None]  # [B,m] True = hard-neg thật

        return {
            "user_idx": torch.from_numpy(u),
            "pos": torch.from_numpy(pos),
            "history_ids": torch.from_numpy(hist).long(),
            "history_mask": torch.from_numpy(hist_mask),
            "hardneg_ids": torch.from_numpy(hn_ids).long(),
            "hardneg_mask": torch.from_numpy(hn_mask),
            "gender_id": torch.from_numpy(self.gender_id[u]),
            "joined_bucket": torch.from_numpy(self.joined_bucket[u]),
        }
