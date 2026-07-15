"""Fixtures synthetic cho test invariants — KHÔNG cần train-data/ (chạy ở mọi máy).

Mô phỏng spec/ItemTable/UserTable nhỏ, đúng các invariant của artifacts thật:
anime_idx 0=PAD/1=OOV/real>=2; studios id 0 chỉ xuất hiện dạng [0,...] standalone
(row toàn 0 = empty) hoặc pad cuối row non-empty; history ragged sort score desc.
"""
import copy
import pathlib
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

import config as cfg_mod  # noqa: E402
import data as data_mod  # noqa: E402
import model as M  # noqa: E402

_SPEC = {
    "special_idx": {"PAD": 0, "OOV": 1, "first_real": 2},
    "item_features": {
        "type": {"vocab": 4, "dim": 2}, "source": {"vocab": 4, "dim": 2},
        "rating": {"vocab": 4, "dim": 2}, "demographics": {"vocab": 4, "dim": 2},
        "start_year": {"vocab": 4, "dim": 2}, "episodes": {"vocab": 4, "dim": 2},
        "genres": {"width": 5}, "themes": {"width": 6},
        "studios": {"vocab": 6, "dim": 4, "empty_id": 0, "oov_id": 1},
    },
    "user_features": {"gender": {"vocab": 4, "dim": 2}, "joined": {"vocab": 5, "dim": 2}},
}

NUM_ITEMS = 10


def _make_item_table(num_items=NUM_ITEMS):
    it = data_mod.ItemTable.__new__(data_mod.ItemTable)
    it.num_items = num_items
    g = torch.Generator().manual_seed(0)
    it.cat = {c: torch.randint(0, 4, (num_items,), generator=g) for c in data_mod.ItemTable.CAT_COLS}
    it.genres = torch.randint(0, 2, (num_items, 5), generator=g).float()
    it.themes = torch.randint(0, 2, (num_items, 6), generator=g).float()
    # row 0/1 (PAD/OOV) + row 4: studios rỗng = [0,0]; còn lại id>=2 pad 0 cuối
    it.studios = torch.tensor([
        [0, 0], [0, 0], [2, 3], [4, 0], [0, 0], [5, 2], [3, 0], [2, 0], [4, 5], [3, 0],
    ]).long()
    # synopsis synthetic (raw_dim 16): PAD/OOV (0,1) = zero + low_info; row 4 = real low_info
    # (test nhánh no_synopsis); còn lại có embedding "thật". L2-norm như artifact.
    it.synopsis_emb = torch.nn.functional.normalize(
        torch.randn(num_items, 16, generator=g), dim=-1)
    it.synopsis_emb[0] = 0.0
    it.synopsis_emb[1] = 0.0
    low = torch.zeros(num_items, dtype=torch.bool)
    low[0] = low[1] = low[4] = True
    it.synopsis_low_info = low
    return it


def _make_users():
    """3 user: u0 history [2,3,4,5] (scores 10,9,8,7); u1 [6,7] (5,0); u2 rỗng."""
    ut = data_mod.UserTable.__new__(data_mod.UserTable)
    ut.num_users = 3
    ut.hist_vals = np.array([2, 3, 4, 5, 6, 7], np.int32)
    ut.hist_offs = np.array([0, 4, 6, 6], np.int64)
    ut.hist_scores = np.array([10, 9, 8, 7, 5, 0], np.int8)
    ut.gender_id = np.array([1, 0, 2], np.int64)
    ut.joined_bucket = np.array([0, 4, 2], np.int64)
    ut.hardneg_pad = np.array([[8, 9, 0], [2, 0, 0], [0, 0, 0]], np.int32)
    ut.hardneg_len = np.array([2, 1, 0], np.int64)
    ut.split = np.array(["train", "val", "test"], dtype=object)
    return ut


@pytest.fixture
def spec():
    return copy.deepcopy(_SPEC)


@pytest.fixture
def users():
    return _make_users()


@pytest.fixture
def make_cfg():
    def _make(**kw):
        kw.setdefault("d", 8)
        kw.setdefault("mlp_hidden", [16])
        kw.setdefault("genres_proj", 4)
        kw.setdefault("themes_proj", 4)
        kw.setdefault("id_dim", 8)
        return cfg_mod.TwoTowerConfig(**kw)
    return _make


@pytest.fixture
def make_model(make_cfg, spec):
    def _make(**kw):
        cfg = make_cfg(**kw)
        table = _make_item_table()
        torch.manual_seed(0)
        return M.TwoTower(spec, cfg, table), cfg, table
    return _make
