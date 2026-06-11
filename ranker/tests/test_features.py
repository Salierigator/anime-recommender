"""Invariants features.py — thứ tự FEATURE_NAMES, cold policy, determinism (artifacts thật)."""
import numpy as np
import pytest

import config
from features import CAT_COLS, FEATURE_NAMES, N_CROSS, ItemFeatures, build_frame

pytestmark = pytest.mark.skipif(
    not (config.ARTIFACTS / "item_index.parquet").exists(), reason="cần artifacts/")


@pytest.fixture(scope="module")
def itemfeat():
    return ItemFeatures.load(config.ARTIFACTS, config.CLEANED)


def test_build_frame_column_order(itemfeat):
    cand = np.array([2, 3, 4])
    cross = {n: np.zeros(3, np.float32) for n in FEATURE_NAMES[:N_CROSS]}
    df = build_frame(itemfeat, cand, cross)
    assert list(df.columns) == FEATURE_NAMES


def test_cold_policy_no_leak(itemfeat):
    cold = itemfeat.is_cold
    assert cold.sum() > 0
    assert itemfeat.item["mal_score_missing"][cold].all(), "cold phải flag missing"
    assert (itemfeat.item["log_scored_by"][cold] == 0).all()
    assert (itemfeat.item["log_members"][cold] == 0).all()
    assert (itemfeat.item["log_favorites"][cold] == 0).all()
    assert itemfeat.item["rank_missing"][cold].all()
    # imputed về median warm — mọi row cold cùng 1 giá trị
    assert len(np.unique(itemfeat.item["mal_score"][cold])) == 1
    assert len(np.unique(itemfeat.item["popularity"][cold])) == 1


def test_categorical_codes_deterministic(itemfeat):
    again = ItemFeatures.load(config.ARTIFACTS, config.CLEANED)
    for c in CAT_COLS:
        assert (itemfeat.item[c] == again.item[c]).all()
    assert (itemfeat.genres == again.genres).all()


def test_affinity_shapes(itemfeat):
    cand = np.array([[2, 3], [4, 5]])                  # batched [2, 2]
    g_pref = np.zeros((2, itemfeat.genres.shape[1]), np.float32)
    t_pref = np.zeros((2, itemfeat.themes.shape[1]), np.float32)
    ga, ta, go = itemfeat.affinity(cand, g_pref, t_pref)
    assert ga.shape == ta.shape == go.shape == (2, 2)
    assert (ga == 0).all()
