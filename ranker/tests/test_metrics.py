"""Invariants metrics.py — recall/ndcg pin số tính tay, rank_norm/blend, group_offsets."""
import numpy as np
import pytest

from metrics import blend, eval_pool, group_offsets, rank_norm


def test_group_offsets_dense():
    qid = np.array([0, 0, 0, 1, 1, 2])
    assert group_offsets(qid).tolist() == [0, 3, 5, 6]
    with pytest.raises(AssertionError):
        group_offsets(np.array([0, 0, 2, 2]))          # thiếu qid 1


def test_recall_ndcg_hand_computed():
    # 1 group, 3 cand score desc [3,2,1], label [0,1,0], R_total=2 (1 query ngoài pool)
    scores = np.array([3.0, 2.0, 1.0])
    labels = np.array([0, 1, 0], dtype=np.int8)
    off = np.array([0, 3])
    m = eval_pool(scores, labels, off, np.array([2]), ks=[1, 2, 3])
    assert m["recall@1"] == 0.0
    assert m["recall@2"] == pytest.approx(0.5)         # 1 hit / R=2
    assert m["recall@3"] == pytest.approx(0.5)         # query thứ 2 không trong pool
    dcg2 = 1 / np.log2(3)                              # hit ở rank 2
    idcg2 = 1 + 1 / np.log2(3)                         # min(R,k)=2 relevant đứng đầu
    assert m["ndcg@2"] == pytest.approx(dcg2 / idcg2)
    assert m["n_users"] == 1


def test_idcg_truncation_min_R_k():
    # R=5 > k=2 -> idcg chỉ cộng 2 discount đầu
    scores = np.array([2.0, 1.0])
    labels = np.array([1, 1], dtype=np.int8)
    m = eval_pool(scores, labels, np.array([0, 2]), np.array([5]), ks=[2])
    idcg2 = 1 + 1 / np.log2(3)
    assert m["ndcg@2"] == pytest.approx((1 + 1 / np.log2(3)) / idcg2) == pytest.approx(1.0)
    assert m["recall@2"] == pytest.approx(2 / 5)


def test_pooled_hitrate():
    scores = np.array([2.0, 1.0, 2.0, 1.0])
    labels = np.array([1, 0, 0, 0], dtype=np.int8)
    off = np.array([0, 2, 4])
    m = eval_pool(scores, labels, off, np.array([1, 1]), ks=[1], pooled=True)
    assert m["hitrate@1"] == pytest.approx(0.5)        # 1 hit / 2 pairs
    assert m["n_pairs"] == 2


def test_rank_norm_range_and_monotonic():
    off = np.array([0, 4])
    x = np.array([0.1, 3.0, -2.0, 0.5])
    r = rank_norm(x, off)
    assert r.min() == 0.0 and r.max() == 1.0
    assert (np.argsort(r) == np.argsort(x)).all()      # giữ nguyên thứ tự


def test_blend_alpha0_is_cosine_alpha1_is_pred():
    off = np.array([0, 5])
    cos = np.random.default_rng(0).random(5)
    pred = np.random.default_rng(1).random(5)
    assert (np.argsort(blend(cos, pred, off, 0.0)) == np.argsort(cos)).all()
    assert (np.argsort(blend(cos, pred, off, 1.0)) == np.argsort(pred)).all()
