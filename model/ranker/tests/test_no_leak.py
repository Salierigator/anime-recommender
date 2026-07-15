"""Invariants chống leak — chạy trên artifacts + pool/dataset thật (slice nhỏ, nhanh).

Skip nếu chưa build (fresh clone): cần artifacts/ + model/ranker/train-data/pools + datasets.
"""
import numpy as np
import polars as pl
import pytest

import config
from build_train import split_support_target
from pool import UsersHistory, load_eval_seen, load_queries

needs_artifacts = pytest.mark.skipif(
    not (config.ARTIFACTS / "users_history.parquet").exists(), reason="cần artifacts/")
needs_pools = pytest.mark.skipif(
    not (config.POOLS / "eval_val.parquet").exists(), reason="cần build_eval.py")
needs_train = pytest.mark.skipif(
    not (config.DATASETS / "train.parquet").exists(), reason="cần build_train.py")


@pytest.fixture(scope="module")
def uh():
    return UsersHistory()


@needs_artifacts
def test_split_support_target_disjoint_and_deterministic(uh):
    rng = np.random.default_rng(0)
    train_ids = pl.read_parquet(config.ARTIFACTS / "user_split.parquet") \
        .filter(pl.col("split") == "train")["user_idx"].to_numpy()
    sample = rng.choice(train_ids[uh.history_lens()[train_ids] >= config.MIN_POS_TRAIN],
                        50, replace=False)
    for u in sample:
        ids, sc = uh.history(int(u))
        si, ss, ti, ts = split_support_target(ids, sc, int(u))
        assert len(np.intersect1d(si, ti)) == 0, "support ∩ target phải rỗng"
        assert len(si) + len(ti) == len(ids)
        assert len(ti) >= 1 and len(si) >= 1
        assert (ss[:-1] >= ss[1:]).all(), "support phải giữ sort score desc"
        si2, _, ti2, _ = split_support_target(ids, sc, int(u))
        assert (si == si2).all() and (ti == ti2).all(), "split phải deterministic theo user"


@needs_train
def test_train_pool_no_cold_no_eval_users_labels_on_targets(uh):
    df = pl.read_parquet(config.DATASETS / "train.parquet").filter(pl.col("qid") < 50)
    is_cold = pl.read_parquet(config.ARTIFACTS / "item_index.parquet")["is_cold"].to_numpy()
    assert not is_cold[df["anime_idx"].to_numpy()].any(), "train pool không được chứa cold"

    split = pl.read_parquet(config.ARTIFACTS / "user_split.parquet")
    smap = dict(zip(split["user_idx"].to_numpy(), split["split"].to_list()))
    assert all(smap[int(u)] == "train" for u in df["user_idx"].unique().to_numpy())

    for (u,), g in df.group_by("user_idx"):
        ids, sc = uh.history(int(u))
        si, _, ti, ts = split_support_target(ids, sc, int(u))
        tgt = dict(zip(ti.tolist(), config.grade(ts).tolist()))
        cand = g["anime_idx"].to_numpy()
        lab = g["label"].to_numpy()
        assert not np.isin(cand, si).any(), "support (seen) phải bị mask khỏi pool"
        exp = np.array([tgt.get(int(a), 0) for a in cand], dtype=np.int8)
        assert (lab == exp).all(), "label phải = grade(target) / 0"
        assert (lab > 0).any(), "group giữ lại phải có ≥1 positive"


@needs_pools
def test_eval_pool_mask_and_labels():
    df = pl.read_parquet(config.POOLS / "eval_val.parquet").filter(pl.col("qid") < 50)
    seen = load_eval_seen()
    queries, _ = load_queries("val")
    for (u,), g in df.group_by("user_idx"):
        cand = g["anime_idx"].to_numpy()
        q = queries[int(u)]
        mask = np.setdiff1d(seen[int(u)], q)
        assert not np.isin(cand, mask).any(), "pool không được chứa item bị mask (seen−query)"
        assert (g["label"].to_numpy() == np.isin(cand, q)).all(), "label = cand ∈ query"
        assert (cand >= 2).all(), "PAD/OOV không được vào pool"


@needs_pools
def test_eval_pool_sorted_by_cosine_and_r_total():
    df = pl.read_parquet(config.POOLS / "eval_val.parquet").filter(pl.col("qid") < 20)
    users = pl.read_parquet(config.POOLS / "eval_val_users.parquet").filter(pl.col("qid") < 20)
    queries, _ = load_queries("val")
    for (q,), g in df.group_by("qid"):
        cos = g.sort("pool_rank")["cos_uv"].to_numpy()
        assert (cos[:-1] >= cos[1:] - 1e-6).all(), "trong group phải sort cosine desc"
    for u, rt in zip(users["user_idx"].to_numpy(), users["r_total"].to_numpy()):
        assert rt == len(queries[int(u)]), "r_total = TỔNG query (kể cả ngoài pool)"
