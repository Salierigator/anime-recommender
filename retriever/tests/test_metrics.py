"""Protocol v2: seen-mask (mask seen − query, KHÔNG mask query), pooled hit-rate,
candidate_mask (chế độ chỉ-H), eval_history_batch prefix/cap."""
import numpy as np
import torch

import metrics


class _StubModel:
    """item_cache = eye(N); encode_users trả U cố định theo thứ tự batch."""

    def __init__(self, n_items, U_by_user):
        self.item_cache = torch.eye(n_items)
        self._U = U_by_user                          # dict user -> vector [N]
        self._calls = []

    def eval(self):
        pass

    def train(self):
        pass

    def encode_users(self, ub):
        # thứ tự user trong batch suy từ gender_id đã gài = user_idx
        users = ub["gender_id"].tolist()
        return torch.stack([self._U[u] for u in users])


def _stub_users(num_users, n_items):
    import data as data_mod
    ut = data_mod.UserTable.__new__(data_mod.UserTable)
    ut.num_users = num_users
    ut.hist_vals = np.zeros(1, np.int32)
    ut.hist_offs = np.zeros(num_users + 1, np.int64)     # history rỗng (stub không dùng)
    ut.hist_scores = np.zeros(1, np.int8)
    ut.gender_id = np.arange(num_users, dtype=np.int64)  # gài user_idx vào gender cho stub
    ut.joined_bucket = np.zeros(num_users, np.int64)
    return ut


def _logq(n):
    lq = torch.zeros(n)
    lq[0] = lq[1] = float("-inf")                        # PAD/OOV non-candidate
    return lq


def test_seen_masked_query_not_masked():
    N = 6
    # user 0 thích item2 > item3 > item4 > item5; seen = {2,3}; query = {3}
    U = {0: torch.tensor([0.0, 0.0, 0.9, 0.8, 0.7, 0.6])}
    model = _StubModel(N, U)
    users = _stub_users(1, N)
    queries = {0: [3]}
    mask_ids = metrics.build_masks({0: np.array([2, 3])}, queries)
    assert mask_ids[0].tolist() == [2], "mask phải = seen − query"
    out = metrics.evaluate(model, users, queries, _logq(N), [1, 2], mask_ids)
    # item2 (seen, không phải query) bị mask -> top1 = item3 = query -> recall@1 = 1
    assert out["recall@1"] == 1.0, "query bị mask nhầm hoặc seen không bị mask"


def test_without_mask_seen_pollutes():
    N = 6
    U = {0: torch.tensor([0.0, 0.0, 0.9, 0.8, 0.7, 0.6])}
    model = _StubModel(N, U)
    users = _stub_users(1, N)
    out = metrics.evaluate(model, users, {0: [3]}, _logq(N), [1], {0: np.empty(0, np.int64)})
    assert out["recall@1"] == 0.0, "không mask thì item2 (seen) phải chiếm top1"


def test_pooled_hitrate_and_counts():
    N = 6
    U = {0: torch.tensor([0.0, 0.0, 0.9, 0.8, 0.1, 0.2]),
         1: torch.tensor([0.0, 0.0, 0.1, 0.2, 0.9, 0.8])}
    model = _StubModel(N, U)
    users = _stub_users(2, N)
    queries = {0: [2, 3], 1: [4, 5]}
    masks = {0: np.empty(0, np.int64), 1: np.empty(0, np.int64)}
    out = metrics.evaluate(model, users, queries, _logq(N), [2], masks, pooled=True)
    assert out["n_users"] == 2 and out["n_pairs"] == 4
    assert out["recall@2"] == 1.0 and out["hitrate@2"] == 1.0


def test_candidate_mask_h_only():
    N = 6
    U = {0: torch.tensor([0.0, 0.0, 0.9, 0.8, 0.7, 0.6])}
    model = _StubModel(N, U)
    users = _stub_users(1, N)
    cand = torch.zeros(N, dtype=torch.bool)
    cand[4] = cand[5] = True                             # "H-only": chỉ rank item 4,5
    out = metrics.evaluate(model, users, {0: [4]}, _logq(N), [1], {0: np.empty(0, np.int64)},
                           candidate_mask=cand)
    assert out["recall@1"] == 1.0, "ngoài candidate set phải bị loại (item2,3 không được rank)"


def test_eval_history_batch_prefix_cap(users):
    ids, mask, sc = users.eval_history_batch(np.array([0, 1, 2]), cap=2)
    assert ids.shape[1] == 2
    assert ids[0].tolist() == [2, 3], "prefix = top-by-score (list đã sort desc)"
    assert mask[0].tolist() == [True, True]
    assert ids[1].tolist() == [6, 7] and sc[1].tolist() == [5, 0]
    assert not mask[2].any() and ids[2].tolist() == [0, 0], "row rỗng: pad 0 + mask False"
