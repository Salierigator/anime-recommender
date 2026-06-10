"""Model invariants: h_empty, attn NaN-safe, studios empty-vs-pad, id-dropout,
cold_oov cache, history_source cache (no grad) vs embed (grad)."""
import numpy as np
import torch

import data as data_mod


def _user_batch(B=2, L=3, empty=False):
    return {
        "history_ids": torch.zeros(B, L, dtype=torch.long) if empty
        else torch.tensor([[2, 3, 0], [4, 0, 0]]),
        "history_mask": torch.zeros(B, L, dtype=torch.bool) if empty
        else torch.tensor([[True, True, False], [True, False, False]]),
        "history_scores": torch.zeros(B, L, dtype=torch.long),
        "gender_id": torch.tensor([1, 2]),
        "joined_bucket": torch.tensor([0, 3]),
    }


def test_empty_history_h_empty_all_pools(make_model):
    for pool in ["mean", "attn"]:
        model, cfg, _ = make_model(history_pool=pool)
        model.refresh_item_cache()
        pooled = model.pool_history(torch.zeros(2, 3, dtype=torch.long),
                                    torch.zeros(2, 3, dtype=torch.bool))
        assert torch.isfinite(pooled).all(), f"{pool}: NaN với history rỗng"
        assert torch.allclose(pooled[0], model.user_tower.h_empty), f"{pool}: row rỗng phải = h_empty"
        U = model.encode_users(_user_batch(empty=True))
        assert torch.isfinite(U).all()


def test_studios_empty_vs_pad(make_model):
    """Row studios toàn 0 = empty -> dùng emb[0] (đổi emb[0] phải đổi output);
    0 lẻ cuối row non-empty = pad -> mask (đổi emb[0] KHÔNG đổi output)."""
    model, _, _ = make_model()
    model.eval()
    idx_empty, idx_nonempty = torch.tensor([4]), torch.tensor([3])  # studios [0,0] vs [4,0]
    with torch.no_grad():
        before_e = model.encode_items(idx_empty).clone()
        before_n = model.encode_items(idx_nonempty).clone()
        model.item_tower.studio_emb.weight[0] += 5.0
        after_e = model.encode_items(idx_empty)
        after_n = model.encode_items(idx_nonempty)
    assert not torch.allclose(before_e, after_e), "item studios-rỗng phải dùng emb[0] (học được)"
    assert torch.allclose(before_n, after_n), "pad 0 trong row non-empty phải bị mask"


def test_id_dropout_only_train_and_real(make_model):
    model, _, _ = make_model(use_item_id=True, id_dropout=1.0)
    idx = torch.tensor([5, 6])
    model.eval()                                  # eval: id thật
    with torch.no_grad():
        v_eval = model.encode_items(idx).clone()
    model.train()                                 # train + p=1: mọi real id -> OOV
    with torch.no_grad():
        v_drop = model.encode_items(idx)
        v_oov = model.item_tower(model._gather(idx), torch.full_like(idx, model.oov_idx))
    assert torch.allclose(v_drop, v_oov), "id_dropout=1 lúc train phải encode bằng OOV"
    assert not torch.allclose(v_eval, v_drop), "eval phải dùng id thật (khác OOV)"


def test_cold_oov_cache(make_model):
    model, _, table = make_model(use_item_id=True, id_dropout=0.0)
    cold_mask = torch.zeros(table.num_items, dtype=torch.bool)
    cold_mask[5] = True
    model.refresh_item_cache()
    warm5 = model.item_cache[5].clone()
    warm6 = model.item_cache[6].clone()
    model.refresh_item_cache(cold_mask=cold_mask)
    model.eval()
    with torch.no_grad():
        oov5 = model.item_tower(model._gather(torch.tensor([5])),
                                torch.tensor([model.oov_idx]))[0]
    assert torch.allclose(model.item_cache[5], oov5), "item cold phải encode id->OOV"
    assert not torch.allclose(model.item_cache[5], warm5), "cache cold phải khác warm (id real)"
    assert torch.allclose(model.item_cache[6], warm6), "item warm không đổi"


def test_history_source_grad(make_model):
    # cache: KHÔNG grad về item tower qua đường history
    model, _, _ = make_model(history_source="cache")
    model.refresh_item_cache()
    U = model.encode_users(_user_batch())
    U.sum().backward()
    assert all(p.grad is None for p in model.item_tower.parameters()), \
        "cache mode: item tower không được nhận grad từ user side"

    # embed: grad chảy về bảng hist_emb
    model2, _, _ = make_model(history_source="embed")
    U2 = model2.encode_users(_user_batch())
    U2.sum().backward()
    g = model2.hist_emb.weight.grad
    assert g is not None and g.abs().sum() > 0, "embed mode: hist_emb phải nhận grad"
    assert g[0].abs().sum() == 0, "padding_idx=0 không được nhận grad"
