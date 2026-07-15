"""InfoNCE + logQ correction + temperature + hard-neg per-anchor (xem thesis-final:docs/TWO_TOWER_MODEL.md).

- s(i,x) = (U_i · V_x)/tau, U,V đã L2-normalize (cosine).
- logQ (× logq_alpha) áp lên in-batch (gồm positive), KHÔNG áp hard-neg.
- mask false-negative: 2 anchor trùng pos_item -> -inf (đừng tự phạt positive của mình).
- mask pad hard-neg -> -inf. beta scale nhánh hard-neg trong mẫu số (cộng log beta).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

NEG_INF = float("-inf")


def info_nce_logq(U, V_pos, V_hn, hardneg_mask, pos_idx, logq, tau, beta, logq_alpha=1.0):
    """
    U          [B, d]      user vec (normalized)
    V_pos      [B, d]      positive item vec (normalized)
    V_hn       [B, m, d]   hard-neg item vec (normalized)
    hardneg_mask [B, m]    bool, True = hard-neg thật
    pos_idx    [B]         anime_idx của positive (cho logQ + false-neg mask)
    logq       [num_items] dense logQ (-inf cho non-candidate)
    logq_alpha float       hệ số correction (1.0 = full, 0 = tắt)
    """
    B = U.shape[0]

    # in-batch logits: cột j = positive của anchor j
    s_in = (U @ V_pos.t()) / tau                          # [B, B]
    if logq_alpha != 0.0:
        s_in = s_in - logq_alpha * logq[pos_idx].unsqueeze(0)  # trừ logQ theo cột (item-as-positive)

    # mask false-negative: i != j nhưng pos[i] == pos[j]
    same = pos_idx.unsqueeze(0) == pos_idx.unsqueeze(1)  # [B, B]
    eye = torch.eye(B, dtype=torch.bool, device=U.device)
    s_in = s_in.masked_fill(same & ~eye, NEG_INF)

    # hard-neg logits: KHÔNG logQ; + log(beta); pad -> -inf
    s_hn = (U.unsqueeze(1) * V_hn).sum(-1) / tau         # [B, m]
    if beta != 1.0:
        s_hn = s_hn + torch.log(torch.tensor(beta, device=U.device, dtype=s_hn.dtype))
    s_hn = s_hn.masked_fill(~hardneg_mask, NEG_INF)

    logits = torch.cat([s_in, s_hn], dim=1)              # [B, B + m]
    target = torch.arange(B, device=U.device)           # positive = diagonal
    return F.cross_entropy(logits, target)
