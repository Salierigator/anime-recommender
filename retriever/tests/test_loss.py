"""Loss invariants: false-neg mask, pad hard-neg -inf, logq_alpha, không NaN."""
import torch

from loss import info_nce_logq


def _unit(*rows):
    v = torch.tensor(rows, dtype=torch.float32)
    return torch.nn.functional.normalize(v, dim=-1)


def test_false_negative_mask_zero_loss():
    """2 anchor trùng pos + cùng U/V, không hard-neg: off-diagonal bị mask -> mẫu số
    chỉ còn chính positive -> loss = 0 (không mask sẽ là log(2))."""
    U = _unit([1.0, 0.0], [1.0, 0.0])
    V = U.clone()
    V_hn = torch.zeros(2, 1, 2)
    hn_mask = torch.zeros(2, 1, dtype=torch.bool)
    pos = torch.tensor([5, 5])
    logq = torch.zeros(10)
    loss = info_nce_logq(U, V, V_hn, hn_mask, pos, logq, tau=0.1, beta=1.0)
    assert torch.isfinite(loss)
    assert loss.item() < 1e-6, f"false-neg phải bị mask (loss={loss.item()})"


def test_pad_hardneg_ignored():
    """Đổi giá trị V_hn ở cột bị mask không được đổi loss."""
    torch.manual_seed(0)
    U = _unit([1.0, 0.2], [0.3, 1.0])
    V = _unit([0.9, 0.1], [0.2, 1.1])
    pos = torch.tensor([3, 4])
    logq = torch.zeros(10)
    hn_mask = torch.tensor([[True, False], [False, False]])
    hn_a = torch.randn(2, 2, 2)
    hn_b = hn_a.clone()
    hn_b[0, 1], hn_b[1] = 99.0, -99.0                  # chỉ sửa cột mask=False
    la = info_nce_logq(U, V, hn_a, hn_mask, pos, logq, 0.1, 1.0)
    lb = info_nce_logq(U, V, hn_b, hn_mask, pos, logq, 0.1, 1.0)
    assert torch.allclose(la, lb), "pad hard-neg phải bị -inf (không ảnh hưởng loss)"
    assert torch.isfinite(la)


def test_logq_alpha():
    torch.manual_seed(0)
    U = _unit([1.0, 0.2], [0.3, 1.0])
    V = _unit([0.9, 0.1], [0.2, 1.1])
    pos = torch.tensor([3, 4])
    hn = torch.zeros(2, 1, 2)
    hn_mask = torch.zeros(2, 1, dtype=torch.bool)
    logq = torch.full((10,), -3.0)
    logq[3], logq[4] = -1.0, -6.0                      # lệch nhau để alpha có tác dụng
    l_full = info_nce_logq(U, V, hn, hn_mask, pos, logq, 0.1, 1.0, logq_alpha=1.0)
    l_off = info_nce_logq(U, V, hn, hn_mask, pos, logq, 0.1, 1.0, logq_alpha=0.0)
    l_zeroq = info_nce_logq(U, V, hn, hn_mask, pos, torch.zeros(10), 0.1, 1.0, logq_alpha=1.0)
    assert not torch.allclose(l_full, l_off), "alpha=0 phải tắt logQ"
    assert torch.allclose(l_off, l_zeroq), "alpha=0 tương đương logq=0"


def test_all_hardneg_masked_no_nan():
    U = _unit([1.0, 0.0], [0.0, 1.0])
    V = _unit([1.0, 0.1], [0.1, 1.0])
    loss = info_nce_logq(U, V, torch.zeros(2, 3, 2), torch.zeros(2, 3, dtype=torch.bool),
                         torch.tensor([2, 3]), torch.zeros(10), 0.07, 1.0)
    assert torch.isfinite(loss), "toàn hard-neg pad không được sinh NaN"
