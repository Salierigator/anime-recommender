"""Test artifacts của export.py: dựng lại user-encoder TỪ artifacts (như service sẽ làm)
-> chạy eval protocol đầy đủ (mask = seen − query) trên warm val/test + cold val.

Firewall-faithful: chỉ đọc `artifacts/` (item_vectors.npy + user_tower.pt) + import *định nghĩa*
model (UserTower, _masked_mean) — KHÔNG load best.pt để encode. Tái dùng nguyên `metrics.evaluate`
nên số liệu so sánh trực tiếp được với lúc train.

Đọc kết quả:
  - [val warm] ~khớp metrics ghi trong best.pt (lệch nhẹ vài phần nghìn là bình thường:
    artifact cache encode row H bằng OOV — serve-path — còn lúc train warm cache dùng id thật).
  - [val cold] phải ~khớp log run_cold_eval (cell cold trong notebook) — cache giống hệt.
  - test_cold KHÔNG đụng ở đây (final exam, chấm 1 lần khi chốt model).

    python retriever/test_export.py [--out artifacts] [--device cpu]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

SRC = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC))

import config  # noqa: E402
from data import UserTable, load_feature_spec, load_logq  # noqa: E402
from metrics import evaluate, load_eval_protocol  # noqa: E402
from model import UserTower, _masked_mean  # noqa: E402


class ArtifactEncoder:
    """Bắt chước interface model mà metrics.evaluate cần (.item_cache/.encode_users/.eval/.train),
    nhưng chạy hoàn toàn bằng artifacts. Pool history = masked-mean qua item_vectors (rỗng -> h_empty)."""

    def __init__(self, out: Path, device: str):
        ut = torch.load(out / "user_tower.pt", map_location=device, weights_only=False)
        if (ut["history_pool"], ut["score_pool"], ut.get("history_source", "cache")) != ("mean", "none", "cache"):
            raise SystemExit(
                f"test_export chỉ hỗ trợ history_pool=mean/score_pool=none/history_source=cache "
                f"(artifact: {ut['history_pool']}/{ut['score_pool']}/{ut.get('history_source')}) "
                "— mở rộng pooling nếu config đổi.")
        self.eval_history_cap = ut.get("eval_history_cap", 1024)
        self.item_cache = torch.from_numpy(np.load(out / "item_vectors.npy")).to(device)
        self.tower = UserTower({"user_features": ut["user_features"]}, ut["d"], ut["mlp_hidden"]).to(device)
        self.tower.load_state_dict(
            {k.replace("user_tower.", ""): v for k, v in ut["state_dict"].items()
             if k.startswith("user_tower.")})
        self.tower.eval()

    def eval(self):
        return self

    def train(self):
        return self

    @torch.no_grad()
    def encode_users(self, batch):
        vecs = self.item_cache[batch["history_ids"]]               # [B, L, d]
        pooled = _masked_mean(vecs, batch["history_mask"])         # [B, d]
        empty = ~batch["history_mask"].any(dim=-1)
        pooled = torch.where(empty.unsqueeze(-1), self.tower.h_empty, pooled)
        return self.tower(pooled, batch["gender_id"], batch["joined_bucket"])


def fmt(m: dict, ks) -> str:
    cols = " ".join(f"r@{k}={m[f'recall@{k}']:.4f}" for k in ks)
    nd = " ".join(f"ndcg@{k}={m[f'ndcg@{k}']:.4f}" for k in (10, 200))
    return f"{cols}  {nd}  (n_users={m['n_users']:,})"


def main():
    ap = argparse.ArgumentParser(description="Eval artifacts export theo protocol đầy đủ")
    ap.add_argument("--out", type=Path, default=config.ROOT.parent / "artifacts")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    ks = [10, 50, 100, 200, 500]
    spec = load_feature_spec(config.TRAIN_DATA)
    users = UserTable(config.TRAIN_DATA, spec["hard_neg_cap"])
    logq = load_logq(config.TRAIN_DATA).to(args.device)
    enc = ArtifactEncoder(args.out, args.device)
    cap = enc.eval_history_cap

    print(f"Eval bằng artifacts ({args.out}) — encoder từ user_tower.pt + item_vectors.npy "
          f"(history cap {cap})\n")
    for split in ("val", "test"):
        q_warm, m_warm, q_cold, m_cold = load_eval_protocol(config.TRAIN_DATA, split)
        m = evaluate(enc, users, q_warm, logq, ks, m_warm, eval_history_cap=cap)
        print(f"[{split:4s} warm] {fmt(m, ks)}")
        if split == "val":  # cold chỉ chấm val — test_cold là final exam
            mc = evaluate(enc, users, q_cold, logq, ks, m_cold, eval_history_cap=cap, pooled=True)
            print(f"[val  cold] {fmt(mc, ks)}")
            print("            hitrate " + " ".join(f"@{k}={mc[f'hitrate@{k}']:.4f}" for k in ks)
                  + f"  (n_pairs={mc['n_pairs']:,})")

    ckpt_path = config.ROOT / "checkpoints" / "best.pt"
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        ref = ck.get("metrics", {})
        if ref:
            cols = " ".join(f"r@{k}={ref.get(f'recall@{k}', float('nan')):.4f}" for k in ks)
            print(f"\n[ref ] best.pt recorded (val warm, epoch={ck.get('epoch')}): {cols}")
            print("       -> [val warm] qua artifacts ~khớp dòng này (lệch nhẹ do row H = OOV);"
                  " [val cold] ~khớp log run_cold_eval.")


if __name__ == "__main__":
    main()
