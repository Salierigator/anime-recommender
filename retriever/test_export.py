"""Test artifacts của export.py: dựng lại user-encoder TỪ artifacts (như service sẽ làm)
-> chạy eval cold-by-user trên tập test (+ val để đối chiếu) -> in recall@K / ndcg@K.

Firewall-faithful: chỉ đọc `artifacts/` (item_vectors.npy + user_tower.pt) + import *định nghĩa*
model (UserTower, _masked_mean) — KHÔNG load best.pt để encode. Tái dùng nguyên `metrics.evaluate`
nên số liệu so sánh trực tiếp được với lúc train.

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
from data import ExamplesDataset, UserTable, load_feature_spec, load_logq  # noqa: E402
from metrics import evaluate, group_examples  # noqa: E402
from model import UserTower, _masked_mean  # noqa: E402


class ArtifactEncoder:
    """Bắt chước interface model mà metrics.evaluate cần (.item_cache/.encode_users/.eval/.train),
    nhưng chạy hoàn toàn bằng artifacts. Pool history = masked-mean qua item_vectors (rỗng -> h_empty)."""

    def __init__(self, out: Path, device: str):
        ut = torch.load(out / "user_tower.pt", map_location=device, weights_only=False)
        if ut["history_pool"] != "mean" or ut["score_pool"] != "none":
            raise SystemExit(
                f"test_export chỉ hỗ trợ history_pool='mean'/score_pool='none' "
                f"(artifact: {ut['history_pool']}/{ut['score_pool']}) — mở rộng pooling nếu config đổi.")
        self.item_cache = torch.from_numpy(np.load(out / "item_vectors.npy")).to(device)
        self.tower = UserTower({"user_features": ut["user_features"]}, ut["d"], ut["mlp_hidden"]).to(device)
        self.tower.load_state_dict(
            {k.replace("user_tower.", ""): v for k, v in ut["state_dict"].items()})
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


def queries_for(split: str, users: UserTable) -> dict:
    """Examples của split -> {user_idx: [anime_idx query]}, chỉ giữ user có trong split đó."""
    ex = ExamplesDataset(config.TRAIN_DATA, split)
    return group_examples(np.asarray(ex.user_idx), np.asarray(ex.anime_idx))


def fmt(m: dict, ks) -> str:
    cols = " | ".join(f"recall@{k} {m[f'recall@{k}']:.4f}  ndcg@{k} {m[f'ndcg@{k}']:.4f}" for k in ks)
    return f"{cols}   (n_users={m['n_users']})"


def main():
    ap = argparse.ArgumentParser(description="Eval artifacts export trên test/val split")
    ap.add_argument("--out", type=Path, default=config.ROOT.parent / "artifacts")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    ks = [10, 50, 100]
    spec = load_feature_spec(config.TRAIN_DATA)
    users = UserTable(config.TRAIN_DATA, spec["k_history"], spec["hard_neg_cap"])
    logq = load_logq(config.TRAIN_DATA).to(args.device)
    enc = ArtifactEncoder(args.out, args.device)

    print(f"Eval bằng artifacts ({args.out}) — encoder dựng từ user_tower.pt + item_vectors.npy\n")
    for split in ("test", "val"):
        m = evaluate(enc, users, queries_for(split, users), logq, ks)
        print(f"[{split:4s}] {fmt(m, ks)}")

    # đối chiếu: best.pt ghi metric trên split nào (config.eval_split) lúc train
    ckpt_path = config.ROOT / "checkpoints" / "best.pt"
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        ref = ck.get("metrics", {})
        if ref:
            cols = " | ".join(f"recall@{k} {ref.get(f'recall@{k}', float('nan')):.4f}  "
                              f"ndcg@{k} {ref.get(f'ndcg@{k}', float('nan')):.4f}" for k in ks)
            print(f"\n[ref ] best.pt recorded (split='{ck['cfg'].eval_split}', "
                  f"epoch={ck.get('epoch')}): {cols}")
            print("       -> [val] qua artifacts phải ~khớp dòng ref này (xác nhận export trung thực).")


if __name__ == "__main__":
    main()
