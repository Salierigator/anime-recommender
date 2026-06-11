"""train_nn.py — neural ranker (so sánh với GBDT — Colab GPU; --smoke local CPU/MPS).

Per candidate: [V, U, U⊙V, tabular z-scored, cat embeddings, DIN-attention(hist_top64, query=V)]
→ MLP [512, 256, 1]. Loss listwise softmax CE trên group, gain 2^grade − 1. Early stop trên
two-stage val ndcg@10 @ best α (metrics.py — số CHÍNH THỨC, không proxy).

Checkpoint .pt tự chứa (state_dict + z-stats + vocab) → eval.py::predict gọi predict_pool.

    venv/bin/python ranker/src/train_nn.py --smoke
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from features import CAT_COLS, FEATURE_NAMES
from metrics import load_pool_arrays, sweep_best_alpha

NUM_COLS = [f for f in FEATURE_NAMES if f not in CAT_COLS]
ALPHAS = [0.25, 0.4, 0.5, 0.6, 0.75, 1.0]
KS = [10, 50, 100, 200]
D = 128


def auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class GroupData:
    """Pool parquet -> mảng group-major [G, C, ...] (C = group size cố định)."""

    def __init__(self, pool_path, users_path, k: int, mean=None, std=None, cat_vocabs=None,
                 max_groups: int | None = None):
        df, cos, lab, off, rt, users = load_pool_arrays(pool_path, users_path, k,
                                                        max_groups=max_groups)
        sizes = np.diff(off)
        assert (sizes == sizes[0]).all(), "group size phải đồng nhất"
        G, C = len(sizes), int(sizes[0])
        self.G, self.C = G, C
        self.cos, self.r_total, self.off, self.labels_flat = cos, rt, off, lab

        # NaN (vd u_account_age thiếu profile) hợp lệ với LightGBM nhưng NN cần impute:
        # z-score bằng nanmean/nanstd rồi nan→0 (= mean)
        Xnum = df.select(NUM_COLS).to_numpy().astype(np.float32).reshape(G, C, -1)
        flat = Xnum.reshape(-1, Xnum.shape[-1])
        self.mean = np.nanmean(flat, 0) if mean is None else mean
        self.std = (np.nanstd(flat, 0) + 1e-6) if std is None else std
        self.num = np.nan_to_num((Xnum - self.mean) / self.std)
        cats = df.select(CAT_COLS).to_numpy().astype(np.int64).reshape(G, C, -1)
        self.cat_vocabs = (cats.reshape(-1, cats.shape[-1]).max(0) + 1).tolist() \
            if cat_vocabs is None else cat_vocabs
        self.cats = np.minimum(cats, np.asarray(self.cat_vocabs) - 1)   # clip code lạ
        self.anime = df["anime_idx"].to_numpy().astype(np.int64).reshape(G, C)
        self.label = df["label"].to_numpy().astype(np.float32).reshape(G, C)
        self.U = np.stack([np.asarray(u, np.float32) for u in users["U"].to_list()])
        h = users["hist_top64"].to_list()
        H = max(max((len(x) for x in h), default=1), 1)
        self.hist = np.zeros((G, H), dtype=np.int64)
        self.hist_mask = np.zeros((G, H), dtype=bool)
        for i, x in enumerate(h):
            self.hist[i, : len(x)] = x
            self.hist_mask[i, : len(x)] = True


class NeuralRanker(nn.Module):
    def __init__(self, n_num: int, cat_vocabs: list[int], cat_dim: int = 4,
                 hidden=(512, 256), dropout: float = 0.1):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(v, cat_dim) for v in cat_vocabs])
        in_dim = 3 * D + n_num + len(cat_vocabs) * cat_dim + D
        layers, prev = [], in_dim
        for hdim in hidden:
            layers += [nn.Linear(prev, hdim), nn.ReLU(), nn.Dropout(dropout)]
            prev = hdim
        layers.append(nn.Linear(prev, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, Vc, U, num, cats, Vh, hmask):
        """Vc [B,C,d] U [B,d] num [B,C,n] cats [B,C,5] Vh [B,H,d] hmask [B,H] -> score [B,C]."""
        att = torch.einsum("bcd,bhd->bch", Vc, Vh) / D ** 0.5
        att = att.masked_fill(~hmask[:, None, :], float("-inf"))
        att = att.masked_fill(~hmask.any(1)[:, None, None], 0.0)        # guard hist rỗng
        hist_pooled = torch.einsum("bch,bhd->bcd", att.softmax(-1), Vh)
        Ue = U[:, None, :].expand_as(Vc)
        emb = torch.cat([e(cats[..., i]) for i, e in enumerate(self.embs)], dim=-1)
        x = torch.cat([Vc, Ue, Ue * Vc, num, emb, hist_pooled], dim=-1)
        return self.mlp(x).squeeze(-1)


def listwise_loss(score: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    gain = torch.pow(2.0, label) - 1.0                                   # grade 0 -> 0
    target = gain / gain.sum(dim=1, keepdim=True).clamp(min=1e-9)
    return -(target * F.log_softmax(score, dim=1)).sum(1).mean()


def _batch(ds: GroupData, idx: np.ndarray, V: torch.Tensor, device: str):
    g = lambda a: torch.from_numpy(a[idx]).to(device)
    Vc = V[g(ds.anime)]
    Vh = V[g(ds.hist)] * torch.from_numpy(ds.hist_mask[idx]).to(device)[..., None]
    return (Vc, g(ds.U), g(ds.num), g(ds.cats), Vh,
            torch.from_numpy(ds.hist_mask[idx]).to(device), g(ds.label))


@torch.no_grad()
def predict_groups(model, ds: GroupData, V: torch.Tensor, device: str,
                   batch: int = 256) -> np.ndarray:
    model.eval()
    out = []
    for s in range(0, ds.G, batch):
        idx = np.arange(s, min(s + batch, ds.G))
        Vc, U, num, cats, Vh, hm, _ = _batch(ds, idx, V, device)
        out.append(model(Vc, U, num, cats, Vh, hm).float().cpu().numpy())
    model.train()
    return np.concatenate(out).ravel()


def train(data_dir: Path, out_dir: Path, run_name: str = "nn_din",
          epochs: int = 2, batch_size: int = 32, lr: float = 1e-3,
          eval_every: int = 400, patience: int = 5, device: str | None = None,
          max_groups: int | None = None, item_vectors: Path | None = None) -> dict:
    device = device or auto_device()
    t0 = time.time()
    V_np = np.load(item_vectors or data_dir / "item_vectors.npy").astype(np.float32)
    V = torch.from_numpy(V_np).to(device)

    tr = GroupData(data_dir / "datasets" / "train.parquet",
                   data_dir / "datasets" / "train_users.parquet", k=200,
                   max_groups=max_groups)
    va = GroupData(data_dir / "pools" / "eval_val.parquet",
                   data_dir / "pools" / "eval_val_users.parquet", k=200,
                   mean=tr.mean, std=tr.std, cat_vocabs=tr.cat_vocabs,
                   max_groups=2_000 if max_groups else None)
    n_groups = tr.G
    print(f"train {n_groups:,} groups × {tr.C} | val {va.G:,} | device={device}")

    model = NeuralRanker(len(NUM_COLS), tr.cat_vocabs).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    amp = device == "cuda"
    scaler = torch.amp.GradScaler(enabled=amp)

    def val_metrics():
        pred = predict_groups(model, va, V, device)
        return sweep_best_alpha(va.cos, pred, va.labels_flat, va.off, va.r_total, KS, ALPHAS)

    rng = np.random.default_rng(42)
    best = {"ndcg@10": -1.0}
    best_alpha, best_state, bad, step = 0.5, None, 0, 0
    stop = False
    for ep in range(epochs):
        order = rng.permutation(n_groups)
        for s in range(0, n_groups, batch_size):
            idx = order[s : s + batch_size]
            Vc, U, num, cats, Vh, hm, label = _batch(tr, idx, V, device)
            with torch.autocast(device_type=device.split(":")[0], enabled=amp):
                loss = listwise_loss(model(Vc, U, num, cats, Vh, hm), label)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            step += 1
            if step % eval_every == 0:
                a, m, _ = val_metrics()
                flag = ""
                if m["ndcg@10"] > best["ndcg@10"]:
                    best, best_alpha, bad = m, a, 0
                    best_state = {k: v.detach().cpu().clone()
                                  for k, v in model.state_dict().items()}
                    flag = " *best"
                else:
                    bad += 1
                print(f"  ep{ep} step{step} loss={loss.item():.4f} "
                      f"ndcg@10={m['ndcg@10']:.4f} α={a}{flag} ({time.time() - t0:.0f}s)",
                      flush=True)
                if bad >= patience:
                    stop = True
                    break
        if stop:
            break
    if best_state is None:                                  # run quá ngắn chưa kịp eval
        _, best, _ = val_metrics()
        best_alpha, best_state = 0.5, model.state_dict()

    run_dir = out_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": best_state, "num_cols": NUM_COLS, "cat_cols": CAT_COLS,
        "cat_vocabs": tr.cat_vocabs, "mean": tr.mean, "std": tr.std,
    }, run_dir / "model.pt")
    row = {
        "run": run_name, "model_type": "neural", "arch": "din_mlp_512_256",
        "epochs": epochs, "lr": lr, "steps": step, "best_alpha": best_alpha,
        **{f"val_{m}": round(best[m], 5)
           for m in ("recall@10", "recall@100", "ndcg@10", "ndcg@100")},
        "train_sec": round(time.time() - t0),
    }
    (run_dir / "row.json").write_text(json.dumps(row, indent=2))
    print(f"[{run_name}] ndcg@10={best['ndcg@10']:.4f} α={best_alpha} ({row['train_sec']}s)")
    return row


def predict_pool(model_path: Path, df: pl.DataFrame, users: pl.DataFrame,
                 item_vectors: np.ndarray | None = None) -> np.ndarray:
    """Interface cho eval.py: score pool df (đã slice k) bằng checkpoint .pt."""
    import config
    ck = torch.load(model_path, map_location="cpu", weights_only=False)
    if item_vectors is None:
        item_vectors = np.load(config.ARTIFACTS / "item_vectors.npy")
    V = torch.from_numpy(item_vectors.astype(np.float32))

    qid = df["qid"].to_numpy()
    sizes = np.diff(np.r_[np.flatnonzero(np.r_[True, qid[1:] != qid[:-1]]), len(qid)])
    assert (sizes == sizes[0]).all()
    G, C = len(sizes), int(sizes[0])
    num = np.nan_to_num((df.select(ck["num_cols"]).to_numpy().astype(np.float32)
                         .reshape(G, C, -1) - ck["mean"]) / ck["std"])
    cats = np.minimum(df.select(ck["cat_cols"]).to_numpy().astype(np.int64).reshape(G, C, -1),
                      np.asarray(ck["cat_vocabs"]) - 1)
    anime = df["anime_idx"].to_numpy().astype(np.int64).reshape(G, C)
    users = users.sort("qid")
    U = np.stack([np.asarray(u, np.float32) for u in users["U"].to_list()])
    h = users["hist_top64"].to_list()
    H = max(max((len(x) for x in h), default=1), 1)
    hist = np.zeros((G, H), np.int64)
    hmask = np.zeros((G, H), bool)
    for i, x in enumerate(h):
        hist[i, : len(x)] = x
        hmask[i, : len(x)] = True

    model = NeuralRanker(len(ck["num_cols"]), ck["cat_vocabs"])
    model.load_state_dict(ck["state_dict"])
    model.eval()
    ds = type("DS", (), {"G": G, "C": C, "anime": anime, "num": num, "cats": cats,
                         "U": U, "hist": hist, "hist_mask": hmask,
                         "label": np.zeros((G, C), np.float32)})()
    return predict_groups(model, ds, V, "cpu")


def main() -> None:
    import argparse

    import config
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    kw = dict(max_groups=2_000, epochs=1, eval_every=20, patience=2) if args.smoke else {}
    train(config.DATA, config.MODELS, run_name="smoke_nn" if args.smoke else "nn_din",
          item_vectors=config.ARTIFACTS / "item_vectors.npy", **kw)


if __name__ == "__main__":
    main()
