"""Export artifacts từ checkpoint Two-Tower (best.pt) -> artifacts/ (firewall).

Chạy lại bất cứ lúc nào best.pt đổi. Không hardcode hyperparam/path: đọc cfg đã pickle
trong checkpoint + hằng số path chung trong retriever/src/config.py.

    python retriever/export.py [--ckpt retriever/checkpoints/best.pt]
                               [--out artifacts] [--device cpu]

Ghi (contract docs/PROJECT_STRUCTURE.md §4):
    item_vectors.npy     [num_items,128] float32 L2-norm, row==anime_idx (kể cả PAD/OOV)
    item_index.parquet   anime_idx -> mal_id (-1 cho PAD/OOV)
    user_tower.pt        user-side state_dict + pooling cfg + user_features spec
    user_split.parquet   username, user_idx, split
    CONTRACT.md          shape/dtype/version + metrics checkpoint
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl
import torch

SRC = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC))

import config  # noqa: E402
from data import ItemTable, load_feature_spec  # noqa: E402
from model import ItemTower, TwoTower  # noqa: E402


def reconcile_spec_to_ckpt(spec: dict, sd: dict) -> list[tuple[str, int, int]]:
    """Spec local có thể đã regenerate khác version đã train best.pt (vd joined vocab 6->5).
    Source of truth cho KIẾN TRÚC là checkpoint -> ép vocab trong spec khớp shape embedding
    đã train để load_state_dict không vỡ. Trả list mismatch để cảnh báo (không nuốt im)."""
    targets = [(spec["item_features"][n], f"item_tower.cat_emb.cat_{n}.weight", f"item.{n}")
               for n, _ in ItemTower.CAT_FEATS]
    targets += [
        (spec["item_features"]["studios"], "item_tower.studio_emb.weight", "item.studios"),
        (spec["user_features"]["gender"], "user_tower.gender_emb.weight", "user.gender"),
        (spec["user_features"]["joined"], "user_tower.joined_emb.weight", "user.joined"),
    ]
    fixes = []
    for entry, key, label in targets:
        ck_vocab = sd[key].shape[0]
        if entry["vocab"] != ck_vocab:
            fixes.append((label, entry["vocab"], ck_vocab))
            entry["vocab"] = ck_vocab
    return fixes


def build_model(ckpt: dict, device: str) -> tuple[TwoTower, dict]:
    """Dựng TwoTower từ cfg đã pickle, override path/device về local, load weights."""
    cfg = ckpt["cfg"]
    cfg.train_data = config.TRAIN_DATA      # cfg pickle từ Colab mang path Colab -> ép local
    cfg.device = device
    spec = load_feature_spec(cfg.train_data)
    fixes = reconcile_spec_to_ckpt(spec, ckpt["model"])
    if fixes:
        print("WARNING: spec local lệch checkpoint (train-data đã regenerate sau khi train?):")
        for label, spec_v, ck_v in fixes:
            print(f"  - {label}: spec vocab={spec_v} -> dùng checkpoint vocab={ck_v}")
        print("  Item vectors KHÔNG ảnh hưởng; chỉ user-side. Cân nhắc retrain/regenerate train-data.")
    item_table = ItemTable(cfg.train_data).to(device)
    model = TwoTower(spec, cfg, item_table).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, spec


def export_item_vectors(model: TwoTower, out: Path) -> np.ndarray:
    model.refresh_item_cache(chunk=8192)
    vecs = model.item_cache.detach().cpu().numpy().astype(np.float32)
    np.save(out / "item_vectors.npy", vecs)
    return vecs


def export_item_index(num_items: int, out: Path) -> None:
    """anime_id_map chỉ có anime thật (idx 2..N+1); dựng dense mal_id dài num_items, PAD/OOV=-1."""
    amap = pl.read_parquet(config.TRAIN_DATA / "anime_id_map.parquet")
    mal_id = np.full(num_items, -1, dtype=np.int64)
    idx = amap["anime_idx"].to_numpy()
    mal_id[idx] = amap["mal_id"].to_numpy()
    pl.DataFrame({
        "anime_idx": np.arange(num_items, dtype=np.int32),
        "mal_id": mal_id,
    }).write_parquet(out / "item_index.parquet")


def export_user_tower(ckpt: dict, spec: dict, out: Path) -> None:
    """Lọc user-side state_dict (loại item_tower.*) + đóng gói đủ để service dựng encoder
    mà không cần ItemTable. Lấy cả param pooling (score_weight/attn_*) nếu config bật."""
    cfg = ckpt["cfg"]
    user_sd = {k: v for k, v in ckpt["model"].items() if not k.startswith("item_tower.")}
    torch.save({
        "state_dict": user_sd,
        "d": cfg.d,
        "mlp_hidden": cfg.mlp_hidden,        # service cần để dựng lại UserTower(spec, d, hidden)
        "history_pool": cfg.history_pool,
        "score_pool": cfg.score_pool,
        "user_features": spec["user_features"],
        "special_idx": spec["special_idx"],
    }, out / "user_tower.pt")
    return user_sd


def export_user_split(out: Path) -> pl.DataFrame:
    split = pl.read_parquet(config.TRAIN_DATA / "_user_split.parquet").select(
        "username", "user_idx", "split")
    split.write_parquet(out / "user_split.parquet")
    return split


def write_contract(out: Path, ckpt: dict, vecs: np.ndarray, user_sd: dict,
                   split: pl.DataFrame) -> None:
    cfg = ckpt["cfg"]
    metrics = "\n".join(f"  - {k}: {v}" for k, v in ckpt.get("metrics", {}).items())
    user_keys = "\n".join(f"  - `{k}`: {tuple(v.shape)}" for k, v in user_sd.items())
    counts = dict(zip(*[c.to_list() for c in split.group_by("split").len().get_columns()]))
    text = f"""# artifacts/ — CONTRACT

Tự sinh bởi `retriever/export.py` — **không sửa tay**. Rerun export để cập nhật.

- Generated: {datetime.now(timezone.utc).isoformat(timespec="seconds")}
- Source checkpoint: epoch={ckpt.get("epoch")}, step={ckpt.get("step")}
- Embedding dim d = {cfg.d} | history_pool = `{cfg.history_pool}` | score_pool = `{cfg.score_pool}` | use_item_id = {cfg.use_item_id}

Checkpoint metrics (cold-by-user, val):
{metrics or "  (n/a)"}

## Files

### `item_vectors.npy`
- shape `{vecs.shape}`, dtype `{vecs.dtype}`, L2-normalized (cosine = dot).
- **row index == anime_idx** (0=PAD, 1=OOV neutral; real anime 2..N-1).
- Dùng kép: (a) brute-force cosine candidate, (b) bảng pool history cho UserTower.
- Consumer lọc `anime_idx >= 2` khi tạo candidate.

### `item_index.parquet`
- cột `anime_idx: int32` (0..N-1, khớp row của item_vectors), `mal_id: int64` (-1 cho PAD/OOV).

### `user_tower.pt`
- `state_dict` (chỉ user-side, không có `item_tower.*`):
{user_keys}
- `d`, `mlp_hidden`, `history_pool`, `score_pool`, `user_features` (vocab/dim), `special_idx`.
- Service: `import UserTower` + logic `pool_history`, pool history qua `item_vectors.npy`.

### `user_split.parquet`
- cột `username`, `user_idx: int32`, `split` ∈ {{train,val,test}}.
- Counts: {counts}

## Firewall
retriever GHI; ranker+service chỉ ĐỌC file ở đây + import *định nghĩa* model — không import code train.
"""
    (out / "CONTRACT.md").write_text(text)


def main() -> None:
    root = config.ROOT.parent
    ap = argparse.ArgumentParser(description="Export Two-Tower artifacts -> artifacts/")
    ap.add_argument("--ckpt", type=Path, default=config.ROOT / "checkpoints" / "best.pt")
    ap.add_argument("--out", type=Path, default=root / "artifacts")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"Loading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)

    model, spec = build_model(ckpt, args.device)
    num_items = model.item_table.num_items

    vecs = export_item_vectors(model, args.out)
    export_item_index(num_items, args.out)
    user_sd = export_user_tower(ckpt, spec, args.out)
    split = export_user_split(args.out)
    write_contract(args.out, ckpt, vecs, user_sd, split)

    print(f"Exported {num_items:,} item vectors {vecs.shape} -> {args.out}")
    for f in ("item_vectors.npy", "item_index.parquet", "user_tower.pt",
              "user_split.parquet", "CONTRACT.md"):
        print(f"  - {args.out / f}")


if __name__ == "__main__":
    main()
