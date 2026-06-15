"""Export artifacts từ checkpoint Two-Tower (best.pt) -> artifacts/ (firewall).

Chạy lại bất cứ lúc nào best.pt đổi. Không hardcode hyperparam/path: đọc cfg đã pickle
trong checkpoint + hằng số path chung trong retriever/src/config.py.

    python retriever/export.py [--ckpt retriever/checkpoints/best.pt]
                               [--out artifacts] [--device cpu]

Ghi (contract docs/PROJECT_STRUCTURE.md §4):
    item_vectors.npy     [num_items,128] float32 L2-norm, row==anime_idx (kể cả PAD/OOV).
                         Row item cold (H) encode id->OOV — id của H chưa từng được train,
                         encode id thật = noise; OOV khớp serve-path đã đo ở cold eval.
    item_index.parquet   anime_idx -> mal_id (-1 cho PAD/OOV) + is_cold
    user_tower.pt        user-side state_dict + pooling cfg + user_features spec
    user_split.parquet   username, user_idx, split
    eval_queries_{val,test,val_cold}.parquet + eval_seen.parquet + users_history.parquet
                         eval protocol + history cho ranker (test_cold chỉ khi --final-exam)
    CONTRACT.md          shape/dtype/version + metrics checkpoint
"""
from __future__ import annotations

import argparse
import dataclasses
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl
import torch

SRC = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC))

import config  # noqa: E402
from data import ItemTable, load_cold_mask, load_feature_spec  # noqa: E402
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


def _cfg_with_defaults(saved_cfg) -> config.TwoTowerConfig:
    """ckpt cũ pickle theo class TwoTowerConfig TRƯỚC khi thêm field (synopsis/optimizer/...).
    Rebuild điền default cho field thiếu (như _cfg_from_ckpt ở notebook) -> TwoTower đọc
    cfg.synopsis_*/optimizer không vỡ với best.pt cũ."""
    base = config.TwoTowerConfig()
    kw = {f.name: getattr(saved_cfg, f.name, getattr(base, f.name))
          for f in dataclasses.fields(config.TwoTowerConfig)}
    return config.TwoTowerConfig(**kw)


def build_model(ckpt: dict, device: str) -> tuple[TwoTower, dict]:
    """Dựng TwoTower từ cfg đã pickle, override path/device về local, load weights."""
    cfg = _cfg_with_defaults(ckpt["cfg"])
    cfg.train_data = config.TRAIN_DATA      # cfg pickle từ Colab mang path Colab -> ép local
    cfg.device = device
    spec = load_feature_spec(cfg.train_data)
    fixes = reconcile_spec_to_ckpt(spec, ckpt["model"])
    if fixes:
        print("WARNING: spec local lệch checkpoint (train-data đã regenerate sau khi train?):")
        for label, spec_v, ck_v in fixes:
            print(f"  - {label}: spec vocab={spec_v} -> dùng checkpoint vocab={ck_v}")
        print("  Item vectors KHÔNG ảnh hưởng; chỉ user-side. Cân nhắc retrain/regenerate train-data.")
    item_table = ItemTable(cfg.train_data, cfg.synopsis_emb_file,
                           cfg.synopsis_low_info_file, cfg.use_synopsis).to(device)
    model = TwoTower(spec, cfg, item_table).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, spec


def export_item_vectors(model: TwoTower, out: Path) -> np.ndarray:
    """Serve-path cache: item H (cold) encode id->OOV (content thật), item warm id thật."""
    cold_mask = load_cold_mask(config.TRAIN_DATA, model.item_table.num_items)
    model.refresh_item_cache(chunk=8192, cold_mask=cold_mask)
    vecs = model.item_cache.detach().cpu().numpy().astype(np.float32)
    np.save(out / "item_vectors.npy", vecs)
    return vecs


def export_item_index(num_items: int, out: Path) -> None:
    """anime_id_map chỉ có anime thật (idx 2..N+1); dựng dense mal_id dài num_items, PAD/OOV=-1."""
    amap = pl.read_parquet(config.TRAIN_DATA / "anime_id_map.parquet")
    mal_id = np.full(num_items, -1, dtype=np.int64)
    idx = amap["anime_idx"].to_numpy()
    mal_id[idx] = amap["mal_id"].to_numpy()
    is_cold = np.zeros(num_items, dtype=bool)
    is_cold[pl.read_parquet(config.TRAIN_DATA / "cold_items.parquet")["anime_idx"].to_numpy()] = True
    pl.DataFrame({
        "anime_idx": np.arange(num_items, dtype=np.int32),
        "mal_id": mal_id,
        "is_cold": is_cold,
    }).write_parquet(out / "item_index.parquet")


def export_user_tower(ckpt: dict, spec: dict, out: Path) -> None:
    """Lọc user-side state_dict (loại item_tower.*) + đóng gói đủ để service dựng encoder
    mà không cần ItemTable. Lấy cả param pooling (score_weight/attn_*) nếu config bật."""
    cfg = ckpt["cfg"]
    user_sd = {k: v for k, v in ckpt["model"].items() if not k.startswith("item_tower.")}
    if cfg.history_source == "embed":
        # embed: bảng pool history riêng — nằm trong user_sd (key hist_emb.*), service pool qua nó
        assert any(k.startswith("hist_emb.") for k in user_sd), "history_source=embed mà thiếu hist_emb"
    torch.save({
        "state_dict": user_sd,
        "d": cfg.d,
        "mlp_hidden": cfg.mlp_hidden,        # service cần để dựng lại UserTower(spec, d, hidden)
        "history_pool": cfg.history_pool,
        "score_pool": cfg.score_pool,
        "history_source": cfg.history_source,  # cache: pool qua item_vectors.npy | embed: qua hist_emb
        "eval_history_cap": cfg.eval_history_cap,
        "user_features": spec["user_features"],
        "special_idx": spec["special_idx"],
    }, out / "user_tower.pt")
    return user_sd


def export_eval_protocol(out: Path, final_exam: bool) -> dict[str, int]:
    """Eval protocol cho ranker (two-stage phải chấm ĐÚNG queries/seen/history của retriever):
    eval_queries_{split} + eval_seen (copy) + users_history (MỌI user — ranker không stream
    ratings.csv nữa). test_cold (final exam) CHỈ export khi --final-exam; default xoá file cũ
    nếu có để harness ranker không thể chấm nhầm."""
    counts = {}
    for split in ("val", "test", "val_cold") + (("test_cold",) if final_exam else ()):
        q = pl.read_parquet(config.TRAIN_DATA / "examples" / f"split={split}" / "part-0.parquet")
        q.select(pl.col("user_idx").cast(pl.Int32), pl.col("anime_idx").cast(pl.Int32)) \
            .write_parquet(out / f"eval_queries_{split}.parquet")
        counts[split] = q.height
    if not final_exam:
        (out / "eval_queries_test_cold.parquet").unlink(missing_ok=True)
    shutil.copyfile(config.TRAIN_DATA / "eval_seen.parquet", out / "eval_seen.parquet")
    (pl.scan_parquet(config.TRAIN_DATA / "users.parquet")
        .select("user_idx", "gender_id", "joined_bucket",
                "history_ids", "history_scores", "hard_neg_ids")
        .sink_parquet(out / "users_history.parquet"))
    return counts


def export_user_split(out: Path) -> pl.DataFrame:
    split = pl.read_parquet(config.TRAIN_DATA / "_user_split.parquet").select(
        "username", "user_idx", "split")
    split.write_parquet(out / "user_split.parquet")
    return split


def write_contract(out: Path, ckpt: dict, vecs: np.ndarray, user_sd: dict,
                   split: pl.DataFrame, q_counts: dict[str, int]) -> None:
    cfg = ckpt["cfg"]
    metrics = "\n".join(f"  - {k}: {v}" for k, v in ckpt.get("metrics", {}).items())
    user_keys = "\n".join(f"  - `{k}`: {tuple(v.shape)}" for k, v in user_sd.items())
    counts = dict(zip(*[c.to_list() for c in split.group_by("split").len().get_columns()]))
    text = f"""# artifacts/ — CONTRACT

Tự sinh bởi `retriever/export.py` — **không sửa tay**. Rerun export để cập nhật.

- Generated: {datetime.now(timezone.utc).isoformat(timespec="seconds")}
- Source checkpoint: epoch={ckpt.get("epoch")}, step={ckpt.get("step")}
- Embedding dim d = {cfg.d} | history_pool = `{cfg.history_pool}` | score_pool = `{cfg.score_pool}` | history_source = `{cfg.history_source}` | use_item_id = {cfg.use_item_id}

Checkpoint metrics (warm slice, val — mask = seen − query, history cap {cfg.eval_history_cap}):
{metrics or "  (n/a)"}

## Files

### `item_vectors.npy`
- shape `{vecs.shape}`, dtype `{vecs.dtype}`, L2-normalized (cosine = dot).
- **row index == anime_idx** (0=PAD, 1=OOV neutral; real anime 2..N-1).
- **Row item cold (`is_cold` trong item_index) encode id->OOV** — id của H chưa từng được
  train (cách ly khỏi mọi đường training), encode id thật = noise; OOV (content thật)
  khớp serve-path đã đo ở cold eval.
- Dùng kép: (a) brute-force cosine candidate, (b) bảng pool history cho UserTower
  (khi `history_source = cache`).
- Consumer lọc `anime_idx >= 2` khi tạo candidate.

### `item_index.parquet`
- cột `anime_idx: int32` (0..N-1, khớp row của item_vectors), `mal_id: int64` (-1 cho PAD/OOV),
  `is_cold: bool` (item thuộc tập cold H — row tương ứng trong item_vectors là content-only).

### `user_tower.pt`
- `state_dict` (chỉ user-side, không có `item_tower.*`):
{user_keys}
- `d`, `mlp_hidden`, `history_pool`, `score_pool`, `history_source`, `eval_history_cap`,
  `user_features` (vocab/dim), `special_idx`.
- Service: `import UserTower` + logic `pool_history`. History pool qua `item_vectors.npy`
  (`history_source=cache`) hoặc bảng `hist_emb.*` trong state_dict (`embed`).
  History đầu vào: list MAL của user sort score desc, cắt ở `eval_history_cap`.

### `user_split.parquet`
- cột `username`, `user_idx: int32`, `split` ∈ {{train,val,test}}.
- Counts: {counts}

### Eval protocol (cho ranker — two-stage chấm ĐÚNG protocol retriever)
- Protocol: mask = `seen(user) − query_đang_chấm` (KHÔNG mask thẳng seen — query ⊆ seen);
  metrics mean-per-user recall@K / ndcg@K (binary relevance, IDCG chuẩn hoá `min(R,K)`);
  history lúc encode U = prefix `eval_history_cap` của list full (đã sort score desc).
- `eval_queries_val.parquet` / `eval_queries_test.parquet` / `eval_queries_val_cold.parquet`:
  `user_idx: int32, anime_idx: int32` — positive held-out (query). Rows: {q_counts}
- `eval_seen.parquet`: `user_idx, seen_ids: list[int32]` — MỌI interaction mọi status (kể cả PTW)
  của eval user; nguồn duy nhất cho seen-mask.
- `users_history.parquet`: `user_idx, gender_id, joined_bucket, history_ids, history_scores,
  hard_neg_ids` — MỌI user. History FULL sort (score desc, tie asc); eval user = support
  (đã trừ query + cold H); hard_neg = dropped ∪ score 1-4 (≤64). Ranker dựng U + train data
  từ đây, KHÔNG stream ratings.csv.
- `eval_queries_test_cold.parquet`: **final exam** — CHỈ tồn tại khi chạy
  `export.py --final-exam` (chấm 1 lần khi chốt pipeline); default export xoá file này.

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
    ap.add_argument("--final-exam", action="store_true",
                    help="export thêm eval_queries_test_cold.parquet (chấm 1 lần cuối pipeline)")
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
    q_counts = export_eval_protocol(args.out, args.final_exam)
    write_contract(args.out, ckpt, vecs, user_sd, split, q_counts)

    print(f"Exported {num_items:,} item vectors {vecs.shape} -> {args.out}")
    files = ["item_vectors.npy", "item_index.parquet", "user_tower.pt", "user_split.parquet",
             "eval_seen.parquet", "users_history.parquet", "CONTRACT.md"]
    files += [f"eval_queries_{s}.parquet" for s in q_counts]
    for f in files:
        print(f"  - {args.out / f}")
    if args.final_exam:
        print("⚠ FINAL EXAM: eval_queries_test_cold.parquet đã export — chấm 1 lần rồi rerun "
              "export thường để xoá.")


if __name__ == "__main__":
    main()
