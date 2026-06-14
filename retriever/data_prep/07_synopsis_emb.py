"""07 — Synopsis text-embedding (frozen artifact) cho ItemTower.

Gọi embedding model 1 LẦN (offline) trên toàn bộ synopsis -> fix cứng thành .npy, KHÔNG
train, TÁCH khỏi vòng re-export retriever. ItemTower chỉ train MLP chiếu raw->synopsis_dim
(xem retriever/src/model.py, gate cfg.use_synopsis).

Chạy SAU 01 (cần anime_id_map.parquet + feature_spec.json). Là inference model frozen,
KHÔNG phải train -> chạy local được. Đẩy 2 .npy lên Drive train-data/ để Colab dùng.

Tiền xử lý (SYNOPSIS_EMB.md): strip "(Source: ...)" ở đuôi (~32.9% dính); low_info =
NaN | <50 ký tự | placeholder "No synopsis…". Row low_info + PAD/OOV -> ItemTower thay
bằng vec no_synopsis học được (không dùng embedding rác).

Ghi (vào retriever/train-data/):
    synopsis_emb.npy        [num_items, dim] f32, L2-normalized, row==anime_idx (0,1=zeros)
    synopsis_low_info.npy   [num_items] bool (row 0,1 PAD/OOV = True)
    synopsis_meta.json      provenance (model, dim, n_low_info, preprocessing) cho báo cáo

Usage:
    python retriever/data_prep/07_synopsis_emb.py [--model all-MiniLM-L6-v2] [--device cpu]
"""
import argparse
import json
import pathlib
import re

import numpy as np
import pandas as pd
import polars as pl

ROOT = pathlib.Path(__file__).parent.parent.parent
SRC = ROOT / "cleaned-data" / "details.csv"
OUT = ROOT / "retriever" / "train-data"

SOURCE_RE = re.compile(r"\s*\(Source:[^)]*\)\s*$", re.IGNORECASE)  # strip "(Source: ...)" ở đuôi
MIN_CHARS = 50                                                     # <50 ký tự -> low_info
PLACEHOLDER_RE = re.compile(r"^\s*no synopsis", re.IGNORECASE)     # "No synopsis (has been added|yet)…"


def clean(text) -> str:
    """Strip (Source:...) đuôi + trim. NaN/không phải str -> ''."""
    if not isinstance(text, str):
        return ""
    return SOURCE_RE.sub("", text).strip()


def is_low_info(t: str) -> bool:
    return len(t) < MIN_CHARS or bool(PLACEHOLDER_RE.match(t))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="all-MiniLM-L6-v2", help="sentence-transformers model")
    ap.add_argument("--device", default=None, help="cpu|cuda|mps (None = auto)")
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args()

    spec = json.loads((OUT / "feature_spec.json").read_text())
    num_items = spec["num_items"]                                  # = N real + 2 (PAD,OOV)

    # join details(synopsis) với anime_id_map(mal_id->anime_idx) -> scatter theo anime_idx (robust,
    # không giả định thứ tự). amap chỉ có anime thật (idx 2..N+1); PAD/OOV để zeros + low_info.
    amap = pl.read_parquet(OUT / "anime_id_map.parquet").to_pandas()        # mal_id, anime_idx
    df = pd.read_csv(SRC, usecols=["mal_id", "synopsis"])
    merged = amap.merge(df, on="mal_id", how="left").sort_values("anime_idx").reset_index(drop=True)
    assert len(merged) == num_items - 2, f"amap {len(merged)} != num_items-2 {num_items - 2}"

    texts = [clean(s) for s in merged["synopsis"]]
    low = np.array([is_low_info(t) for t in texts], dtype=bool)
    print(f"synopsis: {len(texts):,} anime · low_info {int(low.sum()):,} "
          f"({low.mean() * 100:.1f}%) · source-tag stripped")

    from sentence_transformers import SentenceTransformer                   # lazy: chỉ cần lúc chạy
    model = SentenceTransformer(args.model, device=args.device)
    emb = model.encode(texts, batch_size=args.batch_size, show_progress_bar=True,
                       normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)  # L2 sẵn
    dim = emb.shape[1]

    # dense [num_items, dim] theo anime_idx; row 0,1 (PAD,OOV) = zeros + low_info=True.
    emb_full = np.zeros((num_items, dim), dtype=np.float32)
    low_full = np.zeros(num_items, dtype=bool)
    idx = merged["anime_idx"].to_numpy()
    emb_full[idx] = emb
    low_full[idx] = low
    low_full[0] = low_full[1] = True                              # PAD/OOV -> no_synopsis nhánh

    np.save(OUT / "synopsis_emb.npy", emb_full)
    np.save(OUT / "synopsis_low_info.npy", low_full)
    (OUT / "synopsis_meta.json").write_text(json.dumps({
        "model": args.model, "dim": int(dim), "num_items": int(num_items),
        "n_low_info": int(low_full.sum()), "min_chars": MIN_CHARS,
        "normalize": "l2", "strip_source_tag": True,
    }, indent=2))
    print(f"synopsis_emb.npy [{num_items},{dim}] + synopsis_low_info.npy ghi vào {OUT}")
    print("DONE 07.  -> đẩy 2 file .npy lên Drive recommender_train_colab/train-data/")


if __name__ == "__main__":
    main()
