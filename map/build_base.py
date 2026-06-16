"""build_base.py — dựng bảng base 1 lần cho map/ (CLI).

Join artifacts/item_vectors + item_index + cleaned-data/details.csv -> outputs/base.parquet
(metadata mỗi anime real) + outputs/vectors_real.npy (vector 128-d căn hàng theo row base).
Mọi script sau (project/cluster/viz) đọc 2 file này.

    venv/bin/python map/build_base.py
"""
from __future__ import annotations

import _common as C


def main() -> None:
    base, vectors = C.build_base_table()
    C.OUTPUTS.mkdir(parents=True, exist_ok=True)
    base.to_parquet(C.OUTPUTS / "base.parquet")
    import numpy as np
    np.save(C.OUTPUTS / "vectors_real.npy", vectors)

    n_cold = int(base["is_cold"].sum())
    n_genre = int((base["primary_genre"] != "Unknown").sum())
    print(f"base.parquet: {len(base):,} anime real ({n_cold:,} cold, {len(base) - n_cold:,} warm)")
    print(f"vectors_real.npy: {vectors.shape}")
    print(f"genre-join phủ: {n_genre:,}/{len(base):,} có primary_genre "
          f"({n_genre / len(base):.1%})")
    print("top primary_genre:")
    print(base["primary_genre"].value_counts().head(10).to_string())


if __name__ == "__main__":
    main()
