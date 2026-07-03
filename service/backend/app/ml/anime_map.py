"""anime_map.py — bản đồ anime cho web: payload tĩnh + đặt user lên map ("you are here").

numpy/pandas-only, KHÔNG kéo torch/TF/lightgbm — độc lập với recommender.py (không dính bẫy
import-order). Đọc `map/outputs/service/` do `map/export_service.py` sinh (schema + sync rule:
`map/outputs/service/CONTRACT.md`). Encoder pumap = forward MLP từ pumap_encoder.npz (đã verify
≡ encoder.keras từng bit) → transform user vector U 128-d sang toạ độ map, không cần TF.

Fingerprint: sha256(item_vectors.npy) phải khớp lúc export — LỆCH = retriever re-export mà map
chưa re-fit → raise MapOutOfSync; RealService bắt để TẮT map (degrade), KHÔNG chặn recommend.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]                 # anime recommender/
MAP_DIR = ROOT / "map" / "outputs" / "service"


class MapOutOfSync(RuntimeError):
    """item_vectors.npy đã đổi sau lần export map cuối — phải chạy lại pipeline map."""


class AnimeMap:
    """Load map/outputs/service 1 lần: payload GET /api/map cache sẵn (bytes) + locate(U) -> [x,y]."""

    def __init__(self):
        self.meta = json.loads((MAP_DIR / "map_meta.json").read_text())
        sha = hashlib.sha256((ROOT / "artifacts" / "item_vectors.npy").read_bytes()).hexdigest()
        if sha != self.meta["item_vectors_sha256"]:
            raise MapOutOfSync(
                "map/outputs/service/ lệch item_vectors.npy — chạy lại pipeline map: build_base "
                "→ project (Colab) → cluster --k 28 → export_service (map/outputs/service/CONTRACT.md)")

        z = np.load(MAP_DIR / "pumap_encoder.npz")
        self._layers = [(z[f"W{i}"], z[f"b{i}"]) for i in range(4)]

        pts = pd.read_parquet(MAP_DIR / "map_points.parquet")
        clusters = pd.read_parquet(MAP_DIR / "map_clusters.parquet")
        for col in ("cx", "cy"):
            clusters[col] = np.round(clusters[col].astype(float), 4)
        # payload tĩnh serialize 1 LẦN -> bytes (route trả thẳng, khỏi re-validate 21k điểm/request)
        self.payload_bytes = json.dumps({
            "points": {                                    # columnar: gọn + vào thẳng typed array WebGL
                "mal_id": pts["mal_id"].tolist(),
                "title": pts["title"].tolist(),
                "x": np.round(pts["x"].astype(float), 4).tolist(),
                "y": np.round(pts["y"].astype(float), 4).tolist(),
                "label": pts["label"].tolist(),
                "popularity": pts["popularity"].tolist(),
                "is_cold": pts["is_cold"].tolist(),
            },
            "clusters": clusters.to_dict(orient="records"),
            "meta": {
                "k": self.meta["k"], "n_points": self.meta["n_points"],
                "extent": self.meta["extent"], "generated": self.meta["generated"],
                "territory_url": "/api/map/territory.png",
                "bg": self.meta["territory"]["bg"],
            },
        }, ensure_ascii=False, separators=(",", ":")).encode("utf-8")

    @property
    def territory_path(self) -> Path:
        return MAP_DIR / self.meta["territory"]["file"]

    def locate(self, U: np.ndarray) -> list:
        """User vector U [128] / [1,128] -> [x, y] trên map (forward encoder pumap)."""
        h = np.asarray(U, dtype=np.float32).reshape(1, -1)
        for w, b in self._layers[:-1]:
            h = np.maximum(h @ w + b, 0.0)
        w, b = self._layers[-1]
        xy = (h @ w + b)[0]
        return [round(float(xy[0]), 4), round(float(xy[1]), 4)]
