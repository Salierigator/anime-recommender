"""MockService — trả fixtures JSON ở repo root, KHÔNG load model.

Cho frontend dev độc lập (không cần torch/lightgbm/artifacts). Đọc `fixtures/recommend_sample.json`
1 lần lúc khởi tạo; cắt theo top_k/cold_k của request.
"""
from __future__ import annotations

import json

from app.config import Settings
from app.schemas.recommend import (AnimeItem, RecommendMeta, RecommendRequest,
                                    RecommendResponse)
from app.services.base import RecommenderService


class MockService(RecommenderService):
    model_loaded = False

    def __init__(self, settings: Settings):
        path = settings.fixtures_dir / "recommend_sample.json"
        self._data = json.loads(path.read_text(encoding="utf-8"))

    def recommend(self, req: RecommendRequest) -> RecommendResponse:
        main = [AnimeItem(**r) for r in self._data["main"]][: req.top_k]
        cold = [AnimeItem(**r) for r in self._data["cold"]][: req.cold_k]
        fmeta = self._data.get("meta", {})
        meta = RecommendMeta(
            source="mock",
            split=fmeta.get("split", "-"),
            history_count=fmeta.get("history_count", 0),
            total_entries=fmeta.get("total_entries"),
            alpha=fmeta.get("alpha"),
            k_retrieve=fmeta.get("k_retrieve"),
            mode="mock",
        )
        return RecommendResponse(main=main, cold=cold, meta=meta)
