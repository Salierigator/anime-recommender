"""MockService — trả fixtures JSON ở repo root, KHÔNG load model.

Cho frontend dev độc lập (không cần torch/lightgbm/artifacts). Đọc `fixtures/recommend_sample.json`
+ `fixtures/map_sample.json` 1 lần lúc khởi tạo; cắt theo top_k/cold_k của request.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from fastapi import HTTPException

from app.config import ROOT, Settings
from app.schemas.anime import SearchResult
from app.schemas.recommend import (AnimeItem, RecommendMeta, RecommendRequest,
                                    RecommendResponse)
from app.services.base import RecommenderService


class MockService(RecommenderService):
    model_loaded = False

    def __init__(self, settings: Settings):
        path = settings.fixtures_dir / "recommend_sample.json"
        self._data = json.loads(path.read_text(encoding="utf-8"))
        map_path = settings.fixtures_dir / "map_sample.json"    # subset shape thật (xem fixtures/)
        self._map_bytes = map_path.read_bytes() if map_path.exists() else None

    def recommend(self, req: RecommendRequest) -> RecommendResponse:
        drop = set(req.exclude_ids or [])                    # mock mô phỏng seen-mask
        main = [AnimeItem(**r) for r in self._data["main"]
                if r["mal_id"] not in drop][: req.top_k]
        cold = [AnimeItem(**r) for r in self._data["cold"]
                if r["mal_id"] not in drop][: req.cold_k]
        fmeta = self._data.get("meta", {})
        meta = RecommendMeta(
            source="mock",
            split=fmeta.get("split", "-"),
            history_count=fmeta.get("history_count", 0),
            total_entries=fmeta.get("total_entries"),
            alpha=fmeta.get("alpha"),
            k_retrieve=fmeta.get("k_retrieve"),
            mode="mock",
            map_xy=fmeta.get("map_xy"),
        )
        return RecommendResponse(main=main, cold=cold, meta=meta)

    def search(self, q: str, limit: int) -> List[SearchResult]:
        # mock: substring-match trên item của fixture (main + cold) — frontend dev offline
        ql = q.lower()
        hits = [r for r in self._data["main"] + self._data["cold"]
                if ql in r["title"].lower()]
        return [SearchResult(mal_id=r["mal_id"], title=r["title"], type=r.get("type"),
                             year=r.get("year"), mal_score=r.get("mal_score"))
                for r in hits[:limit]]

    def map_payload(self) -> bytes:
        if self._map_bytes is None:
            raise HTTPException(status_code=503, detail="thiếu fixtures/map_sample.json")
        return self._map_bytes

    def territory_path(self) -> Optional[Path]:
        # mock không có asset riêng — dùng luôn map/outputs/service/ nếu repo này đã export (dev
        # frontend clone không có export -> 404, points sample vẫn đủ dựng UI)
        p = ROOT / "map" / "outputs" / "service" / "territory.png"
        return p if p.exists() else None
