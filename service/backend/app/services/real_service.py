"""RealService — load Recommender 1 lần, serve recommend cho web app.

Username → LUÔN crawl live MAL (animelist + profile) → user_from_animelist.
mal_ids   → user_from_mal_ids (path test, không cần MAL API).

⚠ LAZY-IMPORT (service/CLAUDE.md §4):
  - Recommender import BÊN TRONG __init__ → mock mode không kéo torch/lightgbm.
  - mal_api import BÊN TRONG _fetch_live → module nạp MAL_CLIENT_ID lúc import (mal_api.py),
    chỉ cần khi thật sự crawl live; mal_ids vẫn chạy được khi thiếu client id.

Map (GET /api/map + meta.map_xy): chỉ load AnimeMap khi MAP_ENABLED=1 (mặc định TẮT — map/ đóng
băng). Tắt, hoặc thiếu/lệch map/outputs/service → cùng path degrade: /api/map 503, map_xy=null,
recommend VẪN chạy.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import HTTPException

from app.config import Settings
from app.schemas.recommend import (AnimeItem, RecommendMeta, RecommendRequest,
                                    RecommendResponse)
from app.services.base import RecommenderService


class RealService(RecommenderService):
    def __init__(self, settings: Settings):
        self.settings = settings
        from app.ml.recommender import Recommender          # lazy (torch/lightgbm)
        self.rec = Recommender()                            # load artifacts ~5s
        self.model_loaded = True
        self.map = None
        if settings.map_enabled:
            from app.ml.anime_map import AnimeMap, MapOutOfSync  # numpy-only, nhẹ
            try:
                self.map = AnimeMap()
            except (FileNotFoundError, MapOutOfSync) as e:
                print(f"⚠ map TẮT (recommend vẫn chạy): {e}")

    def recommend(self, req: RecommendRequest) -> RecommendResponse:
        if req.mal_ids:
            user = self.rec.user_from_mal_ids(req.mal_ids)
            total_entries = len(req.mal_ids)
        else:                                                # username: luôn crawl live
            animelist, profile = self._fetch_live(req.username)
            if not animelist:                                # rỗng / user không tồn tại / private / lỗi mạng
                raise HTTPException(
                    status_code=404,
                    detail=f"Can't get recommendations for '{req.username}' "
                           "(invalid username, private list, or connection error).",
                )
            user = self.rec.user_from_animelist(animelist, profile, req.username)
            total_entries = len(animelist)                   # tổng entries trên list (khớp profile)

        try:
            out = self.rec.recommend(user, top_k=req.top_k, cold_k=req.cold_k,
                                     anchor_mal_id=req.anchor_mal_id, sfw=req.sfw)
        except KeyError:
            raise HTTPException(
                status_code=422,
                detail=f"mal_id {req.anchor_mal_id} không có trong corpus.",
            )
        # "you are here" = centroid top-20 item HISTORY user chấm cao nhất (> mean score của
        # chính user) — vị trí là GU user, không phải output model (centroid recs bị ranker kéo
        # về vùng mainstream; locate(U) thì pumap fit trên item space, U lệch hệ).
        # Fallback: history không dùng được → centroid top-30 recs → locate(U).
        map_xy = None
        if self.map:
            map_xy = (self.map.locate_items(self._liked_mal_ids(user))
                      or self.map.locate_items([r["mal_id"] for r in out["main"][:30]])
                      or self.map.locate(self.rec.encode_U(user).numpy()))
        return RecommendResponse(
            main=[AnimeItem(**r) for r in out["main"]],
            cold=[AnimeItem(**r) for r in out["cold"]],
            meta=RecommendMeta(
                source=user["source"], split=user["split"],
                history_count=len(user["hist_idx"]), total_entries=total_entries,
                alpha=self.rec.alpha, k_retrieve=self.rec.k_retrieve, mode="live",
                map_xy=map_xy,
            ),
        )

    def _liked_mal_ids(self, user: dict, k: int = 20) -> list:
        """Top-k mal_id user thích nhất cho "you are here": score > mean score của user
        (mean tính trên item CÓ chấm điểm; score 0 = chưa chấm). Không có score nào
        (path mal_ids / user không chấm) hoặc mọi score bằng nhau → lấy k đầu của history."""
        hist, sc = user["hist_idx"], user["hist_score"]
        scored = sc[sc > 0]
        keep = np.flatnonzero(sc > scored.mean()) if len(scored) else np.array([], dtype=int)
        if len(keep) == 0:                                   # không chấm / chấm đều nhau
            keep = np.arange(len(hist))
        keep = keep[np.argsort(-sc[keep], kind="stable")][:k]
        return [int(self.rec.idx2mal[i]) for i in hist[keep]]

    def map_payload(self) -> bytes:
        if self.map is None:
            raise HTTPException(status_code=503,
                                detail="map export thiếu/lệch — xem log backend "
                                       "(chạy map/export_service.py)")
        return self.map.payload_bytes

    def territory_path(self) -> Optional[Path]:
        return self.map.territory_path if self.map else None

    def _fetch_live(self, username: str):
        try:
            from app.clients import mal_api                  # lazy: nạp MAL_CLIENT_ID lúc import
        except RuntimeError as e:                            # thiếu MAL_CLIENT_ID
            raise HTTPException(status_code=500, detail=str(e))
        animelist = mal_api.get_user_anime_list(username)
        profile = mal_api.get_user_profile(username)
        return animelist, profile
