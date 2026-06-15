"""RealService — load Recommender 1 lần, serve recommend cho web app.

Username → LUÔN crawl live MAL (animelist + profile) → user_from_animelist.
mal_ids   → user_from_mal_ids (path test, không cần MAL API).

⚠ LAZY-IMPORT (service/CLAUDE.md §4):
  - Recommender import BÊN TRONG __init__ → mock mode không kéo torch/lightgbm.
  - mal_api import BÊN TRONG _fetch_live → module nạp MAL_CLIENT_ID lúc import (mal_api.py),
    chỉ cần khi thật sự crawl live; mal_ids vẫn chạy được khi thiếu client id.
"""
from __future__ import annotations

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

    def recommend(self, req: RecommendRequest) -> RecommendResponse:
        if req.mal_ids:
            user = self.rec.user_from_mal_ids(req.mal_ids)
        else:                                                # username: luôn crawl live
            animelist, profile = self._fetch_live(req.username)
            if not animelist:
                raise HTTPException(
                    status_code=404,
                    detail=f"Không lấy được animelist cho '{req.username}' "
                           "(user không tồn tại, list private, hoặc rỗng).",
                )
            user = self.rec.user_from_animelist(animelist, profile, req.username)

        try:
            out = self.rec.recommend(user, top_k=req.top_k, cold_k=req.cold_k,
                                     anchor_mal_id=req.anchor_mal_id)
        except KeyError:
            raise HTTPException(
                status_code=422,
                detail=f"mal_id {req.anchor_mal_id} không có trong corpus.",
            )
        return RecommendResponse(
            main=[AnimeItem(**r) for r in out["main"]],
            cold=[AnimeItem(**r) for r in out["cold"]],
            meta=RecommendMeta(
                source=user["source"], split=user["split"],
                history_count=len(user["hist_idx"]),
                alpha=self.rec.alpha, k_retrieve=self.rec.k_retrieve, mode="live",
            ),
        )

    def _fetch_live(self, username: str):
        try:
            from app.clients import mal_api                  # lazy: nạp MAL_CLIENT_ID lúc import
        except RuntimeError as e:                            # thiếu MAL_CLIENT_ID
            raise HTTPException(status_code=500, detail=str(e))
        animelist = mal_api.get_user_anime_list(username)
        profile = mal_api.get_user_profile(username)
        return animelist, profile
