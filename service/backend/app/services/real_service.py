"""RealService — STUB. Wiring model thật build ở lần sau.

⚠ LAZY-IMPORT bắt buộc: chỉ `from app.ml.recommender import Recommender` BÊN TRONG hàm (không
top-level module) để mock mode KHÔNG kéo torch/lightgbm vào process. recommender.py lo đúng thứ tự
import nội bộ (torch trước lightgbm; features/pool trước user_encode — xem service/CLAUDE.md §4).

Kế hoạch wiring (lần sau):
  __init__:  from app.ml.recommender import Recommender;  self.rec = Recommender()   # load ~5s
  recommend: chọn nguồn user (rec.user_from_dataset / user_from_animelist / user_from_mal_ids;
             live qua app.clients.mal_api) → rec.recommend(user, req.top_k, req.cold_k)
             → map dict {"main","cold"} sang AnimeItem/RecommendResponse (rec._row đã ra đúng field).
"""
from __future__ import annotations

from app.config import Settings
from app.schemas.recommend import RecommendRequest, RecommendResponse
from app.services.base import RecommenderService


class RealService(RecommenderService):
    model_loaded = False               # → True khi đã load Recommender (wiring sau)

    def __init__(self, settings: Settings):
        self.settings = settings
        # TODO(wiring): lazy-import + load Recommender 1 lần, set model_loaded = True.

    def recommend(self, req: RecommendRequest) -> RecommendResponse:
        raise NotImplementedError(
            "RealService chưa wiring — dùng MOCK_MODE=1 để dev. Xem TODO trong file."
        )
