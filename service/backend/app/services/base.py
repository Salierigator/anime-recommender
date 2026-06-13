"""Interface chung cho service gợi ý — Mock và Real cùng implement.

Thêm backend logic mới (vd recommend theo nhiều user, theo genre…) = thêm method ở đây
rồi implement ở cả 2 lớp, route chỉ gọi qua interface → dễ scale, không lệ thuộc model.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from app.schemas.recommend import RecommendRequest, RecommendResponse


class RecommenderService(ABC):
    model_loaded: bool = False         # health endpoint đọc cờ này

    @abstractmethod
    def recommend(self, req: RecommendRequest) -> RecommendResponse:
        ...
