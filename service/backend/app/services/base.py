"""Interface chung cho service gợi ý — Mock và Real cùng implement.

Thêm backend logic mới (vd recommend theo nhiều user, theo genre…) = thêm method ở đây
rồi implement ở cả 2 lớp, route chỉ gọi qua interface → dễ scale, không lệ thuộc model.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from app.schemas.recommend import RecommendRequest, RecommendResponse


class RecommenderService(ABC):
    model_loaded: bool = False         # health endpoint đọc cờ này

    @abstractmethod
    def recommend(self, req: RecommendRequest) -> RecommendResponse:
        ...

    @abstractmethod
    def map_payload(self) -> bytes:
        """JSON bytes cho GET /api/map (points/clusters/meta) — serialize sẵn, route trả thẳng."""
        ...

    @abstractmethod
    def territory_path(self) -> Optional[Path]:
        """Path PNG nền territory cho GET /api/map/territory.png (None = chưa có)."""
        ...
