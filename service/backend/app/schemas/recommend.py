"""Pydantic schemas — hợp đồng I/O của API. Shape khớp `service/API_CONTRACT.md`.

`AnimeItem` khớp `Recommender._row` (service/backend/recommend.py) + thêm `image_url` (TODO:
poster qua Jikan, slot cho frontend — backend hiện CHƯA điền).
(Dùng typing.Optional/List vì venv chạy Python 3.9 — pydantic không eval được cú pháp `X | None`.)
"""
from typing import List, Optional

from pydantic import BaseModel, Field, model_validator


class AnimeItem(BaseModel):
    mal_id: int
    title: str
    type: str
    year: Optional[int] = None
    mal_score: Optional[float] = None
    pred: Optional[float] = None       # điểm ranker (chỉ có ở list "main")
    cos: Optional[float] = None        # cosine retriever
    image_url: Optional[str] = None    # TODO: poster (Jikan) — chưa điền


class RecommendRequest(BaseModel):
    username: Optional[str] = None
    mal_ids: Optional[List[int]] = None
    top_k: int = Field(default=20, ge=1, le=100)
    cold_k: int = Field(default=10, ge=0, le=100)
    live: bool = False                 # ép fetch MAL dù username có trong dataset
    anchor_mal_id: Optional[int] = None  # "tìm anime giống mal_id này" (giữ cá nhân hoá user)

    @model_validator(mode="after")
    def _need_user_or_ids(self) -> "RecommendRequest":
        if not self.username and not self.mal_ids:
            raise ValueError("cần 'username' hoặc 'mal_ids'")
        return self


class RecommendMeta(BaseModel):
    source: str                        # dataset | live | mal_ids | mock
    split: str = "-"                   # train/val/test (dataset) hoặc "-"
    history_count: int = 0
    alpha: Optional[float] = None      # blend α của ranker
    k_retrieve: Optional[int] = None
    mode: str                          # mock | live


class RecommendResponse(BaseModel):
    main: List[AnimeItem]              # gợi ý chính (rerank LightGBM)
    cold: List[AnimeItem]              # anime mới (cold, theo cosine)
    meta: RecommendMeta
