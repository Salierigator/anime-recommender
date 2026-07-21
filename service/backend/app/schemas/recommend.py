"""Pydantic schemas — hợp đồng I/O của API. Shape khớp `service/API_CONTRACT.md`.

`AnimeItem` khớp `Recommender._row` (service/backend/recommend.py) + thêm `image_url` (TODO:
poster qua Jikan, slot cho frontend — backend hiện CHƯA điền).
(Dùng typing.Optional/List vì venv chạy Python 3.9 — pydantic không eval được cú pháp `X | None`.)
"""
from typing import List, Optional

from pydantic import BaseModel, Field


class AnimeItem(BaseModel):
    mal_id: int
    title: str
    type: str
    year: Optional[int] = None
    mal_score: Optional[float] = None
    popularity: Optional[int] = None   # rank độ phổ biến MAL (NHỎ = phổ biến) — sort client-side
    members: Optional[int] = None      # số user MAL có anime trong list — hiện trên card
    start_date: Optional[str] = None   # "YYYY-MM-DD" — sort theo ngày ra mắt client-side
    pred: Optional[float] = None       # điểm ranker (chỉ có ở list "main")
    cos: Optional[float] = None        # cosine retriever
    genres: List[str] = Field(default_factory=list)    # cho filter client-side
    themes: List[str] = Field(default_factory=list)
    studios: List[str] = Field(default_factory=list)
    image_url: Optional[str] = None    # TODO: poster (Jikan) — chưa điền


class RecommendRequest(BaseModel):
    username: Optional[str] = None
    mal_ids: Optional[List[int]] = None
    top_k: int = Field(default=20, ge=1, le=500)
    cold_k: int = Field(default=10, ge=0, le=500)
    live: bool = False                 # ép fetch MAL dù username có trong dataset
    anchor_mal_id: Optional[int] = None  # "tìm anime giống mal_id này" (giữ cá nhân hoá user)
    sfw: bool = True                   # loại hentai khỏi gợi ý (mặc định bật — an toàn demo)
    exclude_ids: Optional[List[int]] = Field(default=None, max_length=1000)
    # mal_id user đánh dấu "đã xem" trên UI → union vào seen-mask (loại khỏi recs,
    # KHÔNG phải tín hiệu thích — thích thì đưa vào mal_ids)
    # KHÔNG có cả username lẫn mal_ids → guest: history rỗng (UserTower h_empty)
    # → gợi ý generic thiên phổ biến, meta.source="guest"


class RecommendMeta(BaseModel):
    source: str                        # dataset | live | mal_ids | mock
    split: str = "-"                   # train/val/test (dataset) hoặc "-"
    history_count: int = 0             # số positive đưa vào model (map được vào corpus)
    total_entries: Optional[int] = None  # tổng entries trên list user (len animelist) — cho UI
    alpha: Optional[float] = None      # blend α của ranker
    k_retrieve: Optional[int] = None
    mode: str                          # mock | live
    map_xy: Optional[List[float]] = None  # [x,y] "you are here" trên GET /api/map (None = map tắt)


class RecommendResponse(BaseModel):
    main: List[AnimeItem]              # gợi ý chính (rerank LightGBM)
    cold: List[AnimeItem]              # anime mới (cold, theo cosine)
    meta: RecommendMeta
