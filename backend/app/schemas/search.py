# backend/app/schemas/search.py
from pydantic import BaseModel
from typing import List, Optional

class Match(BaseModel):
    camera_id: str
    track_id: int
    cls: Optional[str]
    bbox: List[int]
    timestamp: float
    similarity: float

class SearchResponse(BaseModel):
    query_ts: float
    matches: List[Match]
