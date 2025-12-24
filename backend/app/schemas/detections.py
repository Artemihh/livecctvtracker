# backend/app/schemas/detections.py
from pydantic import BaseModel
from typing import List, Optional, Any

class DetectionItem(BaseModel):
    camera_id: str
    track_id: int
    cls: Optional[str] = None
    bbox: List[int]        # [x1,y1,x2,y2]
    cx: Optional[float] = None
    cy: Optional[float] = None
    w: Optional[float] = None
    h: Optional[float] = None
    timestamp: float
    reid_embedding: Optional[List[float]] = None

class DetectionBatch(BaseModel):
    detections: List[DetectionItem]
