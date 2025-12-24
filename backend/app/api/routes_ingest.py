from fastapi import APIRouter
from pydantic import BaseModel
from typing import List
from app.ingest_queue import INGEST_QUEUE

router = APIRouter(prefix="/api", tags=["ingest"])

class Detection(BaseModel):
    camera_id: str
    track_id: int
    cls: str
    bbox: List[float]
    cx: float
    cy: float
    w: float
    h: float
    timestamp: float
    reid_embedding: List[float]

class IngestRequest(BaseModel):
    detections: List[Detection]

@router.post("/ingest", status_code=202)
async def ingest(req: IngestRequest):
    for d in req.detections:
        await INGEST_QUEUE.put(d.model_dump())
    return {"status": "accepted"}
