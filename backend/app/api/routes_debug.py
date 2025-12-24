from fastapi import APIRouter, Request
router = APIRouter()

@router.get("/api/debug/store")
async def debug_store(request: Request):
    store = request.app.state.live_store
    store.prune()
    items = store.query_all()
    # return small summary
    return {"count": len(items), "sample": [{
        "camera_id": e["camera_id"],
        "track_id": e["track_id"],
        "timestamp": e["timestamp"],
        "bbox": e["bbox"]
    } for e in items[:10]]}
