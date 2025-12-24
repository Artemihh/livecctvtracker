from fastapi import APIRouter, UploadFile, File, BackgroundTasks, HTTPException
import numpy as np
import cv2
import time
import requests

router = APIRouter(prefix="/api", tags=["augment"])

BACKEND_BASE = "http://127.0.0.1:8000"

# ----------------------------
# Utils
# ----------------------------
def img_from_bytes(data: bytes):
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")
    return img

def variants(img):
    h, w = img.shape[:2]
    return [
        img,
        cv2.flip(img, 1),
        cv2.GaussianBlur(img, (5, 5), 0),
        cv2.resize(img, (w, h)),
    ]

# ----------------------------
# Background Worker
# ----------------------------
def process_and_ingest(img):
    from app.reid.reid_backend import get_embedding_from_bgr

    print("\n⚡ augment worker started")

    emb_list = []
    for i, v in enumerate(variants(img)):
        try:
            emb = get_embedding_from_bgr(v)
            if emb is None:
                continue
            emb = emb.astype(np.float32)
            emb /= (np.linalg.norm(emb) + 1e-8)
            emb_list.append(emb.tolist())
            print(f"   ✔ Variant {i} embedded")
        except Exception as e:
            print("❌ Embedding error:", e)

    if not emb_list:
        print("❌ No embeddings produced")
        return

    now = time.time()
    detections = []

    for i, e in enumerate(emb_list):
        detections.append({
            "camera_id": "QUERY_CAM",
            "track_id": 9998,
            "cls": "person",
            "bbox": [0, 0, 10, 10],
            "cx": 5,
            "cy": 5,
            "w": 10,
            "h": 10,
            "timestamp": now + i * 0.01,
            "reid_embedding": e
        })

    try:
        r = requests.post(
            BACKEND_BASE + "/api/ingest",
            json={"detections": detections},
            timeout=2
        )
        print("🟢 augment → ingest:", r.status_code)
    except Exception as e:
        print("❌ Failed posting to ingest:", e)

# ----------------------------
# API Endpoint
# ----------------------------
@router.post("/augment_ingest")
async def augment_ingest(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "Only image files allowed")

    data = await file.read()
    img = img_from_bytes(data)

    # Safety limit
    if img.shape[0] * img.shape[1] > 1920 * 1080:
        raise HTTPException(400, "Image too large")

    print("📸 Uploaded:", img.shape)

    # 🔥 Non‑blocking execution
    background_tasks.add_task(process_and_ingest, img)

    return {
        "status": "accepted",
        "message": "Augmentation queued"
    }
