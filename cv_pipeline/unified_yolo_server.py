# unified_yolo_server.py
"""
Unified YOLO Pipeline:
- Captures video (webcam / CCTV / RTSP)
- Runs YOLO detection
- Draws bounding boxes
- Extracts ReID embeddings
- Sends embeddings to backend /api/ingest
- Streams processed frames to frontend at /feed
"""

import sys
import os

# ---------------------------------------------------------
# FIX IMPORT PATH SO WE CAN IMPORT BACKEND MODULES
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_DIR = os.path.join(BASE_DIR, "backend")
sys.path.append(BACKEND_DIR)

# Now import your actual ReID module
from app.reid.reid_backend import get_embedding_from_bgr

import cv2
import time
import requests
import numpy as np
from ultralytics import YOLO
from fastapi import FastAPI
from starlette.responses import StreamingResponse


# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
VIDEO_SOURCE = 0  # webcam OR "rtsp://..." OR file path
CAMERA_ID = "CAM1"
BACKEND_INGEST = "http://127.0.0.1:8000/api/ingest"
MODEL_PATH = "yolov8n.pt"  # path to your YOLO model

# Load YOLO model
model = YOLO(MODEL_PATH)

# Video capture
cap = cv2.VideoCapture(VIDEO_SOURCE)

# FastAPI App
app = FastAPI()


# ---------------------------------------------------------
# MAIN STREAM + INGEST PIPELINE
# ---------------------------------------------------------
def generate_frames():
    track_id_counter = 0

    while True:
        success, frame = cap.read()
        if not success:
            continue

        # YOLO inference
        results = model(frame, conf=0.5)

        detections_payload = {"detections": []}

        # Process detections
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                w = x2 - x1
                h = y2 - y1
                cx = x1 + w / 2
                cy = y1 + h / 2

                # Crop for embedding
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                # ReID embedding
                emb = get_embedding_from_bgr(crop)
                emb = np.asarray(emb, dtype=np.float32)

                norm = np.linalg.norm(emb)
                if norm > 0:
                    emb = emb / norm

                track_id_counter += 1
                tid = track_id_counter

                # Add detection to payload
                detections_payload["detections"].append({
                    "camera_id": CAMERA_ID,
                    "track_id": tid,
                    "cls": "person",
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "cx": float(cx),
                    "cy": float(cy),
                    "w": float(w),
                    "h": float(h),
                    "timestamp": time.time(),
                    "reid_embedding": emb.tolist()
                })

                # Draw bbox & label on frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID {tid}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # SEND DETECTIONS TO BACKEND
        if detections_payload["detections"]:
            try:
                requests.post(BACKEND_INGEST, json=detections_payload, timeout=0.5)
            except Exception:
                pass  # don’t crash on network interruptions

        # STREAM FRAME TO FRONTEND
        ret, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" +
            frame_bytes +
            b"\r\n"
        )


# ---------------------------------------------------------
# STREAM ENDPOINT
# ---------------------------------------------------------
@app.get("/feed")
def video_feed():
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )
