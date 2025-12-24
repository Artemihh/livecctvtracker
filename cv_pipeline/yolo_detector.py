# cv_pipeline/yolo_detector.py

import torch
from ultralytics import YOLO

# --------------------------------------------------
# Load YOLO model ONCE (important)
# --------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = YOLO("yolov8n.pt").to(DEVICE)

PERSON_CLASS_ID = 0  # COCO class for person

print(f"🟢 YOLO loaded on {DEVICE}")

# --------------------------------------------------
# Detect persons only
# --------------------------------------------------
def detect_persons(frame, conf_thres=0.4):
    """
    Returns list of person bounding boxes:
    [(x1, y1, x2, y2), ...]
    """
    results = MODEL.predict(
        source=frame,
        conf=conf_thres,
        classes=[PERSON_CLASS_ID],
        device=DEVICE,
        verbose=False
    )

    boxes = []
    for r in results:
        if r.boxes is None:
            continue
        for b in r.boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = map(int, b[:4])
            boxes.append((x1, y1, x2, y2))

    return boxes
