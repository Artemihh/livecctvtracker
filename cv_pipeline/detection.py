# detection.py (optimized, returns boxes in original frame coords)
import cv2
from ultralytics import YOLO
import numpy as np

# Use YOLOv8n (small & fast)
model = YOLO("yolov8n.pt")

# only these COCO classes are relevant for our app
RELEVANT = {"person", "backpack", "handbag", "suitcase"}


def run_yolo_on_resized(frame_bgr, target_width=640, conf=0.35):
    """
    Resize frame for inference, run YOLO, return detections scaled to original frame.
    Returns list of dicts: {'cls','conf','x1','y1','x2','y2','w','h','cx','cy'}
    """
    orig_h, orig_w = frame_bgr.shape[:2]
    if orig_w == 0:
        return []
    scale = target_width / float(orig_w)
    new_h = max(1, int(orig_h * scale))
    small = cv2.resize(frame_bgr, (target_width, new_h))

    results = model(small, imgsz=target_width, conf=conf, verbose=False)[0]

    dets = []
    for box in results.boxes:
        cls_id = int(box.cls)
        cls_name = model.names[cls_id]
        confv = float(box.conf)

        if cls_name not in RELEVANT:
            continue

        x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())

        # scale coordinates back to original frame size
        x1_o = int(x1 / scale)
        y1_o = int(y1 / scale)
        x2_o = int(x2 / scale)
        y2_o = int(y2 / scale)

        w = x2_o - x1_o
        h = y2_o - y1_o
        cx = x1_o + w / 2
        cy = y1_o + h / 2

        dets.append({
            "cls": cls_name,
            "conf": confv,
            "x1": x1_o, "y1": y1_o,
            "x2": x2_o, "y2": y2_o,
            "w": w, "h": h,
            "cx": cx, "cy": cy
        })
    return dets
