#!/usr/bin/env python3
"""
main_cv.py — CV pipeline with YOLOv8 person detection, simple centroid tracker,
ReID embedding extraction (calls app.reid.reid_backend.get_embedding_from_bgr if present),
and ingestion to backend via ingest_sender.Sender.

Usage:
  cd live-cctv-tracker/cv_pipeline
  python main_cv.py --source 0 --backend http://127.0.0.1:8000 --camera-id CAM_01

Notes:
 - YOLOv8 (ultralytics) is optional. Install with: pip install ultralytics
 - If ultralytics is available, it will load yolov8n.pt automatically (downloads if needed).
 - If YOLOv8 is not available, it falls back to OpenCV HOG people detector.
 - Replace detection/tracking with your DeepSORT pipeline as needed.
"""

import argparse
import time
import sys
from collections import deque
from typing import List, Tuple, Dict, Optional

import cv2
import numpy as np

# Ensure project root is importable when running from cv_pipeline folder
# (helps when main is executed from different cwd)
import pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import the Sender helper from ingest_sender.py placed in cv_pipeline/
try:
    from ingest_sender import Sender
except Exception:
    # fallback to package style import if run from repo root
    try:
        from cv_pipeline.ingest_sender import Sender
    except Exception as e:
        raise ImportError("Could not import ingest_sender.Sender; ensure ingest_sender.py is in cv_pipeline/") from e

# Try to import your project's reid function if available
def import_reid_backend():
    try:
        from app.reid.reid_backend import get_embedding_from_bgr  # type: ignore
        return get_embedding_from_bgr
    except Exception:
        return None

get_embedding_from_bgr = import_reid_backend()

# ----------------------------
# YOLOv8 detector (ultralytics) integration
# ----------------------------
_yolo_model = None
try:
    from ultralytics import YOLO  # type: ignore
    # This will download yolov8n.pt if not present (internet required on first run)
    _yolo_model = YOLO("yolov8n.pt")
    print("[DETECT] YOLOv8 model loaded (yolov8n.pt)")
except Exception as e:
    _yolo_model = None
    print("[DETECT] YOLOv8 model NOT available:", e)

def detect_people_yolo(frame: np.ndarray, conf_thresh: float = 0.25, iou_thresh: float = 0.45) -> List[Tuple[int,int,int,int]]:
    """
    Detect people using YOLOv8 model. Returns list of boxes [x1,y1,x2,y2].
    """
    if _yolo_model is None:
        return []
    try:
        # Ultralytics model call — pass BGR frame
        results = _yolo_model(frame, imgsz=(640, 640), conf=conf_thresh, iou=iou_thresh, verbose=False)
        boxes: List[Tuple[int,int,int,int]] = []
        r = results[0]
        if getattr(r, "boxes", None) is None:
            return []
        # r.boxes.xyxy, r.boxes.conf, r.boxes.cls may be tensors
        xy = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        clsids = r.boxes.cls.cpu().numpy()
        for box, conf, cls in zip(xy, confs, clsids):
            # COCO person class index == 0
            if int(cls) != 0:
                continue
            x1, y1, x2, y2 = box.astype(int).tolist()
            # clip to frame
            h, w = frame.shape[:2]
            x1 = max(0, min(w-1, x1))
            y1 = max(0, min(h-1, y1))
            x2 = max(0, min(w-1, x2))
            y2 = max(0, min(h-1, y2))
            boxes.append((x1, y1, x2, y2))
        return boxes
    except Exception as e:
        print("[DETECT] YOLO inference/parse error:", e)
        return []

# ----------------------------
# Fallback: OpenCV HOG detector for people (keeps earlier behavior)
# ----------------------------
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

def detect_people_hog(frame: np.ndarray, downscale: float = 1.0) -> List[Tuple[int,int,int,int]]:
    """HOG-based person detection (fallback)."""
    if downscale != 1.0:
        small = cv2.resize(frame, (0,0), fx=downscale, fy=downscale)
    else:
        small = frame
    rects, weights = hog.detectMultiScale(small, winStride=(8,8), padding=(8,8), scale=1.05)
    boxes = []
    for (x,y,w,h) in rects:
        if downscale != 1.0:
            x1 = int(x / downscale)
            y1 = int(y / downscale)
            x2 = int((x+w) / downscale)
            y2 = int((y+h) / downscale)
        else:
            x1, y1, x2, y2 = x, y, x+w, y+h
        boxes.append((int(x1), int(y1), int(x2), int(y2)))
    return boxes

# ----------------------------
# Simple centroid tracker
# ----------------------------
class CentroidTracker:
    def __init__(self, max_distance: float = 80.0, max_lost_seconds: float = 2.0):
        self.next_id = 1
        self.tracks: Dict[int, Dict] = {}
        self.max_distance = float(max_distance)
        self.max_lost_seconds = float(max_lost_seconds)

    @staticmethod
    def _centroid_from_box(box):
        x1, y1, x2, y2 = box
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def update(self, detections: List[Tuple[int,int,int,int]]) -> List[Tuple[int, Tuple[int,int,int,int]]]:
        now = time.time()
        assignments: List[Tuple[int, Tuple[int,int,int,int]]] = []
        det_centroids = [self._centroid_from_box(b) for b in detections]
        det_used = [False] * len(detections)

        track_ids = list(self.tracks.keys())
        for tid in track_ids:
            rec = self.tracks[tid]
            last_cent = rec["centroid"]
            if last_cent is None:
                continue
            best_idx = None
            best_dist = None
            for i, c in enumerate(det_centroids):
                if det_used[i]:
                    continue
                dist = np.hypot(c[0] - last_cent[0], c[1] - last_cent[1])
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_idx = i
            if best_idx is not None and best_dist is not None and best_dist <= self.max_distance:
                bbox = detections[best_idx]
                centroid = det_centroids[best_idx]
                rec["centroid"] = centroid
                rec["last_ts"] = now
                rec["bbox"] = bbox
                rec["lost"] = 0
                rec["history"].append((bbox, now))
                assignments.append((tid, bbox))
                det_used[best_idx] = True

        for i, used in enumerate(det_used):
            if not used:
                bbox = detections[i]
                centroid = det_centroids[i]
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = {
                    "centroid": centroid,
                    "last_ts": now,
                    "bbox": bbox,
                    "lost": 0,
                    "history": deque([(bbox, now)], maxlen=16)
                }
                assignments.append((tid, bbox))

        to_delete = []
        for tid, rec in list(self.tracks.items()):
            age = now - rec.get("last_ts", now)
            if age > self.max_lost_seconds:
                to_delete.append(tid)
        for tid in to_delete:
            del self.tracks[tid]

        return assignments

    def get_track(self, track_id: int):
        return self.tracks.get(track_id)

# ----------------------------
# Utilities
# ----------------------------
def crop_box(frame: np.ndarray, box: Tuple[int,int,int,int], pad: int = 8) -> np.ndarray:
    x1,y1,x2,y2 = box
    h, w = frame.shape[:2]
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w-1, x2 + pad)
    y2 = min(h-1, y2 + pad)
    crop = frame[y1:y2, x1:x2]
    return crop

# ----------------------------
# Main loop
# ----------------------------
def main(args):
    # open capture
    try:
        src = int(args.source)
    except Exception:
        src = args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print("ERROR: could not open source:", args.source)
        return

    # instantiate sender and tracker
    sender = Sender(args.backend, timeout=5.0, max_retries=1)
    tracker = CentroidTracker(max_distance=args.max_distance, max_lost_seconds=args.max_lost_seconds)

    print("Starting CV loop. Press 'q' to quit.")
    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of stream or cannot read frame.")
            break
        frame_id += 1

        # optional scale for processing/display
        if args.scale != 1.0:
            frame = cv2.resize(frame, (0,0), fx=args.scale, fy=args.scale)

        # choose detector: YOLO if available, else HOG
        if _yolo_model is not None:
            boxes = detect_people_yolo(frame, conf_thresh=args.yolo_conf, iou_thresh=args.yolo_iou)
        else:
            boxes = detect_people_hog(frame, downscale=args.downscale)

        assignments = tracker.update(boxes)
        now_ts = time.time()

        for track_id, bbox in assignments:
            crop = crop_box(frame, bbox, pad=args.crop_pad)
            # compute embedding
            if get_embedding_from_bgr:
                try:
                    emb = get_embedding_from_bgr(crop)  # should return numpy array
                except Exception as e:
                    print("ReID backend error:", e)
                    emb = None
            else:
                # fallback pseudo-embedding for testing
                small = cv2.resize(crop, (64,128)) if crop.size != 0 else np.zeros((128,64,3), dtype=np.uint8)
                emb = np.mean(small.astype(np.float32), axis=(0,1)).ravel()
                if emb.size < 512:
                    emb = np.concatenate([emb.flatten(), np.zeros(512 - emb.size, dtype=np.float32)])
                emb = emb.astype(np.float32)

            # send to backend
            try:
                status, body = sender.send_detection(
                    camera_id=args.camera_id,
                    track_id=int(track_id),
                    cls="person",
                    bbox=[float(b) for b in bbox],
                    cx=(bbox[0]+bbox[2])/2.0,
                    cy=(bbox[1]+bbox[3])/2.0,
                    w=float(bbox[2]-bbox[0]),
                    h=float(bbox[3]-bbox[1]),
                    timestamp=now_ts,
                    reid_embedding_np=emb
                )
                if status != 200:
                    print("Backend returned non-200:", status, body)
            except Exception as e:
                print("Error sending detection:", e)

            # draw overlay
            x1,y1,x2,y2 = bbox
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(frame, f"ID:{track_id}", (x1, max(0,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        cv2.putText(frame, f"F:{frame_id} D:{len(boxes)}", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 1)
        cv2.imshow("main_cv", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

# ----------------------------
# CLI arguments
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", "-s", default="0", help="Camera index or video file path. Default=0")
    parser.add_argument("--backend", "-b", default="http://127.0.0.1:8000", help="Backend base URL")
    parser.add_argument("--camera-id", default="CAM_01", help="Camera identifier string to send to backend")
    parser.add_argument("--scale", type=float, default=1.0, help="Frame scale for processing/display")
    parser.add_argument("--downscale", type=float, default=1.0, help="Downscale factor for HOG detector (if used)")
    parser.add_argument("--crop-pad", type=int, default=8, help="pad pixels around crop for ReID")
    parser.add_argument("--max-distance", type=float, default=80.0, help="Centroid tracker max distance for matching")
    parser.add_argument("--max-lost-seconds", type=float, default=2.0, help="How long to keep tracks without updates (seconds)")
    # YOLO-specific args
    parser.add_argument("--yolo-conf", type=float, default=0.25, help="YOLO confidence threshold")
    parser.add_argument("--yolo-iou", type=float, default=0.45, help="YOLO NMS IoU threshold")
    args = parser.parse_args()

    main(args)
