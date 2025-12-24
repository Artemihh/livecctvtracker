# color_utils.py (fast HSV-based bag color heuristic)
import cv2
import numpy as np

def get_dominant_color_name(bgr_roi):
    """
    Fast, simple color guess for bag region.
    Returns: 'green', 'red', 'blue', 'black', 'white', 'other' or None
    """
    if bgr_roi is None or bgr_roi.size == 0:
        return None

    roi = cv2.resize(bgr_roi, (64, 64))
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    h_mean = float(hsv[:, :, 0].mean())
    s_mean = float(hsv[:, :, 1].mean())
    v_mean = float(hsv[:, :, 2].mean())

    # heuristics
    if v_mean < 40:
        return "black"
    if v_mean > 220 and s_mean < 35:
        return "white"
    if 35 <= h_mean <= 85:
        return "green"
    if (0 <= h_mean <= 10) or (160 <= h_mean <= 179):
        return "red"
    if 90 <= h_mean <= 130:
        return "blue"
    return "other"


def classify_bag_color(frame, det):
    """
    det: detection dict with x1,y1,x2,y2 in same frame coords
    """
    x1, y1, x2, y2 = map(int, (det["x1"], det["y1"], det["x2"], det["y2"]))
    h, w = frame.shape[:2]
    x1 = max(0, min(x1, w-1))
    x2 = max(0, min(x2, w-1))
    y1 = max(0, min(y1, h-1))
    y2 = max(0, min(y2, h-1))
    roi = frame[y1:y2, x1:x2]
    return get_dominant_color_name(roi)
