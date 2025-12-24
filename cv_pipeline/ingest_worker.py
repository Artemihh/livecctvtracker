import cv2
import time
import numpy as np
import requests
from deep_sort_realtime.deepsort_tracker import DeepSort
from yolo_detector import detect_persons

PROBE_EMB_PATH = "probe_embedding.npy"
MATCH_API = "http://127.0.0.1:8000/probe_match"
PROBE_THRESHOLD = 0.75

def l2(x):
    return x / (np.linalg.norm(x) + 1e-8)

tracker = DeepSort(
    embedder="torchreid",
    embedder_model_name="osnet_x0_25",
    embedder_wts="osnet_x0_25_msmt17.pt",
    embedder_gpu=True
)

cap = cv2.VideoCapture(0)
print("📡 Feed running", flush=True)

while True:
    ok, frame = cap.read()
    if not ok:
        continue

    boxes = detect_persons(frame)
    dets = [([x1, y1, x2-x1, y2-y1], 1.0, "person")
            for x1, y1, x2, y2 in boxes]

    tracks = tracker.update_tracks(dets, frame=frame)

    if not PROBE_EMB_PATH:
        continue

    try:
        probe = np.load(PROBE_EMB_PATH)
    except:
        continue

    for t in tracks:
        if not t.is_confirmed():
            continue

        emb = t.get_feature()
        if emb is None:
            continue

        s = float(np.dot(l2(emb), l2(probe)))
        if s >= PROBE_THRESHOLD:
            print(f"🎯 MATCH FOUND sim={s:.3f}", flush=True)
            requests.post(MATCH_API, json={
                "matched": True,
                "person_id": int(t.track_id),
                "similarity": round(s, 3)
            })
            time.sleep(2)
