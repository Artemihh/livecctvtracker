# backend/app/store/live_store.py

import time
import numpy as np
from typing import Dict, List

class LiveStore:

    def __init__(self, retention_seconds=10.0, max_per_track=8):
        self.retention = retention_seconds
        self.max_per_track = max_per_track
        self.data: Dict[str, Dict[int, List[dict]]] = {}

    def add_detection(self, camera_id, track_id, timestamp, embedding, bbox):
        embedding = np.asarray(embedding, dtype=np.float32)

        # Init camera bucket
        cam = self.data.setdefault(camera_id, {})
        track = cam.setdefault(track_id, [])

        track.append({
            "ts": timestamp,
            "embedding": embedding,
            "bbox": bbox,
        })

        # Limit per track
        if len(track) > self.max_per_track:
            track.pop(0)

        # Cleanup by retention
        cutoff = time.time() - self.retention
        cam[track_id] = [x for x in track if x["ts"] >= cutoff]

        # Remove empty tracks
        if len(cam[track_id]) == 0:
            del cam[track_id]

# GLOBAL STORE INSTANCE
live_store = LiveStore()
