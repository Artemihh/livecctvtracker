# cv_pipeline/ingest_sender.py
"""
ingest_sender.py

Helper to reliably POST detections (with ReID embeddings) from your CV pipeline
to the backend /api/ingest endpoint.

Usage:
    from ingest_sender import Sender
    sender = Sender("http://127.0.0.1:8000")
    sender.send_detection(camera_id=..., track_id=..., cls=..., bbox=..., cx=..., cy=..., w=..., h=..., timestamp=..., reid_embedding_np=embedding)

This module:
 - normalizes embeddings (L2)
 - converts numpy arrays to plain Python lists (JSON serializable)
 - retries on transient HTTP failures
 - prints debug info (you can replace prints with logger calls)
"""

import time
from typing import Sequence, Optional, Tuple
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


DEFAULT_TIMEOUT = 5.0


class Sender:
    """HTTP sender with a requests.Session and retry logic."""

    def __init__(self, backend_url: str, timeout: float = DEFAULT_TIMEOUT, max_retries: int = 2):
        self.backend_url = backend_url.rstrip("/")
        self.timeout = float(timeout)
        self.session = requests.Session()
        retries = Retry(
            total=max_retries,
            backoff_factor=0.5,
            status_forcelist=[502, 503, 504],
            allowed_methods=["POST", "GET"],
        )
        adapter = HTTPAdapter(max_retries=retries)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def send_detection(
        self,
        camera_id: str,
        track_id: int,
        cls: str,
        bbox: Sequence[float],  # [x1, y1, x2, y2]
        cx: float,
        cy: float,
        w: float,
        h: float,
        timestamp: float,
        reid_embedding_np: Optional[np.ndarray],
    ) -> Tuple[int, str]:
        """
        Build the JSON payload and POST to /api/ingest.

        Returns (status_code, response_text).
        Raises requests.RequestException on network errors.
        """
        # Defensive type conversions
        try:
            payload_detection = {
                "camera_id": str(camera_id),
                "track_id": int(track_id),
                "cls": str(cls),
                "bbox": [float(x) for x in bbox],
                "cx": float(cx),
                "cy": float(cy),
                "w": float(w),
                "h": float(h),
                "timestamp": float(timestamp),
            }
        except Exception as e:
            raise ValueError(f"Invalid detection fields: {e}") from e

        # Prepare embedding: normalize then convert to list
        if reid_embedding_np is not None:
            emb = np.asarray(reid_embedding_np, dtype=np.float32)
            norm = float(np.linalg.norm(emb))
            if norm > 0:
                emb = emb / norm
            payload_detection["reid_embedding"] = emb.tolist()

        payload = {"detections": [payload_detection]}

        # Debug prints (can be replaced with logging)
        try:
            print("[INGEST] POST payload keys:", list(payload.keys()))
            print("[INGEST] detection keys:", list(payload["detections"][0].keys()))
            if "reid_embedding" in payload_detection:
                print("[INGEST] embedding len:", len(payload_detection["reid_embedding"]))
                print("[INGEST] embedding first 8:", payload_detection["reid_embedding"][:8])
        except Exception:
            pass

        url = f"{self.backend_url}/api/ingest"
        r = self.session.post(url, json=payload, timeout=self.timeout)
        # print response for debugging; in production use logger
        print(f"[INGEST] status: {r.status_code}  body: {r.text}")
        return r.status_code, r.text


# convenience one-shot function (optional)
def send_detection(
    backend_url: str,
    camera_id: str,
    track_id: int,
    cls: str,
    bbox: Sequence[float],
    cx: float,
    cy: float,
    w: float,
    h: float,
    timestamp: float,
    reid_embedding_np: Optional[np.ndarray],
) -> Tuple[int, str]:
    s = Sender(backend_url)
    return s.send_detection(camera_id, track_id, cls, bbox, cx, cy, w, h, timestamp, reid_embedding_np)


# Quick test harness: run this file directly to POST a random embedding
if __name__ == "__main__":
    BACKEND = "http://127.0.0.1:8000"  # change if needed
    s = Sender(BACKEND)
    emb = np.random.rand(512).astype("float32")
    now = time.time()
    try:
        s.send_detection(
            camera_id="TEST_CAM",
            track_id=123,
            cls="person",
            bbox=[10, 20, 100, 200],
            cx=55.0,
            cy=110.0,
            w=90.0,
            h=180.0,
            timestamp=now,
            reid_embedding_np=emb,
        )
    except Exception as e:
        print("Local test failed:", e)
