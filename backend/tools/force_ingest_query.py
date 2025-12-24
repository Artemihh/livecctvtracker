#!/usr/bin/env python3
"""
force_ingest_query.py

Now supports both:
    - test_person.jpg
    - test_person.jpeg

Whichever exists, it will load automatically.
"""

import time
import sys
from pathlib import Path

import cv2
import requests

# Add repo root to Python path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Try importing your real ReID model
try:
    from app.reid.reid_backend import get_embedding_from_bgr  # type: ignore
except Exception:
    get_embedding_from_bgr = None


def find_image_path(tools_dir: Path):
    """
    Automatically detect test_person.jpg or test_person.jpeg
    """
    jpg = tools_dir / "test_person.jpg"
    jpeg = tools_dir / "test_person.jpeg"

    if jpg.exists():
        return jpg
    if jpeg.exists():
        return jpeg

    raise SystemExit(
        f"ERROR: No test_person.jpg or test_person.jpeg found in {tools_dir}"
    )


def load_query_image(path: Path):
    img = cv2.imread(str(path))
    if img is None:
        raise SystemExit(f"ERROR: Could not read image file: {path}")
    print(f"[force_ingest] Loaded image: {path.name}")
    return img


def get_query_embedding(img_bgr):
    """
    Use actual ReID model if available,
    otherwise fallback to deterministic dummy embedding.
    """
    import numpy as np

    if get_embedding_from_bgr is not None:
        emb = get_embedding_from_bgr(img_bgr)
        if emb is None:
            raise RuntimeError("get_embedding_from_bgr returned None")
        return emb.astype("float32")

    # else: fallback dummy (for debugging ONLY)
    small = cv2.resize(img_bgr, (64, 128)).astype("float32")
    emb = small.mean(axis=(0, 1)).ravel()

    if emb.size < 512:
        emb = np.pad(emb, (0, 512 - emb.size), mode="constant")

    return emb.astype("float32")


def post_force_ingest(embedding, backend_url="http://127.0.0.1:8000"):
    payload = {
        "detections": [
            {
                "camera_id": "FORCE_CAM",
                "track_id": 9999,
                "cls": "person",
                "bbox": [0, 0, 10, 10],
                "cx": 5.0,
                "cy": 5.0,
                "w": 10.0,
                "h": 10.0,
                "timestamp": float(time.time()),
                "reid_embedding": embedding.tolist(),
            }
        ]
    }

    url = backend_url.rstrip("/") + "/api/ingest"
    print(f"[force_ingest] POSTing to {url}...")
    r = requests.post(url, json=payload, timeout=5)

    print("[force_ingest] status:", r.status_code)
    try:
        print("[force_ingest] body:", r.json())
    except:
        print("[force_ingest] response:", r.text)


def main():
    tools_dir = Path(__file__).resolve().parent
    image_path = find_image_path(tools_dir)

    img = load_query_image(image_path)
    emb = get_query_embedding(img)
    post_force_ingest(emb)


if __name__ == "__main__":
    main()
