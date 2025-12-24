import os
import asyncpg
import numpy as np
import cv2
import logging
from fastapi import APIRouter, UploadFile, File, HTTPException
from dotenv import load_dotenv

load_dotenv()

router = APIRouter()
logger = logging.getLogger("live-cctv")

# --------------------------------------------------
# ENV
# --------------------------------------------------
DB_HOST = os.getenv("DB_HOST")
DB_PORT = int(os.getenv("DB_PORT", 5433))
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

TOP_K = int(os.getenv("SEARCH_TOP_K", 5))
SIM_THRESHOLD = float(os.getenv("SEARCH_SIM_THRESHOLD", 0.6))


# --------------------------------------------------
# IMAGE UTILS
# --------------------------------------------------
def img_from_bytes(data: bytes):
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid image")
    return img


# --------------------------------------------------
# SEARCH BY IMAGE
# --------------------------------------------------
@router.post("/api/search/image")
async def search_by_image(file: UploadFile = File(...)):
    try:
        from app.reid.reid_backend import get_embedding_from_bgr

        # 1️⃣ Load image
        data = await file.read()
        img = img_from_bytes(data)

        # 2️⃣ Query embedding
        emb = get_embedding_from_bgr(img)
        if emb is None:
            raise HTTPException(400, "Embedding failed")

        emb = emb.astype(np.float32)
        emb /= np.linalg.norm(emb) + 1e-8

        # pgvector expects string format
        emb_str = "[" + ",".join(map(str, emb.tolist())) + "]"

        # 3️⃣ DB search (COSINE similarity)
        conn = await asyncpg.connect(
            host=DB_HOST,
            port=DB_PORT,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME,
        )

        rows = await conn.fetch(
            """
            SELECT
                camera_id,
                track_id,
                1 - (embedding <=> $1::vector) AS score
            FROM embeddings
            WHERE 1 - (embedding <=> $1::vector) >= $2
            ORDER BY score DESC
            LIMIT $3
            """,
            emb_str,
            SIM_THRESHOLD,
            TOP_K,
        )

        await conn.close()

        # 4️⃣ Format response
        results = [
            {
                "camera_id": r["camera_id"],
                "track_id": r["track_id"],
                "score": float(r["score"]),
            }
            for r in rows
        ]

        logger.info(
            f"🔍 Search results={len(results)} | threshold={SIM_THRESHOLD}"
        )

        return {
            "threshold": SIM_THRESHOLD,
            "count": len(results),
            "matches": results,
        }

    except Exception as e:
        logger.error(f"❌ SEARCH ERROR: {e}")
        raise HTTPException(500, "Search failed")
