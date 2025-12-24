# backend/app/ingest_worker.py

import asyncpg
from app.ingest_queue import INGEST_QUEUE
from app.db_pgvector import register_pgvector

DB_CONFIG = {
    "host": "127.0.0.1",
    "port": 5433,      # docker-exposed port
    "user": "admin",
    "password": "12345",
    "database": "live_cctv",
}

async def db_writer():
    pool = await asyncpg.create_pool(
        **DB_CONFIG,
        min_size=1,
        max_size=5,
    )

    async with pool.acquire() as conn:
        await register_pgvector(conn)

    print("🟢 Background DB writer started")

    batch = []

    while True:
        item = await INGEST_QUEUE.get()
        batch.append(item)

        # batch insert (tuneable)
        if len(batch) >= 20:
            async with pool.acquire() as conn:
                async with conn.transaction():
                    for d in batch:
                        await conn.execute(
                            """
                            INSERT INTO embeddings
                            (camera_id, track_id, ts, bbox_x, bbox_y, bbox_w, bbox_h, embedding)
                            VALUES ($1,$2,$3,$4,$5,$6,$7,$8)
                            """,
                            d["camera_id"],
                            d["track_id"],
                            d["timestamp"],
                            d["bbox"][0],
                            d["bbox"][1],
                            d["bbox"][2],
                            d["bbox"][3],
                            d["reid_embedding"],
                        )
            batch.clear()
