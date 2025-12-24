import asyncio

INGEST_QUEUE = asyncio.Queue(maxsize=5000)
