import os
import asyncpg
from dotenv import load_dotenv

# Load .env explicitly
load_dotenv()

DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "live_cctv")
DB_USER = os.getenv("DB_USER", "admin")
DB_PASSWORD = os.getenv("DB_PASSWORD", "12345")


async def get_db():
    """
    Return an asyncpg connection to the pgvector-enabled Postgres instance.

    This MUST match:
      - docker-compose.yml
      - .env
      - psql connection used for debugging

    Host: 127.0.0.1
    Port: 5432
    DB:   live_cctv
    User: admin
    """
    return await asyncpg.connect(
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        host=DB_HOST,
        port=DB_PORT,
    )
