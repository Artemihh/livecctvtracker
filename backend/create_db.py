import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

# These settings are aligned with backend/docker-compose.yml and are
# intentionally hard-coded here so this script always targets the
# pgvector Docker container (host: 127.0.0.1, port: 5433).
DB_HOST = "127.0.0.1"
DB_PORT = "5433"
DB_NAME = "live_cctv"
DB_USER = "admin"
DB_PASSWORD = "12345"

# pgvector extension (provided by ankane/pgvector image)
CREATE_EXTENSION_SQL = "CREATE EXTENSION IF NOT EXISTS vector;"

# Main embeddings table used by /api/ingest and /api/search
# Note: ReID ResNet50 backend outputs 2048-D embeddings.
CREATE_EMBEDDINGS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS embeddings (
    id SERIAL PRIMARY KEY,
    camera_id TEXT NOT NULL,
    track_id INTEGER NOT NULL,
    ts DOUBLE PRECISION NOT NULL,
    bbox_x REAL,
    bbox_y REAL,
    bbox_w REAL,
    bbox_h REAL,
    embedding VECTOR(2048)
);
"""

def main():
    print("Connecting to Postgres...")
    # Use explicit host/port here to avoid accidentally hitting a non-pgvector instance.
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
    )
    conn.autocommit = True
    cur = conn.cursor()

    # Ensure we operate in the public schema
    cur.execute("SET search_path TO public;")

    print("Enabling pgvector extension...")
    cur.execute(CREATE_EXTENSION_SQL)

    print("Creating embeddings table...")
    cur.execute(CREATE_EMBEDDINGS_TABLE_SQL)

    print("Database setup complete (pgvector + embeddings table)")

    cur.close()
    conn.close()

if __name__ == "__main__":
    main()
