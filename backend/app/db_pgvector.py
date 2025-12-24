import asyncpg

async def register_pgvector(conn: asyncpg.Connection):
    await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

    await conn.set_type_codec(
        "vector",
        encoder=lambda v: "[" + ",".join(map(str, v)) + "]",
        decoder=lambda v: list(map(float, v.strip("[]").split(","))),
        schema="public",
        format="text",
    )
    
