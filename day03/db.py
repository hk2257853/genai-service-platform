import logging
import asyncpg
from pgvector.asyncpg import register_vector

import config

logger = logging.getLogger("db_utils")

# Global reference to the connection pool
_pool: asyncpg.Pool | None = None


async def register_pgvector_conn(conn: asyncpg.Connection):
    """Registers pgvector type mapping on a newly created connection in the pool."""
    await register_vector(conn)


async def get_db_pool() -> asyncpg.Pool:
    """Returns the database connection pool, initializing it if necessary."""
    global _pool
    if _pool is None:
        logger.info("Initializing asyncpg connection pool to database: %s", config.DB_NAME)
        _pool = await asyncpg.create_pool(
            host=config.DB_HOST,
            port=config.DB_PORT,
            database=config.DB_NAME,
            user=config.DB_USER,
            password=config.DB_PASSWORD,
            init=register_pgvector_conn,
            min_size=2,
            max_size=10
        )
    return _pool


async def close_db_pool():
    """Closes the connection pool."""
    global _pool
    if _pool is not None:
        logger.info("Closing database connection pool.")
        await _pool.close()
        _pool = None


async def init_db(vector_dimension: int):
    """Creates pgvector extension, table, and HNSW vector index in the database."""
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        logger.info("Setting up database extensions and tables (embedding dim: %d)...", vector_dimension)
        # 1. Enable pgvector extension
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")

        # 2. Create table for storing chunks (using format string for dimensions since it cannot be parameterized)
        create_table_ddl = f"""
        CREATE TABLE IF NOT EXISTS document_chunks (
            id SERIAL PRIMARY KEY,
            document_name VARCHAR(255) NOT NULL,
            text_content TEXT NOT NULL,
            embedding vector({vector_dimension}) NOT NULL
        );
        """
        await conn.execute(create_table_ddl)

        # 3. Create HNSW Index using cosine ops matching the <=> search operator
        create_index_ddl = """
        CREATE INDEX IF NOT EXISTS document_chunks_embedding_idx 
        ON document_chunks 
        USING hnsw (embedding vector_cosine_ops);
        """
        await conn.execute(create_index_ddl)
        logger.info("Database schema and HNSW vector index initialized successfully.")


async def insert_chunks(chunks: list[dict]):
    """Bulk inserts document chunks and their embeddings into pgvector."""
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        logger.info("Bulk inserting %d chunks into the database...", len(chunks))
        # Map chunks to list of tuples for executemany
        data = [
            (chunk["document_name"], chunk["text_content"], chunk["embedding"])
            for chunk in chunks
        ]
        await conn.executemany(
            """
            INSERT INTO document_chunks (document_name, text_content, embedding)
            VALUES ($1, $2, $3);
            """,
            data
        )


async def search_chunks(query_embedding: list[float], limit: int = 3) -> list[dict]:
    """Queries pgvector for the top k most similar chunks using cosine distance operator (<=>)."""
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        # <=> operator represents cosine distance
        rows = await conn.fetch(
            """
            SELECT document_name, text_content, (embedding <=> $1) as distance
            FROM document_chunks
            ORDER BY distance ASC
            LIMIT $2;
            """,
            query_embedding,
            limit
        )
        return [dict(row) for row in rows]
