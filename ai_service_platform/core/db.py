import logging
import asyncpg
from pgvector.asyncpg import register_vector

from ai_service_platform import config

logger = logging.getLogger("db_utils")

# Global reference to the connection pool
_pool: asyncpg.Pool | None = None

# Flag to trigger fallback when pgvector extension is not installed
USE_FALLBACK = False


async def register_pgvector_conn(conn: asyncpg.Connection):
    """Registers pgvector type mapping on a newly created connection in the pool, handling missing pgvector gracefully."""
    try:
        await register_vector(conn)
    except Exception as e:
        logger.debug("Could not register pgvector mapping (normal in fallback mode): %s", e)


async def get_db_pool() -> asyncpg.Pool:
    """Returns the database connection pool, initializing it if necessary."""
    global _pool, USE_FALLBACK
    if _pool is None:
        # Create extension first using a separate connection to detect pgvector support
        conn = await asyncpg.connect(
            host=config.DB_HOST,
            port=config.DB_PORT,
            database=config.DB_NAME,
            user=config.DB_USER,
            password=config.DB_PASSWORD,
        )
        try:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            USE_FALLBACK = False
            logger.info("Successfully enabled pgvector extension.")
        except Exception as e:
            USE_FALLBACK = True
            logger.warning(
                "PostgreSQL does not support the 'vector' extension. Fallback mode enabled (using REAL[] arrays). Error: %s", 
                e
            )
        finally:
            await conn.close()

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
    """Creates pgvector extension (or falls back), table, and HNSW vector index in the database."""
    global USE_FALLBACK
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        if not USE_FALLBACK:
            logger.info("Setting up database extensions and tables (embedding dim: %d)...", vector_dimension)
            # Create native pgvector table
            create_table_ddl = f"""
            CREATE TABLE IF NOT EXISTS document_chunks (
                id SERIAL PRIMARY KEY,
                document_name VARCHAR(255) NOT NULL,
                text_content TEXT NOT NULL,
                embedding vector({vector_dimension}) NOT NULL
            );
            """
            await conn.execute(create_table_ddl)

            # Create HNSW Index using cosine ops matching the <=> search operator
            create_index_ddl = """
            CREATE INDEX IF NOT EXISTS document_chunks_embedding_idx 
            ON document_chunks 
            USING hnsw (embedding vector_cosine_ops);
            """
            await conn.execute(create_index_ddl)
            logger.info("Database schema and HNSW vector index initialized successfully.")
        else:
            logger.info("Setting up database tables in fallback mode (using REAL[])...")
            # Create standard table structure with native array type
            create_table_ddl = """
            CREATE TABLE IF NOT EXISTS document_chunks (
                id SERIAL PRIMARY KEY,
                document_name VARCHAR(255) NOT NULL,
                text_content TEXT NOT NULL,
                embedding REAL[] NOT NULL
            );
            """
            await conn.execute(create_table_ddl)
            logger.info("Fallback database schema initialized.")


async def insert_chunks(chunks: list[dict]):
    """Bulk inserts document chunks and their embeddings into pgvector (or array fallback)."""
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
    """Queries for the top k most similar chunks using cosine distance."""
    global USE_FALLBACK
    pool = await get_db_pool()
    
    if not USE_FALLBACK:
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
    else:
        # Fallback: Query all chunks and calculate similarity in python using numpy
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT document_name, text_content, embedding
                FROM document_chunks;
                """
            )
            
            from ai_service_platform.rag.chunking import cosine_distance
            
            results = []
            for row in rows:
                dist = cosine_distance(query_embedding, row["embedding"])
                results.append({
                    "document_name": row["document_name"],
                    "text_content": row["text_content"],
                    "distance": dist
                })
            
            # Sort by distance ascending and return top K
            results.sort(key=lambda x: x["distance"])
            return results[:limit]
