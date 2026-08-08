import asyncio
import os
import httpx
import logging

os.environ["MODEL_ID"] = "gemini/gemini-3.1-flash-lite"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("test_llamaindex_rag")

from ai_service_platform.main import app


async def run_llamaindex_rag_tests():
    logger.info("Initializing LlamaIndex Framework-First RAG Integration Tests...")

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        fixtures_dir = os.path.join(os.path.dirname(__file__), "fixtures")
        sample_doc_path = os.path.join(fixtures_dir, "sample_doc.txt")

        # 1. Ingest Database Document with metadata {"category": "database", "domain": "storage"}
        logger.info("1. Ingesting Document 1 into LlamaIndex with metadata: category=database...")
        resp1 = await client.post(
            "/api/v1/llamaindex/ingest",
            json={
                "content": sample_doc_path,
                "document_name": "Database Isolation Guide",
                "metadata": {"category": "database", "domain": "storage"}
            }
        )
        assert resp1.status_code == 200, f"Ingestion 1 failed: {resp1.text}"
        logger.info("Doc 1 Success: %s", resp1.json())

        # 2. Ingest Security Text with metadata {"category": "security", "domain": "auth"}
        sec_text = (
            "Security Policy: All API endpoints must enforce Bearer JWT token authentication. "
            "Internal service communication uses mutual TLS (mTLS) with 2048-bit RSA keys. "
            "Unauthorized access attempts trigger instant 403 Forbidden responses and alert security ops."
        )
        logger.info("2. Ingesting Document 2 into LlamaIndex with metadata: category=security...")
        resp2 = await client.post(
            "/api/v1/llamaindex/ingest",
            json={
                "content": sec_text,
                "document_name": "Security Policy Overview",
                "metadata": {"category": "security", "domain": "auth"}
            }
        )
        assert resp2.status_code == 200, f"Ingestion 2 failed: {resp2.text}"
        logger.info("Doc 2 Success: %s", resp2.json())
        print("=" * 60)

        # 3. Test Metadata Filtering: Search for 'security' category ONLY
        logger.info("3. Querying LlamaIndex with Metadata Filter: category == 'security'...")
        resp_ask_sec = await client.post(
            "/api/v1/llamaindex/ask",
            json={
                "question": "What security mechanisms and keys are used?",
                "limit": 2,
                "filters": {"category": "security"}
            }
        )
        assert resp_ask_sec.status_code == 200, f"LlamaIndex Q&A failed: {resp_ask_sec.text}"
        data_sec = resp_ask_sec.json()
        
        logger.info("Answer: %s", data_sec["answer"])
        logger.info("Filters Applied: %s", data_sec["filters_applied"])
        logger.info("Retrieved Sources:")
        for idx, src in enumerate(data_sec["sources"]):
            logger.info("  Source %d [%s]: %s", idx + 1, src["document_name"], src["snippet"])
            # Assert that the metadata filter strictly isolated the security document
            assert src["document_name"] == "Security Policy Overview", \
                f"Metadata filter failed! Unexpected doc returned: {src['document_name']}"
        print("=" * 60)

        # 4. Test Metadata Filtering: Search for 'database' category ONLY
        logger.info("4. Querying LlamaIndex with Metadata Filter: category == 'database'...")
        resp_ask_db = await client.post(
            "/api/v1/llamaindex/ask",
            json={
                "question": "What is the default isolation level in PostgreSQL?",
                "limit": 2,
                "filters": {"category": "database"}
            }
        )
        assert resp_ask_db.status_code == 200, f"LlamaIndex Q&A failed: {resp_ask_db.text}"
        data_db = resp_ask_db.json()
        
        logger.info("Answer: %s", data_db["answer"])
        logger.info("Filters Applied: %s", data_db["filters_applied"])
        logger.info("Retrieved Sources:")
        for idx, src in enumerate(data_db["sources"]):
            logger.info("  Source %d [%s]: %s", idx + 1, src["document_name"], src["snippet"])
            assert src["document_name"] == "Database Isolation Guide", \
                f"Metadata filter failed! Unexpected doc returned: {src['document_name']}"
        print("=" * 60)

        logger.info("LlamaIndex Framework-First Integration Tests Completed Successfully!")


if __name__ == "__main__":
    asyncio.run(run_llamaindex_rag_tests())
