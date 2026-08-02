import asyncio
import os
import httpx
import logging

os.environ["MODEL_ID"] = "gemini/gemini-3.1-flash-lite"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("test_production_rag")

from ai_service_platform.main import app


async def run_production_rag_tests():
    logger.info("Initializing Production RAG Integration Tests...")

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        # 1. Ingest Sample Document
        fixtures_dir = os.path.join(os.path.dirname(__file__), "fixtures")
        doc_path = os.path.join(fixtures_dir, "sample_doc.txt")
        logger.info("1. Ingesting test document into RAG & BM25: %s", doc_path)
        
        resp = await client.post(
            "/api/v1/ingest",
            json={
                "content": doc_path,
                "document_name": "Database Isolation Guide",
                "method": "semantic"
            }
        )
        assert resp.status_code == 200, f"Ingestion failed: {resp.text}"
        logger.info("Ingestion Success: %s", resp.json())
        print("=" * 60)

        # 2. Test Direct Inspection Endpoint: Dense Only vs. Hybrid BM25 + RRF
        keyword_query = "SSI 40001 serialization failure"
        logger.info("2. Testing /search with Exact Keyword Query: '%s'", keyword_query)

        # 2a. Dense search only
        resp_dense = await client.post(
            "/api/v1/search",
            json={
                "question": keyword_query,
                "limit": 3,
                "use_query_expansion": False,
                "use_hybrid": False,
                "use_reranker": False
            }
        )
        assert resp_dense.status_code == 200, f"Dense search failed: {resp_dense.text}"
        data_dense = resp_dense.json()
        logger.info("Dense Only Search Chunks Retrieved: %d", data_dense["retrieved_count"])

        # 2b. Hybrid (Dense + BM25 RRF) + Cross-Encoder Reranking
        resp_hybrid = await client.post(
            "/api/v1/search",
            json={
                "question": keyword_query,
                "limit": 3,
                "use_query_expansion": True,
                "use_hybrid": True,
                "use_reranker": True
            }
        )
        assert resp_hybrid.status_code == 200, f"Hybrid search failed: {resp_hybrid.text}"
        data_hybrid = resp_hybrid.json()
        logger.info("Production Hybrid + Reranker Search Method: %s", data_hybrid["chunks"][0] if data_hybrid["chunks"] else "None")
        logger.info("Expanded Queries Generated: %s", data_hybrid["expanded_queries"])
        for idx, chunk in enumerate(data_hybrid["chunks"]):
            logger.info("  Rank %d [RRF: %.4f | Rerank: %.4f]: %s", 
                        idx + 1, 
                        chunk.get("rrf_score") or 0.0, 
                        chunk.get("rerank_score") or 0.0, 
                        chunk["snippet"])
        print("=" * 60)

        # 3. Test Full Production Q&A Endpoint (/ask)
        rag_query = "How does PostgreSQL handle Serializable conflicts and what is the error code?"
        logger.info("3. Testing /ask Endpoint with Full Production RAG Pipeline...")
        logger.info("Question: %s", rag_query)

        resp_ask = await client.post(
            "/api/v1/ask",
            json={
                "question": rag_query,
                "limit": 3,
                "use_query_expansion": True,
                "use_hybrid": True,
                "use_reranker": True
            }
        )
        assert resp_ask.status_code == 200, f"Production RAG Ask failed: {resp_ask.text}"
        ask_data = resp_ask.json()
        
        logger.info("RAG Answer:\n%s", ask_data["answer"])
        logger.info("Retrieval Pipeline: %s", ask_data["retrieval_method"])
        logger.info("Expanded Queries: %s", ask_data["expanded_queries"])
        logger.info("Retrieved Sources:")
        for idx, source in enumerate(ask_data["sources"]):
            logger.info("  Source %d [%s] (Rerank score: %s): %s", 
                        idx + 1, 
                        source["document_name"], 
                        f"{source['rerank_score']:.4f}" if source.get('rerank_score') else "N/A", 
                        source["snippet"])
        logger.info("Token Usage: Prompt=%d, Completion=%d, Total=%d", 
                    ask_data["prompt_tokens"], ask_data["completion_tokens"], ask_data["total_tokens"])
        logger.info("Estimated Cost: $%.8f", ask_data["cost"])
        print("=" * 60)
        logger.info("Production RAG Integration Tests Completed Successfully!")


if __name__ == "__main__":
    asyncio.run(run_production_rag_tests())
