import os
import logging
from typing import List, Dict, Any
from ai_service_platform import config
from ai_service_platform.core.client import AsyncLLMClient
from ai_service_platform.core.utils import extract_usage_and_cost
from ai_service_platform.core.db import init_db, insert_chunks, search_chunks, get_db_pool
from ai_service_platform.rag.chunking import fixed_size_chunking, semantic_chunking
from ai_service_platform.rag.embeddings import get_embedding, get_embeddings_batch
from ai_service_platform.rag.query_expansion import expand_query
from ai_service_platform.rag.bm25 import BM25Index, reciprocal_rank_fusion
from ai_service_platform.rag.reranker import CrossEncoderReranker
from ai_service_platform.schemas.rag import (
    SourceMetadata, 
    IngestResponse, 
    AskResponse, 
    AskRequest,
    SearchResponse,
    SearchChunkResult
)

logger = logging.getLogger("rag_service")


class RAGService:
    """Production RAG service coordinating query expansion, hybrid search, RRF fusion, reranking, and Q&A."""

    def __init__(
        self, 
        llm_client: AsyncLLMClient | None = None,
        bm25_index: BM25Index | None = None,
        reranker: CrossEncoderReranker | None = None
    ):
        self.llm_client = llm_client or AsyncLLMClient(default_model=config.MODEL_ID)
        self.bm25_index = bm25_index or BM25Index()
        self.reranker = reranker or CrossEncoderReranker()
        self._bm25_synced = False

    async def _ensure_bm25_synced(self):
        """Ensures BM25 index is populated from the database if empty on startup."""
        if not self._bm25_synced and len(self.bm25_index.chunks) == 0:
            try:
                pool = await get_db_pool()
                async with pool.acquire() as conn:
                    rows = await conn.fetch("SELECT document_name, text_content FROM document_chunks;")
                    chunks = [dict(r) for r in rows]
                    if chunks:
                        self.bm25_index.index_chunks(chunks)
                        logger.info("Synced %d chunks from database into BM25 index.", len(chunks))
                self._bm25_synced = True
            except Exception as e:
                logger.debug("Could not auto-sync BM25 index (DB may be uninitialized): %s", e)

    async def ingest_doc(self, content_or_path: str, document_name: str, method: str = "semantic") -> IngestResponse:
        """Reads, chunks, embeds, stores in pgvector/DB, and indexes in BM25."""
        text = content_or_path
        if os.path.exists(content_or_path) and os.path.isfile(content_or_path):
            logger.info("Reading document content from file path: %s", content_or_path)
            with open(content_or_path, "r", encoding="utf-8") as f:
                text = f.read()

        test_emb = await get_embedding("test")
        vector_dim = len(test_emb)
        await init_db(vector_dim)

        logger.info("Chunking document: %s using method: %s", document_name, method)
        if method == "semantic":
            chunks = await semantic_chunking(text, embed_fn=get_embeddings_batch)
        else:
            chunks = fixed_size_chunking(text, chunk_size=500, chunk_overlap=100)

        if not chunks:
            logger.warning("No chunks generated for document: %s", document_name)
            return IngestResponse(document_name=document_name, chunk_count=0, method_used=method)

        logger.info("Generating embeddings for %d chunks...", len(chunks))
        chunk_embeddings = await get_embeddings_batch(chunks)

        db_records = []
        for chunk_text, emb in zip(chunks, chunk_embeddings):
            db_records.append({
                "document_name": document_name,
                "text_content": chunk_text,
                "embedding": emb
            })

        await insert_chunks(db_records)
        
        # Also index into BM25 for sparse keyword search
        self.bm25_index.index_chunks(db_records)
        self._bm25_synced = True
        logger.info("Successfully ingested %d chunks into database and BM25 index.", len(db_records))
        
        return IngestResponse(
            document_name=document_name,
            chunk_count=len(db_records),
            method_used=method
        )

    async def search_pipeline(
        self,
        query: str,
        limit: int = 3,
        use_query_expansion: bool = False,
        use_hybrid: bool = True,
        use_reranker: bool = True
    ) -> Dict[str, Any]:
        """Executes the multi-stage retrieval pipeline returning candidate chunks and intermediate metadata."""
        await self._ensure_bm25_synced()

        # 1. Query Expansion
        if use_query_expansion:
            queries = await expand_query(query, self.llm_client, num_queries=3)
        else:
            queries = [query]

        dense_rankings: List[List[Dict[str, Any]]] = []
        sparse_rankings: List[List[Dict[str, Any]]] = []

        # 2. Multi-query Dense + Sparse retrieval
        for q in queries:
            q_emb = await get_embedding(q)
            dense_hits = await search_chunks(q_emb, limit=10)
            dense_rankings.append(dense_hits)

            if use_hybrid:
                sparse_hits = self.bm25_index.search(q, limit=10)
                sparse_rankings.append(sparse_hits)

        # 3. Fusion (RRF or Flatten)
        if use_hybrid:
            candidates = reciprocal_rank_fusion(dense_rankings, sparse_rankings, k=60)
            retrieval_method = "Hybrid (Dense + BM25 Sparse RRF)"
        else:
            # Flatten dense rankings
            seen = set()
            candidates = []
            for ranking in dense_rankings:
                for hit in ranking:
                    key = f"{hit['document_name']}:::{hit['text_content']}"
                    if key not in seen:
                        seen.add(key)
                        candidates.append(hit)
            retrieval_method = "Dense Vector Search Only"

        if use_query_expansion:
            retrieval_method += " + Query Expansion"

        # 4. Cross-Encoder Reranking
        if use_reranker and candidates:
            final_chunks = self.reranker.rerank(query, candidates, top_k=limit)
            retrieval_method += " + Cross-Encoder Reranking"
        else:
            final_chunks = candidates[:limit]

        return {
            "query": query,
            "expanded_queries": queries,
            "retrieval_method": retrieval_method,
            "final_chunks": final_chunks
        }

    async def ask_question(
        self, 
        query: str, 
        limit: int = 3,
        use_query_expansion: bool = False,
        use_hybrid: bool = True,
        use_reranker: bool = True
    ) -> AskResponse:
        """Executes the full production RAG pipeline (retrieval + reranking + LLM answer synthesis)."""
        logger.info("Processing Production RAG Question: '%s' (hybrid=%s, rerank=%s, expansion=%s)", 
                    query, use_hybrid, use_reranker, use_query_expansion)

        pipeline_result = await self.search_pipeline(
            query=query,
            limit=limit,
            use_query_expansion=use_query_expansion,
            use_hybrid=use_hybrid,
            use_reranker=use_reranker
        )

        final_chunks = pipeline_result["final_chunks"]
        expanded_queries = pipeline_result["expanded_queries"]
        retrieval_method = pipeline_result["retrieval_method"]

        # Assemble prompt sources & context
        sources = []
        context_parts = []

        for row in final_chunks:
            text = row["text_content"]
            snippet = text[:150] + ("..." if len(text) > 150 else "")
            
            sources.append(SourceMetadata(
                document_name=row["document_name"],
                snippet=snippet,
                distance=float(row.get("distance", 0.0)),
                rrf_score=float(row["rrf_score"]) if "rrf_score" in row else None,
                rerank_score=float(row["rerank_score"]) if "rerank_score" in row else None
            ))
            context_parts.append(text)

        context_text = "\n---\n".join(context_parts) if context_parts else "No context found."
        system_prompt = (
            "You are a helpful AI assistant. Use the following retrieved context documents to answer the user's question.\n"
            "If you do not know the answer based on the context, state clearly that you do not know it based on the documentation.\n\n"
            f"Context Documents:\n{context_text}"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]

        response = await self.llm_client.acompletion(
            action_name="rag_ask_llm",
            messages=messages,
            temperature=0.2
        )

        answer = response.choices[0].message.content or ""
        metrics = extract_usage_and_cost(response)

        return AskResponse(
            answer=answer,
            sources=sources,
            expanded_queries=expanded_queries if use_query_expansion else None,
            retrieval_method=retrieval_method,
            **metrics
        )
