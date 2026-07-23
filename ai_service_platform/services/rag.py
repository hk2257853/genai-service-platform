import os
import logging
from typing import List, Dict, Any
from litellm import completion_cost

from ai_service_platform import config
from ai_service_platform.core.client import AsyncLLMClient
from ai_service_platform.core.db import init_db, insert_chunks, search_chunks
from ai_service_platform.rag.chunking import fixed_size_chunking, semantic_chunking
from ai_service_platform.rag.embeddings import get_embedding, get_embeddings_batch
from ai_service_platform.schemas.rag import SourceMetadata, IngestResponse, AskResponse

logger = logging.getLogger("rag_service")


class RAGService:
    """Service layer coordinating document ingestion, chunking, embedding generation, and Q&A retrieval."""

    def __init__(self, llm_client: AsyncLLMClient | None = None):
        self.llm_client = llm_client or AsyncLLMClient(default_model=config.MODEL_ID)

    async def ingest_doc(self, content_or_path: str, document_name: str, method: str = "semantic") -> IngestResponse:
        """Reads, chunks, embeds, and stores a document in the vector database."""
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
        logger.info("Successfully ingested %d chunks from document: %s", len(db_records), document_name)
        
        return IngestResponse(
            document_name=document_name,
            chunk_count=len(db_records),
            method_used=method
        )

    async def ask_question(self, query: str, limit: int = 3) -> AskResponse:
        """Embeds user query, retrieves top context documents, and executes Q&A completion."""
        logger.info("Processing RAG question: %s", query)

        query_emb = await get_embedding(query)
        search_results = await search_chunks(query_emb, limit=limit)
        
        sources = []
        context_parts = []
        
        for row in search_results:
            text = row["text_content"]
            snippet = text[:150] + ("..." if len(text) > 150 else "")
            sources.append(SourceMetadata(
                document_name=row["document_name"],
                snippet=snippet,
                distance=row["distance"]
            ))
            context_parts.append(text)

        context_text = "\n---\n".join(context_parts)
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
        usage = response.usage
        cost = completion_cost(completion_response=response) or 0.0

        return AskResponse(
            answer=answer,
            sources=sources,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            cost=cost
        )
