import logging
from typing import Dict, Any, Optional
from litellm import completion_cost

from ai_service_platform import config
from ai_service_platform.core.client import AsyncLLMClient
from ai_service_platform.rag.llamaindex_pipeline import LlamaIndexPipeline
from ai_service_platform.schemas.rag import SourceMetadata, IngestResponse
from ai_service_platform.schemas.llamaindex_rag import LlamaIndexAskResponse

logger = logging.getLogger("llamaindex_service")


class LlamaIndexRAGService:
    """Service layer coordinating LlamaIndex framework-first ingestion, metadata-filtered retrieval, and Q&A."""

    def __init__(
        self, 
        llm_client: AsyncLLMClient | None = None,
        pipeline: LlamaIndexPipeline | None = None
    ):
        self.llm_client = llm_client or AsyncLLMClient(default_model=config.MODEL_ID)
        self.pipeline = pipeline or LlamaIndexPipeline()

    async def ingest_doc(
        self, 
        content_or_path: str, 
        document_name: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> IngestResponse:
        """Parses a document, attaches metadata, splits into nodes, and indexes using LlamaIndex."""
        chunk_count = self.pipeline.ingest(
            content_or_path=content_or_path,
            document_name=document_name,
            metadata=metadata
        )
        
        return IngestResponse(
            document_name=document_name,
            chunk_count=chunk_count,
            method_used="LlamaIndex SentenceSplitter + VectorStoreIndex"
        )

    async def ask_question(
        self, 
        query: str, 
        limit: int = 3, 
        filters: Optional[Dict[str, Any]] = None
    ) -> LlamaIndexAskResponse:
        """Retrieves metadata-filtered nodes via LlamaIndex and executes grounded LLM Q&A."""
        logger.info("Executing LlamaIndex Q&A for query: '%s' with filters: %s", query, filters)

        retrieved_nodes = self.pipeline.retrieve_with_filters(
            query=query,
            limit=limit,
            filters=filters
        )

        sources = []
        context_parts = []

        for row in retrieved_nodes:
            text = row["text_content"]
            snippet = text[:150] + ("..." if len(text) > 150 else "")
            
            # Distance / score mapping
            score = row["score"]
            distance = 1.0 - score if score <= 1.0 else 0.0
            
            sources.append(SourceMetadata(
                document_name=row["document_name"],
                snippet=snippet,
                distance=distance
            ))
            context_parts.append(text)

        context_text = "\n---\n".join(context_parts) if context_parts else "No context found matching the criteria."
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
            action_name="llamaindex_ask_llm",
            messages=messages,
            temperature=0.2
        )

        answer = response.choices[0].message.content or ""
        usage = response.usage
        cost = completion_cost(completion_response=response) or 0.0

        return LlamaIndexAskResponse(
            answer=answer,
            sources=sources,
            filters_applied=filters,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            cost=cost
        )
