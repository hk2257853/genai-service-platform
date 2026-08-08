from fastapi import APIRouter, Depends

from ai_service_platform.schemas.rag import IngestResponse
from ai_service_platform.schemas.llamaindex_rag import (
    LlamaIndexIngestRequest,
    LlamaIndexAskRequest,
    LlamaIndexAskResponse
)
from ai_service_platform.services.llamaindex_rag import LlamaIndexRAGService

router = APIRouter(tags=["LlamaIndex RAG"])

_llamaindex_service_instance: LlamaIndexRAGService | None = None


def get_llamaindex_service() -> LlamaIndexRAGService:
    global _llamaindex_service_instance
    if _llamaindex_service_instance is None:
        _llamaindex_service_instance = LlamaIndexRAGService()
    return _llamaindex_service_instance


@router.post("/llamaindex/ingest", response_model=IngestResponse)
async def llamaindex_ingest_endpoint(
    request: LlamaIndexIngestRequest,
    service: LlamaIndexRAGService = Depends(get_llamaindex_service)
):
    """Parses a document with LlamaIndex, attaches metadata, splits into nodes, and indexes."""
    try:
        return await service.ingest_doc(
            content_or_path=request.content,
            document_name=request.document_name,
            metadata=request.metadata
        )
    except Exception as e:
        raise e


@router.post("/llamaindex/ask", response_model=LlamaIndexAskResponse)
async def llamaindex_ask_endpoint(
    request: LlamaIndexAskRequest,
    service: LlamaIndexRAGService = Depends(get_llamaindex_service)
):
    """Retrieves metadata-filtered context via LlamaIndex pipeline and synthesizes grounded answers."""
    try:
        return await service.ask_question(
            query=request.question,
            limit=request.limit,
            filters=request.filters
        )
    except Exception as e:
        raise e
