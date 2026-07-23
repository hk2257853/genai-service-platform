from fastapi import APIRouter, Depends, HTTPException

from ai_service_platform.schemas.rag import (
    IngestRequest,
    AskRequest,
    IngestResponse,
    AskResponse
)
from ai_service_platform.services.rag import RAGService

router = APIRouter(tags=["RAG"])


def get_rag_service() -> RAGService:
    return RAGService()


@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(
    request: IngestRequest,
    service: RAGService = Depends(get_rag_service)
):
    """Parses a document, splits it into chunks, embeds them, and saves to database."""
    if request.method not in ("semantic", "fixed"):
        raise HTTPException(status_code=400, detail="Invalid chunking method. Must be 'semantic' or 'fixed'.")
    
    try:
        return await service.ingest_doc(
            content_or_path=request.content,
            document_name=request.document_name,
            method=request.method
        )
    except Exception as e:
        raise e


@router.post("/ask", response_model=AskResponse)
async def ask_question_endpoint(
    request: AskRequest,
    service: RAGService = Depends(get_rag_service)
):
    """Retrieves similar chunks from vector database and routes to LLM for grounded Q&A."""
    try:
        return await service.ask_question(
            query=request.question,
            limit=request.limit
        )
    except Exception as e:
        raise e
