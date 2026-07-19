from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from day03.service import RAGService, IngestResponse, AskResponse

router = APIRouter()


# Request validation schemas
class IngestRequest(BaseModel):
    content: str = Field(..., min_length=5, description="Document raw text content or absolute local file path.")
    document_name: str = Field(..., min_length=2, description="Identifying name of the document.")
    method: str = Field("semantic", description="Chunking strategy: 'semantic' (recommended) or 'fixed'.")


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, description="The RAG query / question to answer.")
    limit: int = Field(3, ge=1, le=10, description="Max source context documents to retrieve.")


# Dependency injection helper
def get_rag_service() -> RAGService:
    return RAGService()


@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(
    request: IngestRequest,
    service: RAGService = Depends(get_rag_service)
):
    """Parses a document, splits it into chunks, embeds them, and saves to pgvector database."""
    if request.method not in ("semantic", "fixed"):
        raise HTTPException(status_code=400, detail="Invalid chunking method. Must be 'semantic' or 'fixed'.")
    
    try:
        response = await service.ingest_doc(
            content_or_path=request.content,
            document_name=request.document_name,
            method=request.method
        )
        return response
    except Exception as e:
        # Custom global handlers map general exceptions, but we log here for visibility
        raise e


@router.post("/ask", response_model=AskResponse)
async def ask_question_endpoint(
    request: AskRequest,
    service: RAGService = Depends(get_rag_service)
):
    """Retrieves similar chunks from pgvector and routes to LLM to return context-based answers."""
    try:
        response = await service.ask_question(
            query=request.question,
            limit=request.limit
        )
        return response
    except Exception as e:
        raise e
