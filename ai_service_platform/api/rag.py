from fastapi import APIRouter, Depends, HTTPException

from ai_service_platform.schemas.rag import (
    IngestRequest,
    AskRequest,
    IngestResponse,
    AskResponse,
    SearchResponse,
    SearchChunkResult
)
from ai_service_platform.services.rag import RAGService

router = APIRouter(tags=["RAG"])

# Global singleton instance of RAGService so BM25 index persists across requests
_rag_service_instance: RAGService | None = None


def get_rag_service() -> RAGService:
    global _rag_service_instance
    if _rag_service_instance is None:
        _rag_service_instance = RAGService()
    return _rag_service_instance


@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(
    request: IngestRequest,
    service: RAGService = Depends(get_rag_service)
):
    """Parses a document, splits it into chunks, embeds them, and saves to database & BM25 index."""
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
    """Retrieves context via production 4-stage RAG pipeline and generates grounded answers."""
    try:
        return await service.ask_question(
            query=request.question,
            limit=request.limit,
            use_query_expansion=request.use_query_expansion,
            use_hybrid=request.use_hybrid,
            use_reranker=request.use_reranker
        )
    except Exception as e:
        raise e


@router.post("/search", response_model=SearchResponse)
async def search_chunks_endpoint(
    request: AskRequest,
    service: RAGService = Depends(get_rag_service)
):
    """Inspection endpoint: returns retrieved & reranked document chunks with scores without LLM synthesis."""
    try:
        pipeline_result = await service.search_pipeline(
            query=request.question,
            limit=request.limit,
            use_query_expansion=request.use_query_expansion,
            use_hybrid=request.use_hybrid,
            use_reranker=request.use_reranker
        )

        chunk_results = []
        for chunk in pipeline_result["final_chunks"]:
            text = chunk["text_content"]
            snippet = text[:150] + ("..." if len(text) > 150 else "")
            chunk_results.append(SearchChunkResult(
                document_name=chunk["document_name"],
                text_content=text,
                snippet=snippet,
                distance=float(chunk.get("distance", 0.0)) if "distance" in chunk else None,
                bm25_score=float(chunk["score"]) if "score" in chunk else None,
                rrf_score=float(chunk["rrf_score"]) if "rrf_score" in chunk else None,
                rerank_score=float(chunk["rerank_score"]) if "rerank_score" in chunk else None
            ))

        return SearchResponse(
            query=pipeline_result["query"],
            expanded_queries=pipeline_result["expanded_queries"],
            retrieved_count=len(chunk_results),
            chunks=chunk_results
        )
    except Exception as e:
        raise e
