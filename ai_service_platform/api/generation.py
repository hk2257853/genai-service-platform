from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from ai_service_platform.schemas.generation import (
    ChatRequest,
    SentimentRequest,
    LLMResponse,
    SentimentAnalysisResponse
)
from ai_service_platform.services.generation import LLMService

router = APIRouter(tags=["Generation"])


def get_llm_service() -> LLMService:
    return LLMService()


@router.post("/chat", response_model=LLMResponse)
async def generate_chat(
    request: ChatRequest, 
    service: LLMService = Depends(get_llm_service)
):
    """Generates standard completion responses for a given prompt."""
    try:
        return await service.generate_text(request.prompt)
    except Exception as e:
        raise e


@router.post("/sentiment", response_model=SentimentAnalysisResponse)
async def analyze_sentiment(
    request: SentimentRequest, 
    service: LLMService = Depends(get_llm_service)
):
    """Analyzes customer feedback sentiment returning a validated Pydantic response."""
    try:
        return await service.analyze_sentiment(request.feedback)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise e


@router.post("/stream")
async def stream_chat(
    request: ChatRequest, 
    service: LLMService = Depends(get_llm_service)
):
    """Streams token-by-token response content using Server-Sent Events (SSE)."""
    async def sse_generator():
        try:
            async for token in service.stream_text(request.prompt):
                yield f"data: {token}\n\n"
        except Exception as e:
            yield f"event: error\ndata: {str(e)}\n\n"

    return StreamingResponse(sse_generator(), media_type="text/event-stream")
