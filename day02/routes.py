from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from day02.service import LLMService, LLMResponse, SentimentAnalysisResponse

router = APIRouter()

# Input Validation Models
class ChatRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="Prompt message to send to the LLM.")


class SentimentRequest(BaseModel):
    feedback: str = Field(..., min_length=5, description="Feedback text to perform sentiment analysis on.")


# Dependency Injection for service layer
def get_llm_service() -> LLMService:
    return LLMService()


@router.get("/health", status_code=200)
async def health_check():
    """Simple API health check endpoint."""
    return {"status": "healthy"}


@router.post("/chat", response_model=LLMResponse)
async def generate_chat(
    request: ChatRequest, 
    service: LLMService = Depends(get_llm_service)
):
    """Generates standard completion responses for a given prompt."""
    try:
        response = await service.generate_text(request.prompt)
        return response
    except Exception as e:
        # Unexpected exceptions caught here; custom domain exceptions handled globally
        raise e


@router.post("/sentiment", response_model=SentimentAnalysisResponse)
async def analyze_sentiment(
    request: SentimentRequest, 
    service: LLMService = Depends(get_llm_service)
):
    """Analyzes customer feedback sentiment returning a validated Pydantic model response."""
    try:
        response = await service.analyze_sentiment(request.feedback)
        return response
    except ValueError as e:
        # Schema validation error or invalid input format
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
                # Standard Server-Sent Events formatting
                yield f"data: {token}\n\n"
        except Exception as e:
            # Yield error event for client visibility mid-stream
            yield f"event: error\ndata: {str(e)}\n\n"

    return StreamingResponse(sse_generator(), media_type="text/event-stream")
