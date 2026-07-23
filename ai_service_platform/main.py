import logging
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from ai_service_platform.api.health import router as health_router
from ai_service_platform.api.generation import router as generation_router
from ai_service_platform.api.rag import router as rag_router
from ai_service_platform.core.exceptions import (
    LLMException,
    LLMAuthenticationError,
    LLMRateLimitError,
    LLMConnectionError,
    LLMTimeoutError,
    LLMAPIError
)

# Production logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("main_app")

app = FastAPI(
    title="AI Service Platform",
    description="Production-grade AI microservice platform with layered clean architecture: Core Infra, Schemas, Services, API, and RAG pipelines.",
    version="1.0.0"
)

# Register domain API routers under /api/v1 prefix
app.include_router(health_router, prefix="/api/v1")
app.include_router(generation_router, prefix="/api/v1")
app.include_router(rag_router, prefix="/api/v1")


# Global Exception Handlers mapping domain exceptions to HTTP status codes
@app.exception_handler(LLMAuthenticationError)
async def auth_exception_handler(request: Request, exc: LLMAuthenticationError):
    logger.critical("Authentication error raised globally: %s", exc)
    return JSONResponse(
        status_code=401,
        content={"detail": "Upstream LLM authentication failed. Ensure correct keys are configured.", "error": str(exc)}
    )


@app.exception_handler(LLMRateLimitError)
async def rate_limit_exception_handler(request: Request, exc: LLMRateLimitError):
    logger.error("Rate limit error raised globally: %s", exc)
    return JSONResponse(
        status_code=429,
        content={"detail": "Upstream LLM provider rate limit exceeded. Please retry later.", "error": str(exc)}
    )


@app.exception_handler(LLMTimeoutError)
async def timeout_exception_handler(request: Request, exc: LLMTimeoutError):
    logger.error("Timeout error raised globally: %s", exc)
    return JSONResponse(
        status_code=504,
        content={"detail": "Upstream LLM request timed out.", "error": str(exc)}
    )


@app.exception_handler(LLMConnectionError)
async def connection_exception_handler(request: Request, exc: LLMConnectionError):
    logger.error("Connection error raised globally: %s", exc)
    return JSONResponse(
        status_code=503,
        content={"detail": "Upstream LLM service unavailable or unreachable.", "error": str(exc)}
    )


@app.exception_handler(LLMAPIError)
async def api_exception_handler(request: Request, exc: LLMAPIError):
    logger.error("API error raised globally: %s", exc)
    return JSONResponse(
        status_code=502,
        content={"detail": "Upstream LLM provider returned bad gateway / internal error.", "error": str(exc)}
    )


@app.exception_handler(LLMException)
async def general_llm_exception_handler(request: Request, exc: LLMException):
    logger.error("General LLM exception raised globally: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal LLM service error occurred.", "error": str(exc)}
    )


if __name__ == "__main__":
    uvicorn.run("ai_service_platform.main:app", host="127.0.0.1", port=8000, reload=True)
