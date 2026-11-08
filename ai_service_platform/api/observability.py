"""API Router for AI Observability, Telemetry, and Financial Cost Monitoring."""

from typing import List
from fastapi import APIRouter, Depends, Query

from ai_service_platform.core.cost_tracker import CostTracker, get_cost_tracker
from ai_service_platform.core.tracing import is_tracing_enabled
from ai_service_platform import config
from ai_service_platform.schemas.observability import (
    CostSummaryResponse,
    LLMCallRecord
)

router = APIRouter(prefix="/observability", tags=["Observability & Tracing"])


@router.get("/costs", response_model=CostSummaryResponse)
async def get_cost_summary(
    tracker: CostTracker = Depends(get_cost_tracker)
) -> CostSummaryResponse:
    """Returns cumulative platform-wide token consumption and USD cost analytics."""
    return tracker.get_summary()


@router.get("/costs/recent", response_model=List[LLMCallRecord])
async def get_recent_calls(
    limit: int = Query(default=50, ge=1, le=500, description="Maximum number of historical records to retrieve"),
    tracker: CostTracker = Depends(get_cost_tracker)
) -> List[LLMCallRecord]:
    """Retrieves the most recent individual LLM completion invocations with latency and token telemetry."""
    return tracker.get_recent_records(limit=limit)


@router.post("/costs/reset")
async def reset_cost_tracker(
    tracker: CostTracker = Depends(get_cost_tracker)
):
    """Resets all live cost and token telemetry counters (useful for benchmarks and isolated test runs)."""
    tracker.reset()
    return {"message": "Operational cost and token counters successfully reset."}


@router.get("/tracing/status")
async def get_tracing_status():
    """Returns the live status of LangSmith hierarchical run-tree tracing."""
    return {
        "tracing_enabled": is_tracing_enabled(),
        "langchain_project": config.LANGCHAIN_PROJECT,
        "langchain_endpoint": config.LANGCHAIN_ENDPOINT,
        "has_api_key": bool(config.LANGCHAIN_API_KEY)
    }
