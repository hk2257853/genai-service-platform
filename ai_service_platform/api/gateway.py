"""API Endpoints for LLM Gateway Operations: Semantic Caching, TPM Budgeting, and Circuit Resilience."""

from typing import Any
from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

from ai_service_platform.core.semantic_cache import semantic_cache
from ai_service_platform.core.tpm_limiter import tpm_limiter
from ai_service_platform.core.fallback_router import fallback_router

router = APIRouter(prefix="/gateway", tags=["LLM Gateway"])


class CacheStatsResponse(BaseModel):
    hits: int
    misses: int
    total_lookups: int
    hit_ratio: float
    saved_tokens: int
    saved_cost_usd: float
    total_cached_entries: int
    threshold: float
    ttl_seconds: int
    max_entries: int
    enabled: bool


class TPMStatusResponse(BaseModel):
    current_window_tokens: int
    max_tpm: int
    remaining_tokens: int
    utilization_pct: float
    active_window_seconds: float
    window_entries_count: int


class CircuitDeploymentStatus(BaseModel):
    model: str
    role: str
    circuit_state: str
    cooldown_active: bool
    fail_count: int


@router.get("/cache/stats", response_model=CacheStatsResponse)
async def get_cache_stats():
    """Returns real-time semantic cache performance, hit ratio, and token/cost savings."""
    return semantic_cache.get_stats()


@router.post("/cache/clear")
async def clear_cache():
    """Flushes all semantic cache entries and resets hit/miss counters."""
    semantic_cache.clear()
    return {"status": "success", "message": "Semantic cache flushed successfully"}


@router.get("/tpm/status", response_model=TPMStatusResponse)
async def get_tpm_status():
    """Returns sliding 60-second Token-Per-Minute budget consumption and remaining capacity."""
    return tpm_limiter.get_status()


@router.post("/tpm/reset")
async def reset_tpm_ledger():
    """Resets the sliding window token ledger."""
    tpm_limiter.reset()
    return {"status": "success", "message": "TPM ledger reset successfully"}


@router.get("/circuits", response_model=list[CircuitDeploymentStatus])
async def get_circuit_status():
    """Returns provider deployment status, fail counts, and circuit breaker states (CLOSED/OPEN)."""
    return fallback_router.get_deployment_status()


@router.post("/circuits/trip")
async def trip_model_circuit(
    model: str = Query(..., description="Model identifier to trip into cooldown"),
    cooldown_seconds: int = Query(30, description="Duration of cooldown in seconds")
):
    """Simulates upstream failure by forcing a provider into OPEN circuit state."""
    fallback_router.trip_model(model, cooldown_seconds=cooldown_seconds)
    return {
        "status": "success",
        "message": f"Model '{model}' tripped into cooldown for {cooldown_seconds}s",
        "model": model,
        "cooldown_seconds": cooldown_seconds,
    }


@router.post("/circuits/reset")
async def reset_circuit_breakers():
    """Resets all provider cooldowns and failure counters to normal CLOSED state."""
    fallback_router.reset_cooldowns()
    return {"status": "success", "message": "All circuit breakers reset to CLOSED"}

