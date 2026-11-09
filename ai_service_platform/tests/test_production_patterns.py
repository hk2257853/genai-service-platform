"""Unit and Integration Tests for Day 17: Production AI Patterns.

Covers:
1. Semantic Caching (Cosine similarity, namespaces, TTL, LRU eviction)
2. Token-Per-Minute (TPM) Sliding-Window Rate Limiting (Acquire, Settle, Release, Window slide)
3. Cross-Provider Fallback Routing & Circuit Breaker Cooldowns
4. REST API Telemetry & Circuit Management Endpoints
"""

import time
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np
import pytest
import httpx

from ai_service_platform.core.semantic_cache import SemanticCache, _l2_normalize
from ai_service_platform.core.tpm_limiter import TPMRateLimiter
from ai_service_platform.core.fallback_router import FallbackRouter
from ai_service_platform.core.exceptions import (
    LLMTPMRateLimitError,
    AllProvidersUnavailableError,
    LLMRateLimitError,
)
from ai_service_platform.main import app


# ---------------------------------------------------------------------------
# Test Helpers: Deterministic Vectors
# ---------------------------------------------------------------------------
def make_vector(val: float, dim: int = 4) -> np.ndarray:
    """Creates a normalized vector for deterministic cosine similarity testing."""
    vec = np.zeros(dim, dtype=np.float32)
    vec[0] = val
    vec[1] = np.sqrt(max(0.0, 1.0 - val * val))
    return _l2_normalize(vec)


# ===========================================================================
# 1. Semantic Cache Tests
# ===========================================================================
class TestSemanticCache:
    """Verifies vector cosine similarity intent matching, namespaces, TTL, and LRU."""

    @pytest.mark.asyncio
    async def test_semantic_cache_hit_on_similar_queries(self):
        print("\n" + "=" * 60)
        print("[TEST INPUT] Query 1: 'What is the refund policy?' (Stored)")
        print("[TEST INPUT] Query 2: 'How do I get my money back?' (Target similarity: 0.95)")
        print("=" * 60)

        # Vector 1: [1.0, 0.0, 0.0, 0.0]
        # Vector 2: [0.95, ~0.312, 0.0, 0.0] -> Dot product = 0.95
        v1 = make_vector(1.0)
        v2 = make_vector(0.95)

        cache = SemanticCache(threshold=0.92, ttl_seconds=3600, max_entries=10)
        await cache.store(
            prompt="What is the refund policy?",
            response={"content": "Refunds are processed within 5 business days."},
            namespace="default",
            metadata={"total_tokens": 42, "cost": 0.0001, "latency_ms": 320.0},
            embedding=v1,
        )

        hit = await cache.lookup(
            prompt="How do I get my money back?",
            namespace="default",
            embedding=v2,
        )

        print("[TEST OUTPUT] Received Cache Lookup Result:")
        print(f"  Hit Found     : {hit is not None}")
        print(f"  Similarity    : {hit.similarity if hit else 0.0:.4f}")
        print(f"  Cached Answer : {hit.response if hit else None}")

        assert hit is not None
        assert hit.similarity >= 0.92
        assert hit.response["content"] == "Refunds are processed within 5 business days."
        assert cache.get_stats()["hits"] == 1
        assert cache.get_stats()["saved_tokens"] == 42

    @pytest.mark.asyncio
    async def test_semantic_cache_miss_on_distinct_queries(self):
        print("\n" + "=" * 60)
        print("[TEST INPUT] Query 1: 'What is the refund policy?' (Stored)")
        print("[TEST INPUT] Query 2: 'What is the capital of France?' (Target similarity: 0.20)")
        print("=" * 60)

        v1 = make_vector(1.0)
        v_distinct = make_vector(0.20)  # Dot product = 0.20 < 0.92

        cache = SemanticCache(threshold=0.92)
        await cache.store(
            prompt="What is the refund policy?",
            response={"content": "Refunds are processed in 5 days."},
            embedding=v1,
        )

        hit = await cache.lookup(
            prompt="What is the capital of France?",
            embedding=v_distinct,
        )

        print("[TEST OUTPUT] Received Cache Lookup Result:")
        print(f"  Hit Found: {hit is not None}")
        print(f"  Stats    : {cache.get_stats()}")

        assert hit is None
        assert cache.get_stats()["misses"] == 1

    @pytest.mark.asyncio
    async def test_semantic_cache_namespace_isolation(self):
        print("\n" + "=" * 60)
        print("[TEST INPUT] Store in namespace='customer_support', Lookup same vector in namespace='code_gen'")
        print("=" * 60)

        v = make_vector(1.0)
        cache = SemanticCache(threshold=0.90)

        await cache.store(
            prompt="Help me with order 101",
            response="Order 101 is shipped.",
            namespace="customer_support",
            embedding=v,
        )

        # Lookup in different namespace
        hit_other_ns = await cache.lookup(
            prompt="Help me with order 101",
            namespace="code_gen",
            embedding=v,
        )

        print(f"[TEST OUTPUT] Namespace 'code_gen' lookup hit: {hit_other_ns}")
        assert hit_other_ns is None

        # Lookup in correct namespace
        hit_same_ns = await cache.lookup(
            prompt="Help me with order 101",
            namespace="customer_support",
            embedding=v,
        )
        print(f"[TEST OUTPUT] Namespace 'customer_support' lookup hit: {hit_same_ns is not None}")
        assert hit_same_ns is not None

    @pytest.mark.asyncio
    async def test_semantic_cache_ttl_expiry(self):
        print("\n" + "=" * 60)
        print("[TEST INPUT] Store entry with ttl_seconds=0.1s, Lookup after 0.2s")
        print("=" * 60)

        v = make_vector(1.0)
        cache = SemanticCache(ttl_seconds=0.1)

        await cache.store(prompt="Status check", response="OK", embedding=v)
        await asyncio.sleep(0.15)

        hit = await cache.lookup(prompt="Status check", embedding=v)
        print(f"[TEST OUTPUT] Post-TTL Lookup Hit: {hit}")
        assert hit is None

    @pytest.mark.asyncio
    async def test_semantic_cache_lru_capacity_eviction(self):
        print("\n" + "=" * 60)
        print("[TEST INPUT] Max entries = 2. Insert Entry 1, 2, then 3. Verify Entry 1 is evicted.")
        print("=" * 60)

        cache = SemanticCache(max_entries=2)
        v1 = make_vector(1.0)
        v2 = make_vector(0.8)
        v3 = make_vector(0.6)

        await cache.store(prompt="q1", response="r1", embedding=v1)
        await cache.store(prompt="q2", response="r2", embedding=v2)
        assert cache.get_stats()["total_cached_entries"] == 2

        # Storing 3rd entry triggers LRU eviction of oldest (q1)
        await cache.store(prompt="q3", response="r3", embedding=v3)
        print(f"[TEST OUTPUT] Total cached entries after 3rd insert: {cache.get_stats()['total_cached_entries']}")
        assert cache.get_stats()["total_cached_entries"] == 2

        # q1 should be evicted -> miss
        hit_q1 = await cache.lookup(prompt="q1", min_threshold=0.99, embedding=v1)
        print(f"[TEST OUTPUT] Evicted entry q1 lookup result: {hit_q1}")
        assert hit_q1 is None


# ===========================================================================
# 2. Token-Per-Minute (TPM) Rate Limiter Tests
# ===========================================================================
class TestTPMRateLimiter:
    """Verifies sliding window token accounting, pre-checks, settlement, and release."""

    def test_tpm_limiter_allows_within_budget(self):
        print("\n" + "=" * 60)
        print("[TEST INPUT] Max TPM: 1,000. Request: 400 tokens")
        print("=" * 60)

        limiter = TPMRateLimiter(max_tpm=1000, window_seconds=60.0)
        res_id = limiter.acquire(400)

        print("[TEST OUTPUT] Acquired Reservation:")
        print(f"  Reservation ID   : {res_id}")
        print(f"  Window Tokens    : {limiter.current_tokens_in_window}")
        print(f"  Remaining Tokens : {limiter.get_status()['remaining_tokens']}")

        assert res_id is not None
        assert limiter.current_tokens_in_window == 400
        assert limiter.get_status()["remaining_tokens"] == 600

    def test_tpm_limiter_rejects_over_budget(self):
        print("\n" + "=" * 60)
        print("[TEST INPUT] Max TPM: 500. Request 1: 400 tokens. Request 2: 200 tokens (Burst Exceeded)")
        print("=" * 60)

        limiter = TPMRateLimiter(max_tpm=500, window_seconds=60.0)
        limiter.acquire(400)

        with pytest.raises(LLMTPMRateLimitError) as exc_info:
            limiter.acquire(200)

        print(f"[TEST OUTPUT] Expected Exception Caught: {type(exc_info.value).__name__}")
        print(f"  Message: {exc_info.value}")
        print(f"  Retry-After: {exc_info.value.retry_after_seconds}s")

        assert exc_info.value.retry_after_seconds > 0
        assert "Token-Per-Minute limit exceeded" in str(exc_info.value)

    def test_tpm_limiter_settle_and_release(self):
        print("\n" + "=" * 60)
        print("[TEST INPUT] Acquire 500 estimated tokens, Settle with 320 actual tokens")
        print("=" * 60)

        limiter = TPMRateLimiter(max_tpm=1000)
        res_id = limiter.acquire(500)
        assert limiter.current_tokens_in_window == 500

        limiter.settle(res_id, actual_tokens=320)
        print(f"[TEST OUTPUT] Post-Settlement Window Tokens: {limiter.current_tokens_in_window}")
        assert limiter.current_tokens_in_window == 320

        # Release should zero it out
        limiter.release(res_id)
        print(f"[TEST OUTPUT] Post-Release Window Tokens: {limiter.current_tokens_in_window}")
        assert limiter.current_tokens_in_window == 0

    def test_tpm_limiter_window_slides(self):
        print("\n" + "=" * 60)
        print("[TEST INPUT] Inject token reservation timestamped 65s ago (older than 60s window)")
        print("=" * 60)

        limiter = TPMRateLimiter(max_tpm=1000, window_seconds=60.0)
        res_id = limiter.acquire(400)

        # Artificially age the reservation timestamp past the window
        limiter._ledger[0].timestamp = time.time() - 65.0

        print(f"[TEST OUTPUT] Before prune tokens in window: {sum(e.reserved_tokens for e in limiter._ledger)}")
        # Check current tokens (triggers prune)
        current = limiter.current_tokens_in_window
        print(f"[TEST OUTPUT] After prune sliding tokens: {current}")
        assert current == 0


# ===========================================================================
# 3. Fallback Router & Circuit Cooldown Tests
# ===========================================================================
class TestFallbackRouter:
    """Verifies cross-provider model routing, manual tripping, and fallback cascades."""

    def test_router_initialization_and_status(self):
        print("\n" + "=" * 60)
        print("[TEST INPUT] Initializing FallbackRouter with primary='gemini/gemini-flash-latest' and fallback='openai/gpt-4o-mini'")
        print("=" * 60)

        router = FallbackRouter(
            primary_model="gemini/gemini-flash-latest",
            fallback_models=["openai/gpt-4o-mini"],
            allowed_fails=3,
            cooldown_time=30,
        )

        status = router.get_deployment_status()
        print(f"[TEST OUTPUT] Deployment Status List: {status}")

        assert len(status) == 2
        assert status[0]["role"] == "primary"
        assert status[0]["circuit_state"] == "CLOSED"
        assert status[1]["role"] == "fallback"
        assert status[1]["circuit_state"] == "CLOSED"

    def test_router_circuit_tripping_and_cooldown(self):
        print("\n" + "=" * 60)
        print("[TEST INPUT] Manually tripping primary model into cooldown for 10s")
        print("=" * 60)

        router = FallbackRouter(
            primary_model="gemini/gemini-flash-latest",
            fallback_models=["openai/gpt-4o-mini"],
        )

        router.trip_model("gemini/gemini-flash-latest", cooldown_seconds=10)
        status = router.get_deployment_status()
        print(f"[TEST OUTPUT] Post-Trip Status: {status}")

        assert router.is_in_cooldown("gemini/gemini-flash-latest") is True
        assert status[0]["circuit_state"] == "OPEN"
        assert status[1]["circuit_state"] == "CLOSED"

        router.reset_cooldowns()
        assert router.is_in_cooldown("gemini/gemini-flash-latest") is False
        assert router.get_deployment_status()[0]["circuit_state"] == "CLOSED"

    @pytest.mark.asyncio
    async def test_router_fast_fails_to_fallback_when_primary_open(self):
        print("\n" + "=" * 60)
        print("[TEST INPUT] Primary circuit is OPEN. acompletion() should bypass primary directly to fallback.")
        print("=" * 60)

        router = FallbackRouter(
            primary_model="gemini/gemini-flash-latest",
            fallback_models=["openai/gpt-4o-mini"],
        )
        router.trip_model("gemini/gemini-flash-latest", cooldown_seconds=30)

        # Mock the underlying router.acompletion
        mock_response = MagicMock()
        mock_response.model = "openai/gpt-4o-mini"
        mock_response.choices = [MagicMock(message=MagicMock(content="Fallback response"))]

        with patch.object(router.router, "acompletion", new=AsyncMock(return_value=mock_response)) as mock_acompletion:
            resp = await router.acompletion(messages=[{"role": "user", "content": "hi"}])
            print(f"[TEST OUTPUT] Response from router: {resp.choices[0].message.content}")
            print(f"[TEST OUTPUT] Model served: {resp.model}")

            assert resp.model == "openai/gpt-4o-mini"
            # Verify router targeted fallback_gateway_1 because primary was OPEN
            mock_acompletion.assert_called_once()
            call_kwargs = mock_acompletion.call_args.kwargs
            assert call_kwargs["model"] == "fallback_gateway_1"

    @pytest.mark.asyncio
    async def test_router_raises_all_providers_unavailable_when_all_circuits_open(self):
        print("\n" + "=" * 60)
        print("[TEST INPUT] Both primary and fallback circuits are OPEN. acompletion() should raise AllProvidersUnavailableError.")
        print("=" * 60)

        router = FallbackRouter(
            primary_model="gemini/gemini-flash-latest",
            fallback_models=["openai/gpt-4o-mini"],
        )
        router.trip_model("gemini/gemini-flash-latest", cooldown_seconds=30)
        router.trip_model("openai/gpt-4o-mini", cooldown_seconds=30)

        with pytest.raises(AllProvidersUnavailableError) as exc_info:
            await router.acompletion(messages=[{"role": "user", "content": "hi"}])

        print(f"[TEST OUTPUT] Caught expected exception: {exc_info.value}")
        assert "All LLM providers are currently in cooldown" in str(exc_info.value)


# ===========================================================================
# 4. REST Endpoints Integration Tests (/api/v1/gateway)
# ===========================================================================
class TestGatewayAPIEndpoints:
    """Verifies REST endpoints for cache statistics, TPM status, and circuit management."""

    @pytest.mark.asyncio
    async def test_cache_stats_and_clear_endpoints(self):
        print("\n" + "=" * 60)
        print("[TEST INPUT] GET /api/v1/gateway/cache/stats -> POST /api/v1/gateway/cache/clear")
        print("=" * 60)

        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            stats_resp = await client.get("/api/v1/gateway/cache/stats")
            print(f"[TEST OUTPUT] GET /gateway/cache/stats -> {stats_resp.status_code}")
            print(stats_resp.json())
            assert stats_resp.status_code == 200
            assert "hit_ratio" in stats_resp.json()

            clear_resp = await client.post("/api/v1/gateway/cache/clear")
            print(f"[TEST OUTPUT] POST /gateway/cache/clear -> {clear_resp.status_code}")
            assert clear_resp.status_code == 200
            assert clear_resp.json()["status"] == "success"

    @pytest.mark.asyncio
    async def test_tpm_status_endpoint(self):
        print("\n" + "=" * 60)
        print("[TEST INPUT] GET /api/v1/gateway/tpm/status")
        print("=" * 60)

        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/api/v1/gateway/tpm/status")
            print(f"[TEST OUTPUT] GET /gateway/tpm/status -> {resp.status_code}")
            print(resp.json())
            assert resp.status_code == 200
            assert "current_window_tokens" in resp.json()
            assert "max_tpm" in resp.json()

    @pytest.mark.asyncio
    async def test_circuits_trip_and_reset_endpoints(self):
        print("\n" + "=" * 60)
        print("[TEST INPUT] GET /gateway/circuits -> POST /gateway/circuits/trip -> POST /gateway/circuits/reset")
        print("=" * 60)

        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            # 1. Initial status
            status_resp = await client.get("/api/v1/gateway/circuits")
            print(f"[TEST OUTPUT] GET /gateway/circuits -> {status_resp.status_code}")
            deployments = status_resp.json()
            assert len(deployments) >= 1
            primary_model = deployments[0]["model"]

            # 2. Trip primary model
            trip_resp = await client.post(
                f"/api/v1/gateway/circuits/trip?model={primary_model}&cooldown_seconds=15"
            )
            print(f"[TEST OUTPUT] POST /gateway/circuits/trip -> {trip_resp.status_code}")
            assert trip_resp.status_code == 200
            assert trip_resp.json()["status"] == "success"

            # Check tripped state
            check_resp = await client.get("/api/v1/gateway/circuits")
            assert check_resp.json()[0]["circuit_state"] == "OPEN"

            # 3. Reset circuits
            reset_resp = await client.post("/api/v1/gateway/circuits/reset")
            print(f"[TEST OUTPUT] POST /gateway/circuits/reset -> {reset_resp.status_code}")
            assert reset_resp.status_code == 200

            # Check reset state
            final_resp = await client.get("/api/v1/gateway/circuits")
            assert final_resp.json()[0]["circuit_state"] == "CLOSED"


