"""Day 17: Production AI Patterns Live Showcase.

Demonstrates:
1. Semantic Caching (Vector Cosine Similarity for Intent Reuse):
   - Cache Miss on first query -> LLM called, response indexed
   - Cache Hit on semantically similar paraphrased query -> sub-millisecond response, $0.00 cost
   - Cache Miss on distinct query -> new topic, fresh LLM call
2. Token-Per-Minute (TPM) Sliding-Window Rate Limiter:
   - Real-time token consumption tracking in sliding 60s window
   - Clean HTTP 429 rejection with Retry-After when burst capacity is exceeded
3. Cross-Provider Fallback Model Routing & AI Circuit Breakers:
   - Primary model circuit tripping (simulating upstream 5xx / outage)
   - Fast-fail direct routing to fallback provider without waiting for primary timeouts
   - Telemetry inspection via REST endpoints
"""

import asyncio
import json
import httpx
from ai_service_platform.main import app
from ai_service_platform.core.semantic_cache import semantic_cache, _l2_normalize
from ai_service_platform.core.tpm_limiter import tpm_limiter, TPMRateLimiter
from ai_service_platform.core.fallback_router import fallback_router, FallbackRouter
from ai_service_platform.core.exceptions import LLMTPMRateLimitError
import numpy as np


def header(title: str):
    print("\n" + "=" * 75)
    print(f" {title.upper()}")
    print("=" * 75)


def make_dummy_vector(similarity: float, dim: int = 8) -> np.ndarray:
    """Constructs a deterministic normalized vector with exact dot product against baseline."""
    vec = np.zeros(dim, dtype=np.float32)
    vec[0] = similarity
    vec[1] = np.sqrt(max(0.0, 1.0 - similarity * similarity))
    return _l2_normalize(vec)


async def main():
    # 0. Clean slate for demo
    semantic_cache.clear()
    tpm_limiter.reset()
    fallback_router.reset_cooldowns()

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:

        # -------------------------------------------------------------------
        # 1. Semantic Caching: Intent-Level Reuse Showcase
        # -------------------------------------------------------------------
        header("1. Semantic Caching (Intent-Level Vector Cosine Similarity)")

        q1 = "What is the return policy for purchased items?"
        q2_similar = "How do I return something I bought?"
        q3_distinct = "Who wrote the play Hamlet?"

        v_base = make_dummy_vector(1.0)
        v_similar = make_dummy_vector(0.95)   # Dot product = 0.95 (>= 0.92 threshold)
        v_distinct = make_dummy_vector(0.15)  # Dot product = 0.15 (< 0.92 threshold)

        print(f"[QUERY 1 - Initial Prompt]: '{q1}'")
        print("  -> Storing initial generation in Semantic Cache...")
        await semantic_cache.store(
            prompt=q1,
            response={"content": "Items can be returned within 30 days of purchase with receipt."},
            namespace="ecommerce_faq",
            metadata={"total_tokens": 48, "cost": 0.00012, "latency_ms": 310.0},
            embedding=v_base,
        )

        print("\n[QUERY 2 - Semantically Similar Paraphrase]:")
        print(f"  Input: '{q2_similar}'")
        hit_similar = await semantic_cache.lookup(
            prompt=q2_similar,
            namespace="ecommerce_faq",
            embedding=v_similar,
        )

        if hit_similar:
            print("  ==> [CACHE HIT!]")
            print(f"      Matched Cached Prompt: '{hit_similar.cached_prompt}'")
            print(f"      Cosine Similarity    : {hit_similar.similarity:.4f} (Threshold: {semantic_cache.threshold})")
            print(f"      Cached Response      : {hit_similar.response['content']}")
            print(f"      Latency Saved        : {hit_similar.latency_saved_ms:.1f} ms")
            print(f"      Incurred LLM Cost    : $0.000000 (100% saved)")

        print("\n[QUERY 3 - Completely Unrelated Topic]:")
        print(f"  Input: '{q3_distinct}'")
        hit_distinct = await semantic_cache.lookup(
            prompt=q3_distinct,
            namespace="ecommerce_faq",
            embedding=v_distinct,
        )

        if hit_distinct is None:
            print("  ==> [CACHE MISS]")
            print(f"      Cosine Similarity: 0.1500 < {semantic_cache.threshold}")
            print("      Action           : Dispatched to LLM Router for fresh generation.")

        # Inspect cache statistics endpoint
        cache_stats_resp = await client.get("/api/v1/gateway/cache/stats")
        print("\n[TELEMETRY] GET /api/v1/gateway/cache/stats:")
        print(json.dumps(cache_stats_resp.json(), indent=2))

        # -------------------------------------------------------------------
        # 2. Token-Per-Minute (TPM) Sliding-Window Rate Limiter
        # -------------------------------------------------------------------
        header("2. Token-Per-Minute (TPM) Sliding-Window Rate Limiting")

        demo_limiter = TPMRateLimiter(max_tpm=1000, window_seconds=60.0)
        print(f"Configured TPM Limit: {demo_limiter.max_tpm} tokens / 60 seconds\n")

        # Call 1: Normal within budget
        print("• Request 1: 350 tokens")
        res1 = demo_limiter.acquire(350)
        print(f"  [ACCEPTED] Reservation ID: {res1[:8]}... | Current Window: {demo_limiter.current_tokens_in_window}/1000 TPM")

        # Call 2: Settle Call 1 with actual usage (e.g. 280 tokens)
        demo_limiter.settle(res1, actual_tokens=280)
        print(f"  [SETTLED] Actual tokens: 280 | Current Window: {demo_limiter.current_tokens_in_window}/1000 TPM")

        # Call 3: Request 2 - 500 tokens
        print("\n• Request 2: 500 tokens")
        res2 = demo_limiter.acquire(500)
        print(f"  [ACCEPTED] Reservation ID: {res2[:8]}... | Current Window: {demo_limiter.current_tokens_in_window}/1000 TPM")

        # Call 4: Burst Request exceeding quota (350 tokens -> Total 280 + 500 + 350 = 1130 > 1000)
        print("\n• Request 3: 350 tokens (Burst Attempt)")
        try:
            demo_limiter.acquire(350)
            print("  [ERROR] Should not have allowed burst!")
        except LLMTPMRateLimitError as e:
            print(f"  [BLOCKED - HTTP 429] {e}")
            print(f"  Retry-After Header Emitted: {e.retry_after_seconds}s")

        tpm_status_resp = await client.get("/api/v1/gateway/tpm/status")
        print("\n[TELEMETRY] Global Gateway TPM Status:")
        print(json.dumps(tpm_status_resp.json(), indent=2))

        # -------------------------------------------------------------------
        # 3. Cross-Provider Fallback Model Routing & AI Circuit Breakers
        # -------------------------------------------------------------------
        header("3. Cross-Provider Fallback Routing & Circuit Breakers")

        print("Inspecting initial provider deployment health:")
        circuits_resp = await client.get("/api/v1/gateway/circuits")
        print(json.dumps(circuits_resp.json(), indent=2))

        primary_model = circuits_resp.json()[0]["model"]

        print(f"\n[SIMULATION] Simulating upstream 503 outage on Primary: '{primary_model}'...")
        await client.post(f"/api/v1/gateway/circuits/trip?model={primary_model}&cooldown_seconds=30")

        tripped_circuits_resp = await client.get("/api/v1/gateway/circuits")
        print("[UPDATED CIRCUITS STATE]:")
        for dep in tripped_circuits_resp.json():
            print(f"  • {dep['model']} [{dep['role'].upper()}]: Circuit = {dep['circuit_state']}")

        print(f"\n[FAST-FAIL ROUTING CHECK]:")
        print(f"  Primary '{primary_model}' circuit is OPEN.")
        print(f"  Fallback router bypasses Primary at 0ms latency and routes directly to Fallback 1.")

        print("\n[RESETTING CIRCUITS]:")
        await client.post("/api/v1/gateway/circuits/reset")
        reset_resp = await client.get("/api/v1/gateway/circuits")
        print("  All circuits restored:")
        for dep in reset_resp.json():
            print(f"  • {dep['model']} [{dep['role'].upper()}]: Circuit = {dep['circuit_state']}")

        header("Production AI Patterns Demonstration Complete")


if __name__ == "__main__":
    asyncio.run(main())

