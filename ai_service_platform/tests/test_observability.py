"""Comprehensive Unit and Integration Tests for Day 16: Observability, Tracing, TTFT, and Cost Metering."""

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock, patch
import httpx
import pytest

from ai_service_platform.core.cost_tracker import CostTracker, cost_tracker
from ai_service_platform.core.tracing import setup_tracing, is_tracing_enabled, get_trace_run_config
from ai_service_platform.schemas.observability import StreamMetrics
from ai_service_platform.services.generation import LLMService
from ai_service_platform.main import app

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("test_observability")


class TestCostTrackerUnit:
    """Unit tests for the thread-safe CostTracker engine."""

    def test_cost_tracker_basic_recording_and_aggregation(self):
        tracker = CostTracker(max_history=10)
        tracker.reset()

        # Record call 1: Gemini
        rec1 = tracker.record_call(
            action_name="classify_intent",
            model="gemini/gemini-flash-latest",
            prompt_tokens=100,
            completion_tokens=50,
            cost_usd=0.00015,
            latency_ms=250.0,
            stream=False
        )

        assert rec1.action_name == "classify_intent"
        assert rec1.total_tokens == 150
        assert rec1.cost_usd == 0.00015
        assert rec1.status == "success"

        # Record call 2: GPT-4o
        rec2 = tracker.record_call(
            action_name="service_stream",
            model="openai/gpt-4o",
            prompt_tokens=200,
            completion_tokens=100,
            cost_usd=0.0015,
            latency_ms=1200.0,
            stream=True,
            ttft_ms=350.0
        )

        assert rec2.stream is True
        assert rec2.ttft_ms == 350.0

        summary = tracker.get_summary()
        assert summary.total_calls == 2
        assert summary.total_prompt_tokens == 300
        assert summary.total_completion_tokens == 150
        assert summary.total_tokens == 450
        assert summary.total_cost_usd == pytest.approx(0.00165, rel=1e-5)
        assert summary.avg_latency_ms == pytest.approx(725.0, rel=1e-2)
        assert summary.avg_ttft_ms == pytest.approx(350.0, rel=1e-2)

        # Check per-model breakdown
        assert "gemini/gemini-flash-latest" in summary.per_model
        assert "openai/gpt-4o" in summary.per_model
        assert summary.per_model["gemini/gemini-flash-latest"].total_calls == 1
        assert summary.per_model["openai/gpt-4o"].total_calls == 1
        assert summary.per_model["openai/gpt-4o"].total_cost_usd == pytest.approx(0.0015, rel=1e-5)

    def test_cost_tracker_rolling_history_limit(self):
        tracker = CostTracker(max_history=3)
        tracker.reset()

        for i in range(5):
            tracker.record_call(
                action_name=f"call_{i}",
                model="test-model",
                prompt_tokens=10,
                completion_tokens=10,
                cost_usd=0.0001,
                latency_ms=100.0
            )

        summary = tracker.get_summary()
        assert summary.total_calls == 5  # Total counter remains accurate

        recent = tracker.get_recent_records(limit=10)
        assert len(recent) == 3  # Bounded by max_history
        # Most recent first
        assert recent[0].action_name == "call_4"
        assert recent[1].action_name == "call_3"
        assert recent[2].action_name == "call_2"

    def test_cost_tracker_reset(self):
        tracker = CostTracker()
        tracker.record_call(
            action_name="test",
            model="test-model",
            prompt_tokens=50,
            completion_tokens=25,
            cost_usd=0.0005,
            latency_ms=150.0
        )
        assert tracker.get_summary().total_calls == 1

        tracker.reset()
        summary = tracker.get_summary()
        assert summary.total_calls == 0
        assert summary.total_tokens == 0
        assert summary.total_cost_usd == 0.0
        assert len(tracker.get_recent_records()) == 0


class TestTracingSetup:
    """Unit tests for LangSmith tracing configuration helpers."""

    def test_get_trace_run_config(self):
        cfg = get_trace_run_config(
            run_name="agent_turn_1",
            thread_id="th-12345",
            tags=["eval", "unit-test"],
            metadata={"user_tier": "enterprise"}
        )

        assert cfg["run_name"] == "agent_turn_1"
        assert cfg["configurable"]["thread_id"] == "th-12345"
        assert "eval" in cfg["tags"]
        assert cfg["metadata"]["thread_id"] == "th-12345"
        assert cfg["metadata"]["user_tier"] == "enterprise"

    def test_setup_tracing_disabled_when_flag_false(self):
        with patch("ai_service_platform.config.LANGCHAIN_TRACING_V2", False):
            enabled = setup_tracing()
            assert enabled is False


class TestObservabilityAPI:
    """Integration tests for the Observability REST endpoints."""

    @pytest.mark.asyncio
    async def test_observability_endpoints(self):
        cost_tracker.reset()

        # Seed 2 calls into global cost_tracker
        cost_tracker.record_call(
            action_name="test_api_call",
            model="gemini/gemini-flash-latest",
            prompt_tokens=150,
            completion_tokens=60,
            cost_usd=0.00021,
            latency_ms=310.0,
            stream=False
        )
        cost_tracker.record_call(
            action_name="test_stream_call",
            model="gemini/gemini-flash-latest",
            prompt_tokens=220,
            completion_tokens=90,
            cost_usd=0.00031,
            latency_ms=620.0,
            stream=True,
            ttft_ms=180.0
        )

        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            # 1. GET /api/v1/observability/costs
            resp = await client.get("/api/v1/observability/costs")
            assert resp.status_code == 200
            data = resp.json()
            assert data["total_calls"] == 2
            assert data["total_tokens"] == 520
            assert data["total_cost_usd"] > 0
            assert data["avg_ttft_ms"] == pytest.approx(180.0, rel=1e-2)

            # 2. GET /api/v1/observability/costs/recent
            resp_recent = await client.get("/api/v1/observability/costs/recent?limit=5")
            assert resp_recent.status_code == 200
            recent_list = resp_recent.json()
            assert len(recent_list) == 2
            assert recent_list[0]["action_name"] == "test_stream_call"

            # 3. GET /api/v1/observability/tracing/status
            resp_status = await client.get("/api/v1/observability/tracing/status")
            assert resp_status.status_code == 200
            status_data = resp_status.json()
            assert "tracing_enabled" in status_data
            assert "langchain_project" in status_data

            # 4. POST /api/v1/observability/costs/reset
            resp_reset = await client.post("/api/v1/observability/costs/reset")
            assert resp_reset.status_code == 200

            # Verify reset
            resp_after = await client.get("/api/v1/observability/costs")
            assert resp_after.json()["total_calls"] == 0


class TestStreamingTTFTIntegration:
    """Test streaming text generator with TTFT and metrics emission."""

    @pytest.mark.asyncio
    async def test_stream_text_with_metrics_mock(self):
        # Create a mock client that yields mock chunks
        mock_client = MagicMock()

        chunk1 = MagicMock()
        chunk1.choices = [MagicMock()]
        chunk1.choices[0].delta.content = "Hello"
        chunk1._ttft_ms = 142.5
        chunk1.usage = None

        chunk2 = MagicMock()
        chunk2.choices = [MagicMock()]
        chunk2.choices[0].delta.content = " world!"
        chunk2._ttft_ms = 142.5
        chunk2.usage = None

        usage_chunk = MagicMock()
        usage_chunk.model = "gemini/gemini-flash-latest"
        usage_chunk.choices = []
        usage_chunk.usage = MagicMock()
        usage_chunk.usage.prompt_tokens = 15
        usage_chunk.usage.completion_tokens = 5
        usage_chunk.usage.total_tokens = 20

        async def mock_stream_gen(**kwargs):
            yield chunk1
            yield chunk2
            yield usage_chunk

        mock_client.acompletion_stream = mock_stream_gen

        service = LLMService(client=mock_client)
        events = []
        async for evt_type, payload in service.stream_text_with_metrics("test prompt"):
            events.append((evt_type, payload))

        # Check content chunks
        content_events = [p for t, p in events if t == "content"]
        assert "".join(content_events) == "Hello world!"

        # Check metrics event
        metric_events = [p for t, p in events if t == "metrics"]
        assert len(metric_events) == 1
        metrics: StreamMetrics = metric_events[0]
        assert metrics.ttft_ms == 142.5
        assert metrics.prompt_tokens == 15
        assert metrics.completion_tokens == 5
        assert metrics.total_duration_ms > 0
