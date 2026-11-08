"""Centralized Token Accounting and USD Financial Cost Tracker.

Maintains live, thread-safe operational metrics across every LLM call in the platform:
- Prompt / completion / total token counts
- Exact financial cost in USD (via LiteLLM cost tables)
- Wall-clock execution latencies and Time-to-First-Token (TTFT)
- Per-model aggregations and rolling historical invocations
"""

import threading
import uuid
from collections import deque
from datetime import datetime, timezone
from typing import Dict, List, Optional

from ai_service_platform.schemas.observability import (
    LLMCallRecord,
    ModelCostBreakdown,
    CostSummaryResponse
)


class CostTracker:
    """Thread-safe singleton engine aggregating LLM operational costs and token telemetry."""

    def __init__(self, max_history: int = 1000):
        self._max_history = max_history
        self._lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        """Resets all metrics to clean initial state."""
        with getattr(self, "_lock", threading.Lock()):
            self._tracking_since = datetime.now(timezone.utc).isoformat()
            self._latest_call_at: Optional[str] = None
            self._calls: deque[LLMCallRecord] = deque(maxlen=self._max_history)
            self._total_calls = 0
            self._total_prompt_tokens = 0
            self._total_completion_tokens = 0
            self._total_tokens = 0
            self._total_cost_usd = 0.0
            self._total_latency_ms = 0.0
            self._total_ttft_ms = 0.0
            self._ttft_call_count = 0
            self._per_model: Dict[str, ModelCostBreakdown] = {}

    def record_call(
        self,
        action_name: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        cost_usd: float,
        latency_ms: float,
        stream: bool = False,
        ttft_ms: Optional[float] = None,
        status: str = "success",
        error_message: Optional[str] = None,
        call_id: Optional[str] = None
    ) -> LLMCallRecord:
        """Records an individual LLM completion call and updates cumulative metrics.

        Args:
            action_name: Logical domain operation name (e.g. 'classify_intent').
            model: Model identifier string (e.g. 'gemini/gemini-flash-latest').
            prompt_tokens: Number of prompt/context tokens.
            completion_tokens: Number of generated completion tokens.
            cost_usd: Cost calculated for the request in USD.
            latency_ms: Total duration in milliseconds.
            stream: Whether this was a streaming invocation.
            ttft_ms: Time-to-First-Token in milliseconds (for streaming calls).
            status: 'success' or 'error'.
            error_message: Optional error message if failed.
            call_id: Optional unique identifier; auto-generated if omitted.

        Returns:
            The created LLMCallRecord instance.
        """
        now_iso = datetime.now(timezone.utc).isoformat()
        total_tokens = prompt_tokens + completion_tokens
        cid = call_id or uuid.uuid4().hex

        record = LLMCallRecord(
            call_id=cid,
            timestamp=now_iso,
            action_name=action_name,
            model=model,
            stream=stream,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost_usd=cost_usd,
            latency_ms=latency_ms,
            ttft_ms=ttft_ms,
            status=status,
            error_message=error_message
        )

        with self._lock:
            self._latest_call_at = now_iso
            self._calls.append(record)
            self._total_calls += 1
            self._total_prompt_tokens += prompt_tokens
            self._total_completion_tokens += completion_tokens
            self._total_tokens += total_tokens
            self._total_cost_usd += cost_usd
            self._total_latency_ms += latency_ms

            if ttft_ms is not None:
                self._total_ttft_ms += ttft_ms
                self._ttft_call_count += 1

            # Update per-model breakdown
            if model not in self._per_model:
                self._per_model[model] = ModelCostBreakdown(
                    model=model,
                    total_calls=0,
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                    total_cost_usd=0.0,
                    avg_latency_ms=0.0
                )

            m = self._per_model[model]
            prev_calls = m.total_calls
            new_calls = prev_calls + 1
            m.total_calls = new_calls
            m.prompt_tokens += prompt_tokens
            m.completion_tokens += completion_tokens
            m.total_tokens += total_tokens
            m.total_cost_usd += cost_usd
            # Compute new cumulative moving average for latency
            m.avg_latency_ms = ((m.avg_latency_ms * prev_calls) + latency_ms) / new_calls

        return record

    def get_summary(self) -> CostSummaryResponse:
        """Calculates and returns a platform-wide financial and operational summary."""
        with self._lock:
            avg_latency = (self._total_latency_ms / self._total_calls) if self._total_calls > 0 else 0.0
            avg_ttft = (self._total_ttft_ms / self._ttft_call_count) if self._ttft_call_count > 0 else None

            # Deep copy of per_model mapping
            per_model_copy = {k: v.model_copy() for k, v in self._per_model.items()}

            return CostSummaryResponse(
                total_calls=self._total_calls,
                total_prompt_tokens=self._total_prompt_tokens,
                total_completion_tokens=self._total_completion_tokens,
                total_tokens=self._total_tokens,
                total_cost_usd=round(self._total_cost_usd, 8),
                avg_latency_ms=round(avg_latency, 2),
                avg_ttft_ms=round(avg_ttft, 2) if avg_ttft is not None else None,
                per_model=per_model_copy,
                tracking_since=self._tracking_since,
                latest_call_at=self._latest_call_at
            )

    def get_recent_records(self, limit: int = 50) -> List[LLMCallRecord]:
        """Returns the most recent N LLM invocation records."""
        with self._lock:
            safe_limit = min(max(1, limit), len(self._calls))
            # Slice from the right of deque (most recent first)
            recent = list(self._calls)[-safe_limit:]
            recent.reverse()
            return recent


# Global singleton instance
cost_tracker = CostTracker()


def get_cost_tracker() -> CostTracker:
    """Dependency provider for FastAPI or service injections."""
    return cost_tracker
