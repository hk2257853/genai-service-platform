"""Observability & Telemetry Pydantic Schemas.

Defines data models for:
- StreamMetrics: Time-to-First-Token (TTFT) and streaming latency/token metrics.
- LLMCallRecord: Detailed telemetry for individual LLM invocations.
- CostSummaryResponse: Cumulative token accounting and USD cost breakdown across models.
"""

from datetime import datetime, timezone
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class StreamMetrics(BaseModel):
    """Real-time streaming latency and token consumption metrics."""

    ttft_ms: Optional[float] = Field(
        default=None,
        description="Time-to-First-Token in milliseconds (wall-clock delta from dispatch to first content chunk)"
    )
    total_duration_ms: float = Field(
        default=0.0,
        description="Total duration of the streaming generation in milliseconds"
    )
    total_chunks: int = Field(
        default=0,
        description="Total number of chunks received over the stream"
    )
    prompt_tokens: int = Field(
        default=0,
        description="Number of prompt/input tokens processed"
    )
    completion_tokens: int = Field(
        default=0,
        description="Number of newly generated completion tokens"
    )
    total_tokens: int = Field(
        default=0,
        description="Total tokens consumed (prompt + completion)"
    )
    cost_usd: float = Field(
        default=0.0,
        description="Calculated financial cost of the streaming request in USD"
    )


class LLMCallRecord(BaseModel):
    """Detailed telemetry record captured for an individual LLM call."""

    call_id: str = Field(description="Unique UUID for this specific LLM call")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 UTC timestamp when the call completed"
    )
    action_name: str = Field(description="Logical domain operation name (e.g. classify_intent, query_resolver)")
    model: str = Field(description="LLM model identifier used (e.g. gemini/gemini-flash-latest)")
    stream: bool = Field(default=False, description="Whether this was a streaming invocation")
    prompt_tokens: int = Field(default=0, description="Input prompt token count")
    completion_tokens: int = Field(default=0, description="Output completion token count")
    total_tokens: int = Field(default=0, description="Sum of prompt and completion tokens")
    cost_usd: float = Field(default=0.0, description="Total financial cost in USD")
    latency_ms: float = Field(default=0.0, description="Total execution wall-clock time in milliseconds")
    ttft_ms: Optional[float] = Field(
        default=None,
        description="Time-to-First-Token in milliseconds (for streaming calls)"
    )
    status: str = Field(default="success", description="Outcome status: 'success' or 'error'")
    error_message: Optional[str] = Field(default=None, description="Error details if invocation failed")


class ModelCostBreakdown(BaseModel):
    """Aggregated financial and token metrics partitioned by model identifier."""

    model: str = Field(description="Model identifier")
    total_calls: int = Field(default=0, description="Total number of invocations")
    prompt_tokens: int = Field(default=0, description="Cumulative input tokens")
    completion_tokens: int = Field(default=0, description="Cumulative generated output tokens")
    total_tokens: int = Field(default=0, description="Cumulative total tokens")
    total_cost_usd: float = Field(default=0.0, description="Cumulative USD expenditure")
    avg_latency_ms: float = Field(default=0.0, description="Average wall-clock latency per call in milliseconds")


class CostSummaryResponse(BaseModel):
    """Platform-wide aggregated token accounting and USD cost telemetry."""

    total_calls: int = Field(description="Total number of LLM invocations recorded")
    total_prompt_tokens: int = Field(description="Total input tokens processed across all models")
    total_completion_tokens: int = Field(description="Total completion tokens generated across all models")
    total_tokens: int = Field(description="Total tokens consumed across the platform")
    total_cost_usd: float = Field(description="Total dollar cost in USD incurred across the platform")
    avg_latency_ms: float = Field(description="Average execution latency across all non-streaming and streaming calls")
    avg_ttft_ms: Optional[float] = Field(
        default=None,
        description="Average Time-to-First-Token across all streaming calls in milliseconds"
    )
    per_model: Dict[str, ModelCostBreakdown] = Field(
        default_factory=dict,
        description="Breakdown of token and cost statistics grouped by model"
    )
    tracking_since: str = Field(description="ISO timestamp when this tracking session was initialized")
    latest_call_at: Optional[str] = Field(
        default=None,
        description="ISO timestamp of the most recent recorded LLM invocation"
    )
