import time
import logging
from typing import Any, AsyncIterator, Tuple

from ai_service_platform import config
from ai_service_platform.core.client import AsyncLLMClient
from ai_service_platform.core.utils import extract_usage_and_cost
from ai_service_platform.schemas.generation import (
    ProductSentimentAnalysis,
    LLMResponse,
    SentimentAnalysisResponse
)
from ai_service_platform.schemas.observability import StreamMetrics

logger = logging.getLogger("generation_service")


class LLMService:
    """Service layer executing LLM text generation, structured analysis, and streaming with telemetry."""

    def __init__(self, client: AsyncLLMClient | None = None):
        self.client = client or AsyncLLMClient(default_model=config.MODEL_ID)

    async def generate_text(self, prompt: str) -> LLMResponse:
        """Asynchronously generates text and tracks cost and usage."""
        response = await self.client.acompletion(
            action_name="service_generate",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )

        content = response.choices[0].message.content or ""
        metrics = extract_usage_and_cost(response)

        return LLMResponse(
            content=content,
            **metrics
        )

    async def analyze_sentiment(self, feedback_text: str) -> SentimentAnalysisResponse:
        """Asynchronously analyzes feedback text and validates against structured schema."""
        response = await self.client.acompletion(
            action_name="service_sentiment",
            messages=[{"role": "user", "content": f"Analyze this customer feedback:\n\n{feedback_text}"}],
            response_format=ProductSentimentAnalysis,
            temperature=config.STRUCTURED_TEMPERATURE
        )

        raw_json_str = response.choices[0].message.content or ""
        
        try:
            analysis = ProductSentimentAnalysis.model_validate_json(raw_json_str)
        except Exception as e:
            logger.error("Failed to validate JSON response schema: %s. Raw content: %s", e, raw_json_str)
            raise ValueError(f"Model returned invalid schema format: {e}") from e

        metrics = extract_usage_and_cost(response)

        return SentimentAnalysisResponse(
            analysis=analysis,
            **metrics
        )

    async def stream_text_with_metrics(self, prompt: str) -> AsyncIterator[Tuple[str, Any]]:
        """Asynchronously streams generated tokens and emits a final StreamMetrics event containing TTFT."""
        t_start = time.perf_counter()
        stream_generator = self.client.acompletion_stream(
            action_name="service_stream",
            messages=[{"role": "user", "content": prompt}],
            stream_options={"include_usage": True}
        )

        ttft_ms: float | None = None
        chunk_count = 0
        prompt_tokens = 0
        completion_tokens = 0
        cost_usd = 0.0

        async for chunk in stream_generator:
            chunk_count += 1
            if ttft_ms is None and hasattr(chunk, "_ttft_ms") and chunk._ttft_ms is not None:
                ttft_ms = chunk._ttft_ms

            if hasattr(chunk, "usage") and chunk.usage is not None:
                usage_metrics = extract_usage_and_cost(chunk)
                prompt_tokens = usage_metrics["prompt_tokens"]
                completion_tokens = usage_metrics["completion_tokens"]
                cost_usd = usage_metrics["cost"]
            elif len(chunk.choices) > 0:
                content = chunk.choices[0].delta.content
                if content:
                    yield ("content", content)

        total_ms = (time.perf_counter() - t_start) * 1000
        metrics = StreamMetrics(
            ttft_ms=round(ttft_ms, 2) if ttft_ms is not None else None,
            total_duration_ms=round(total_ms, 2),
            total_chunks=chunk_count,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            cost_usd=round(cost_usd, 8)
        )
        yield ("metrics", metrics)

    async def stream_text(self, prompt: str) -> AsyncIterator[str]:
        """Asynchronously streams text content tokens (backwards-compatible plain string generator)."""
        async for event_type, payload in self.stream_text_with_metrics(prompt):
            if event_type == "content":
                yield payload
