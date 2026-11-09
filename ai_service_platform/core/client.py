import time
import logging
import hashlib
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator
import litellm
from litellm.exceptions import (
    AuthenticationError,
    RateLimitError,
    APIConnectionError,
    ServiceUnavailableError,
    APIError,
    Timeout,
    BadRequestError
)
from tenacity import AsyncRetrying, stop_after_attempt, wait_exponential, retry_if_exception_type

from ai_service_platform import config
from ai_service_platform.core.exceptions import (
    LLMAuthenticationError,
    LLMRateLimitError,
    LLMTPMRateLimitError,
    LLMConnectionError,
    LLMTimeoutError,
    LLMAPIError,
    AllProvidersUnavailableError,
    LLMException
)
from ai_service_platform.core.utils import extract_usage_and_cost
from ai_service_platform.core.cost_tracker import cost_tracker
from ai_service_platform.core.semantic_cache import semantic_cache
from ai_service_platform.core.tpm_limiter import tpm_limiter
from ai_service_platform.core.fallback_router import fallback_router

logger = logging.getLogger("llm_client")


def _extract_prompt_text(messages_or_kwargs: Any) -> tuple[str, str]:
    """Extracts (user_prompt, system_prompt) from messages list or request kwargs."""
    user_prompt = ""
    system_prompt = ""
    messages = []
    if isinstance(messages_or_kwargs, dict):
        messages = messages_or_kwargs.get("messages", [])
        if not messages and "prompt" in messages_or_kwargs:
            user_prompt = str(messages_or_kwargs["prompt"])
    elif isinstance(messages_or_kwargs, list):
        messages = messages_or_kwargs

    for m in messages:
        if isinstance(m, dict):
            role = m.get("role")
            content = m.get("content", "")
            if role == "user":
                user_prompt = str(content)
            elif role == "system":
                system_prompt = str(content)
    return user_prompt, system_prompt

@asynccontextmanager
async def handle_litellm_exceptions_async(action_name: str):
    """Context manager to unify async LiteLLM exception handling and translate to domain exceptions."""
    try:
        yield
    except AuthenticationError as e:
        logger.error("[%s] Authentication failed. Check your API key. Details: %s", action_name, e)
        raise LLMAuthenticationError(f"Authentication failed during: {action_name}", e)
    except RateLimitError as e:
        logger.warning("[%s] Rate limit exceeded. Details: %s", action_name, e)
        raise LLMRateLimitError(f"Rate limit exceeded during: {action_name}", e)
    except BadRequestError as e:
        err_str = str(e).lower()
        if "429" in err_str or "quota" in err_str or "resource_exhausted" in err_str:
            logger.warning("[%s] Quota rate limit in BadRequestError: %s", action_name, e)
            raise LLMRateLimitError(f"Rate limit exceeded during: {action_name}", e)
        logger.error("[%s] Bad request: %s", action_name, e)
        raise LLMException(f"Bad request during: {action_name}", e)
    except Timeout as e:
        logger.warning("[%s] Request timed out. Details: %s", action_name, e)
        raise LLMTimeoutError(f"Request timed out during: {action_name}", e)
    except (APIConnectionError, ServiceUnavailableError) as e:
        logger.warning("[%s] Connection or service unavailable. Details: %s", action_name, e)
        raise LLMConnectionError(f"Connection or service unavailable during: {action_name}", e)
    except APIError as e:
        logger.error("[%s] LiteLLM API error occurred. Details: %s", action_name, e)
        raise LLMAPIError(f"API error during: {action_name}", e)
    except Exception as e:
        logger.error("[%s] Unexpected exception: %s", action_name, e)
        raise LLMException(f"Unexpected error during: {action_name}", e)


class AsyncLLMClient:
    """Production-grade async client wrapping LiteLLM with robust retries, error mapping, TTFT, and telemetry."""

    def __init__(self, default_model: str, max_retries: int = 5, enable_patterns: bool = True):
        self.default_model = default_model
        self.max_retries = max_retries
        self.enable_patterns = enable_patterns

    async def acompletion(self, action_name: str, **kwargs) -> Any:
        """Executes an async completion request wrapped in TPM limiting, semantic caching, fallback routing, and telemetry."""
        model_name = kwargs.setdefault("model", self.default_model)
        t_start = time.perf_counter()

        user_prompt, system_prompt = _extract_prompt_text(kwargs)
        temperature = float(kwargs.get("temperature", 0.7))
        namespace = kwargs.pop("cache_namespace", None)
        if not namespace:
            namespace = f"sys_{hashlib.sha256(system_prompt.encode()).hexdigest()[:12]}" if system_prompt else "default"

        # 1. AI Pattern: TPM Rate Limiter Pre-Check
        reservation_id = None
        if self.enable_patterns:
            estimated_tokens = tpm_limiter.estimate_tokens(kwargs.get("messages", user_prompt))
            reservation_id = tpm_limiter.acquire(estimated_tokens)

        # 2. AI Pattern: Semantic Cache Lookup (for low-temperature deterministic calls)
        if self.enable_patterns and temperature <= 0.3 and user_prompt:
            cache_hit = await semantic_cache.lookup(prompt=user_prompt, namespace=namespace)
            if cache_hit is not None:
                # Release TPM reservation (cache hits consume 0 provider tokens)
                if reservation_id:
                    tpm_limiter.release(reservation_id)

                latency_ms = (time.perf_counter() - t_start) * 1000
                cost_tracker.record_call(
                    action_name=f"{action_name}_cache_hit",
                    model=model_name,
                    prompt_tokens=0,
                    completion_tokens=0,
                    cost_usd=0.0,
                    latency_ms=latency_ms,
                    stream=False,
                    status="success"
                )
                logger.info(
                    "[%s] Semantic Cache HIT (sim=%.3f) in %.1fms | Model: %s | $0.00",
                    action_name,
                    cache_hit.similarity,
                    latency_ms,
                    model_name,
                )
                return cache_hit.response

        # 3. AI Pattern: Cross-Provider Fallback Model Routing
        async def _call_with_translation():
            async with handle_litellm_exceptions_async(action_name):
                if self.enable_patterns and config.FALLBACK_MODELS:
                    return await fallback_router.acompletion(**kwargs)
                return await litellm.acompletion(**kwargs)

        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(self.max_retries),
                wait=wait_exponential(multiplier=2, min=2, max=20),
                retry=retry_if_exception_type((LLMRateLimitError, LLMConnectionError, LLMTimeoutError)),
                reraise=True
            ):
                with attempt:
                    response = await _call_with_translation()

            latency_ms = (time.perf_counter() - t_start) * 1000
            metrics = extract_usage_and_cost(response)
            actual_model = getattr(response, "model", model_name) or model_name

            # 4. Post-Call Settlement & Semantic Cache Indexing
            if reservation_id:
                total_tokens = metrics["prompt_tokens"] + metrics["completion_tokens"]
                tpm_limiter.settle(reservation_id, total_tokens)

            if self.enable_patterns and temperature <= 0.3 and user_prompt:
                await semantic_cache.store(
                    prompt=user_prompt,
                    response=response,
                    namespace=namespace,
                    metadata={
                        **metrics,
                        "latency_ms": latency_ms,
                        "model": actual_model,
                    }
                )

            # Record operational cost and telemetry in centralized tracker
            cost_tracker.record_call(
                action_name=action_name,
                model=actual_model,
                prompt_tokens=metrics["prompt_tokens"],
                completion_tokens=metrics["completion_tokens"],
                cost_usd=metrics["cost"],
                latency_ms=latency_ms,
                stream=False,
                status="success"
            )

            logger.info(
                "[%s] Completed in %.1fms | Model: %s | Tokens: %d in / %d out | Cost: $%.6f",
                action_name,
                latency_ms,
                actual_model,
                metrics["prompt_tokens"],
                metrics["completion_tokens"],
                metrics["cost"]
            )
            return response

        except Exception as e:
            if reservation_id:
                tpm_limiter.release(reservation_id)
            latency_ms = (time.perf_counter() - t_start) * 1000
            cost_tracker.record_call(
                action_name=action_name,
                model=model_name,
                prompt_tokens=0,
                completion_tokens=0,
                cost_usd=0.0,
                latency_ms=latency_ms,
                stream=False,
                status="error",
                error_message=str(e)
            )
            raise e

    async def acompletion_stream(self, action_name: str, **kwargs) -> AsyncIterator[Any]:
        """Asynchronously streams chunks from an LLM call, measuring TTFT and tracking operational costs."""
        model_name = kwargs.setdefault("model", self.default_model)
        kwargs["stream"] = True
        kwargs.setdefault("stream_options", {"include_usage": True})

        t_start = time.perf_counter()
        first_token_captured = False
        ttft_ms: float | None = None
        prompt_tokens = 0
        completion_tokens = 0
        cost_usd = 0.0

        # TPM Pre-reservation for streaming
        reservation_id = None
        if self.enable_patterns:
            user_prompt, _ = _extract_prompt_text(kwargs)
            estimated_tokens = tpm_limiter.estimate_tokens(kwargs.get("messages", user_prompt))
            reservation_id = tpm_limiter.acquire(estimated_tokens)

        async def _initiate_stream():
            async with handle_litellm_exceptions_async(action_name):
                return await litellm.acompletion(**kwargs)

        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(self.max_retries),
                wait=wait_exponential(multiplier=1, min=1, max=10),
                retry=retry_if_exception_type((LLMRateLimitError, LLMConnectionError, LLMTimeoutError)),
                reraise=True
            ):
                with attempt:
                    response_stream = await _initiate_stream()
        except Exception as e:
            if reservation_id:
                tpm_limiter.release(reservation_id)
            latency_ms = (time.perf_counter() - t_start) * 1000
            cost_tracker.record_call(
                action_name=action_name,
                model=model_name,
                prompt_tokens=0,
                completion_tokens=0,
                cost_usd=0.0,
                latency_ms=latency_ms,
                stream=True,
                status="error",
                error_message=str(e)
            )
            raise e

        try:
            async for chunk in response_stream:
                # Capture Time-to-First-Token (TTFT) on the first chunk containing actual generated content
                if not first_token_captured and hasattr(chunk, "choices") and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, "content") and delta.content:
                        first_token_captured = True
                        ttft_ms = (time.perf_counter() - t_start) * 1000
                        logger.info("[%s] TTFT (Time-to-First-Token): %.2f ms", action_name, ttft_ms)

                # Capture usage metrics emitted in final stream chunk
                if hasattr(chunk, "usage") and chunk.usage is not None:
                    usage_metrics = extract_usage_and_cost(chunk)
                    prompt_tokens = usage_metrics["prompt_tokens"]
                    completion_tokens = usage_metrics["completion_tokens"]
                    cost_usd = usage_metrics["cost"]

                # Attach live TTFT metric attribute to chunk for consumer inspection
                try:
                    setattr(chunk, "_ttft_ms", ttft_ms)
                except Exception:
                    pass

                yield chunk

        except Exception as e:
            if reservation_id:
                tpm_limiter.release(reservation_id)
            logger.error("[%s] Exception occurred mid-stream: %s", action_name, e)
            total_latency_ms = (time.perf_counter() - t_start) * 1000
            cost_tracker.record_call(
                action_name=action_name,
                model=model_name,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cost_usd=cost_usd,
                latency_ms=total_latency_ms,
                stream=True,
                ttft_ms=ttft_ms,
                status="error",
                error_message=str(e)
            )
            async with handle_litellm_exceptions_async(action_name):
                raise e

        # Stream completed successfully: settle TPM and record cumulative telemetry
        if reservation_id:
            tpm_limiter.settle(reservation_id, prompt_tokens + completion_tokens)

        total_latency_ms = (time.perf_counter() - t_start) * 1000
        cost_tracker.record_call(
            action_name=action_name,
            model=model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost_usd,
            latency_ms=total_latency_ms,
            stream=True,
            ttft_ms=ttft_ms,
            status="success"
        )
        logger.info(
            "[%s] Stream finished in %.1fms | TTFT: %.1fms | Tokens: %d in / %d out | Cost: $%.6f",
            action_name,
            total_latency_ms,
            ttft_ms or 0.0,
            prompt_tokens,
            completion_tokens,
            cost_usd
        )
