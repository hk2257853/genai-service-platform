import logging
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

from ai_service_platform.core.exceptions import (
    LLMAuthenticationError,
    LLMRateLimitError,
    LLMConnectionError,
    LLMTimeoutError,
    LLMAPIError,
    LLMException
)

logger = logging.getLogger("llm_client")

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
    """Production-grade async client wrapping LiteLLM with robust retries and error mapping."""

    def __init__(self, default_model: str, max_retries: int = 5):
        self.default_model = default_model
        self.max_retries = max_retries

    async def acompletion(self, action_name: str, **kwargs) -> Any:
        """Executes an async completion request wrapped in translation and tenacity retries."""
        kwargs.setdefault("model", self.default_model)

        async def _call_with_translation():
            async with handle_litellm_exceptions_async(action_name):
                return await litellm.acompletion(**kwargs)

        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(self.max_retries),
                wait=wait_exponential(multiplier=2, min=2, max=20),
                retry=retry_if_exception_type((LLMRateLimitError, LLMConnectionError, LLMTimeoutError)),
                reraise=True
            ):
                with attempt:
                    return await _call_with_translation()
        except Exception as e:
            # Tenacity will raise the mapped exception because reraise=True
            raise e

    async def acompletion_stream(self, action_name: str, **kwargs) -> AsyncIterator[Any]:
        """Asynchronously streams chunks from an LLM call, with initial connection retries."""
        kwargs.setdefault("model", self.default_model)
        kwargs["stream"] = True

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
            raise e

        try:
            async for chunk in response_stream:
                yield chunk
        except Exception as e:
            # Map exception if it happens mid-stream
            logger.error("[%s] Exception occurred mid-stream: %s", action_name, e)
            async with handle_litellm_exceptions_async(action_name):
                raise e
