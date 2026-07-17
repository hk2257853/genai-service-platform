import logging
from contextlib import contextmanager
from typing import Iterator, Any
from litellm import completion
from litellm.exceptions import (
    AuthenticationError,
    RateLimitError,
    APIConnectionError,
    ServiceUnavailableError,
    APIError
)
import config

# Set up logging configuration to match production standards
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("api_utils")

@contextmanager
def handle_litellm_exceptions(action_name: str):
    """Context manager to unify LiteLLM exception handling and ensure bare raises."""
    try:
        yield
    except AuthenticationError as e:
        logger.critical("[%s] Authentication failed. Check your API key. Details: %s", action_name, e)
        raise
    except RateLimitError as e:
        logger.error("[%s] Rate limit exceeded. Details: %s", action_name, e)
        raise
    except (APIConnectionError, ServiceUnavailableError) as e:
        logger.error("[%s] Network connection or service unavailable. Details: %s", action_name, e)
        raise
    except APIError as e:
        logger.critical("[%s] API error occurred: %s", action_name, e)
        raise

def execute_completion(action_name: str, **kwargs) -> Any:
    """Executes a litellm completion request wrapped in unified exception handling with automatic retries."""
    kwargs.setdefault("num_retries", config.DEFAULT_RETRIES)
    with handle_litellm_exceptions(action_name):
        return completion(**kwargs)

def execute_completion_stream(action_name: str, **kwargs) -> Iterator[Any]:
    """Yields chunks from a streaming request, capturing mid-stream loop exceptions with automatic retries."""
    kwargs.setdefault("num_retries", config.DEFAULT_RETRIES)
    with handle_litellm_exceptions(action_name):
        response_stream = completion(**kwargs, stream=True)
        for chunk in response_stream:
            yield chunk

