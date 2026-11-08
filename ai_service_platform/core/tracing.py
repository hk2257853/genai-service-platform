"""LangSmith AI Observability & Hierarchical Run-Tree Tracing Configuration.

This module initializes enterprise tracing across:
1. LangChain / LangGraph state machines and runnable execution trees.
2. LiteLLM proxy & client completion requests via native LangSmith callbacks.

When enabled via environment variables (`LANGCHAIN_TRACING_V2=true` and `LANGCHAIN_API_KEY`),
every LLM generation, tool call, and agent cycle automatically emits hierarchical spans
(parent-child DAG trees) directly to LangSmith.
"""

import os
import logging
from typing import Any, Dict, List, Optional
import litellm

from ai_service_platform import config

logger = logging.getLogger("tracing")


def setup_tracing() -> bool:
    """Configures LangSmith environment variables and registers LiteLLM callbacks.

    Returns:
        bool: True if LangSmith tracing is actively configured, False if running in local mode.
    """
    if not config.LANGCHAIN_TRACING_V2:
        logger.info("LangSmith tracing is disabled (LANGCHAIN_TRACING_V2=false). Spans will not be exported.")
        return False

    if not config.LANGCHAIN_API_KEY:
        logger.warning(
            "LANGCHAIN_TRACING_V2 is true, but LANGCHAIN_API_KEY is not set. "
            "LangSmith tracing cannot export spans without a valid API key."
        )
        return False

    # Propagate configurations to os.environ for LangChain/LangGraph and LiteLLM subprocesses
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = config.LANGCHAIN_API_KEY
    os.environ["LANGCHAIN_PROJECT"] = config.LANGCHAIN_PROJECT
    os.environ["LANGCHAIN_ENDPOINT"] = config.LANGCHAIN_ENDPOINT

    # Register LangSmith callback in LiteLLM's global dispatcher
    if not hasattr(litellm, "success_callback") or litellm.success_callback is None:
        litellm.success_callback = []
    if not hasattr(litellm, "failure_callback") or litellm.failure_callback is None:
        litellm.failure_callback = []

    if "langsmith" not in litellm.success_callback:
        litellm.success_callback.append("langsmith")
    if "langsmith" not in litellm.failure_callback:
        litellm.failure_callback.append("langsmith")

    logger.info(
        "LangSmith tracing successfully enabled! Project: '%s', Endpoint: '%s'. LiteLLM callbacks attached.",
        config.LANGCHAIN_PROJECT,
        config.LANGCHAIN_ENDPOINT
    )
    return True


def is_tracing_enabled() -> bool:
    """Checks whether LangSmith tracing is actively enabled and configured."""
    return bool(config.LANGCHAIN_TRACING_V2 and config.LANGCHAIN_API_KEY)


def get_trace_run_config(
    run_name: str,
    thread_id: Optional[str] = None,
    tags: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Generates standard LangGraph / LangChain execution config containing run metadata.

    Args:
        run_name: Human-readable identifier for the root or sub-span in the LangSmith run tree.
        thread_id: Optional conversation/workflow thread ID for session grouping.
        tags: Optional categorization tags (e.g. ['agent', 'mcp', 'production']).
        metadata: Custom key-value attributes (e.g. customer_id, iteration).

    Returns:
        Dict suitable for passing as `config` to `graph.ainvoke(..., config=config)`
    """
    merged_metadata: Dict[str, Any] = {"service": "ai-service-platform"}
    if metadata:
        merged_metadata.update(metadata)
    if thread_id:
        merged_metadata["thread_id"] = thread_id

    trace_config: Dict[str, Any] = {
        "run_name": run_name,
        "tags": tags or ["ai-platform"],
        "metadata": merged_metadata
    }

    if thread_id:
        trace_config["configurable"] = {"thread_id": thread_id}

    return trace_config
