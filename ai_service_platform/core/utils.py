from typing import Any, Dict
from litellm import completion_cost


def extract_usage_and_cost(response: Any) -> Dict[str, Any]:
    """Safely extracts token usage counts and cost in USD from any LiteLLM completion response."""
    usage = getattr(response, "usage", None)
    cost = completion_cost(completion_response=response) or 0.0
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", 0) if usage else 0,
        "completion_tokens": getattr(usage, "completion_tokens", 0) if usage else 0,
        "total_tokens": getattr(usage, "total_tokens", 0) if usage else 0,
        "cost": cost
    }
