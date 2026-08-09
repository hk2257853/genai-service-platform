from pydantic import BaseModel, Field


class BaseLLMUsage(BaseModel):
    """Shared base model for all responses that include LLM token metrics and cost."""
    prompt_tokens: int = Field(0, description="Total prompt tokens consumed.")
    completion_tokens: int = Field(0, description="Total completion tokens generated.")
    total_tokens: int = Field(0, description="Total token count consumed.")
    cost: float = Field(0.0, description="Estimated total cost in USD.")
