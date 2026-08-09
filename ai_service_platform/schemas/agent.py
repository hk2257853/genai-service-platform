from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from ai_service_platform.schemas.common import BaseLLMUsage


class ToolCallRecord(BaseModel):
    tool_name: str = Field(..., description="Name of the tool invoked by the agent.")
    tool_args: Dict[str, Any] = Field(default_factory=dict, description="Arguments passed to the tool.")
    tool_output: Any = Field(..., description="Output or error returned by the tool.")
    status: str = Field("success", description="Status of tool execution: 'success' or 'error'.")


class AgentRunRequest(BaseModel):
    prompt: str = Field(..., min_length=2, description="Goal or instruction for the LangGraph agent.")
    system_prompt: Optional[str] = Field(
        default=None, 
        description="Optional custom system persona/instructions for the agent."
    )
    max_iterations: int = Field(
        default=5, 
        ge=1, 
        le=10, 
        description="Infinite loop guard: max tool-calling iterations allowed."
    )


class AgentRunResponse(BaseLLMUsage):
    """Standardized response model for autonomous LangGraph agent runs."""
    final_answer: str = Field(..., description="Final synthesized response from the agent.")
    tool_calls: List[ToolCallRecord] = Field(
        default_factory=list, 
        description="Chronological record of all tool calls executed during the reasoning loop."
    )
    iterations: int = Field(..., description="Number of agent reasoning cycles executed.")
