import logging
from typing import Optional

from ai_service_platform.agents.langgraph_agent import LangGraphAgent
from ai_service_platform.schemas.agent import AgentRunResponse, ToolCallRecord

logger = logging.getLogger("agent_service")


class AgentService:
    """Service layer orchestrating the LangGraph autonomous agent execution and response formatting."""

    def __init__(self, agent: Optional[LangGraphAgent] = None):
        self.agent = agent or LangGraphAgent()

    async def run_agent(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None, 
        max_iterations: int = 5
    ) -> AgentRunResponse:
        """Executes the agent state machine, traces tool execution records, and calculates cumulative metrics."""
        logger.info("Executing AgentService for goal: '%s' with max_iterations=%d", prompt, max_iterations)
        
        final_state = await self.agent.ainvoke(
            prompt=prompt,
            system_prompt=system_prompt,
            max_iterations=max_iterations
        )

        # Extract last assistant message content
        last_message = final_state["messages"][-1]
        final_answer = last_message.get("content", "") if isinstance(last_message, dict) else getattr(last_message, "content", "")

        # Format tool calls history
        tool_records = [
            ToolCallRecord(
                tool_name=tc["tool_name"],
                tool_args=tc["tool_args"],
                tool_output=tc["tool_output"],
                status=tc.get("status", "success")
            )
            for tc in final_state.get("tool_calls", [])
        ]

        acc_tokens = final_state.get("accumulated_tokens", {})
        
        return AgentRunResponse(
            final_answer=final_answer,
            tool_calls=tool_records,
            iterations=final_state.get("iteration_count", 1),
            prompt_tokens=acc_tokens.get("prompt_tokens", 0),
            completion_tokens=acc_tokens.get("completion_tokens", 0),
            total_tokens=acc_tokens.get("total_tokens", 0),
            cost=final_state.get("accumulated_cost", 0.0)
        )
