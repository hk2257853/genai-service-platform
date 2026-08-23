import logging
from typing import Optional, List, Dict, Any

from ai_service_platform.agents.langgraph_agent import LangGraphAgent
from ai_service_platform.schemas.agent import (
    AgentRunResponse, 
    ToolCallRecord, 
    ThreadHistoryResponse, 
    ThreadMessageRecord, 
    ThreadClearResponse
)

logger = logging.getLogger("agent_service")


class AgentService:
    """Service layer orchestrating the LangGraph autonomous agent execution, memory management, and response formatting."""

    def __init__(self, agent: Optional[LangGraphAgent] = None):
        self.agent = agent or LangGraphAgent()

    async def run_agent(
        self, 
        prompt: str, 
        thread_id: Optional[str] = None,
        system_prompt: Optional[str] = None, 
        max_iterations: int = 5
    ) -> AgentRunResponse:
        """Executes the agent state machine with short-term memory, traces tool execution records, and tracks cumulative metrics."""
        logger.info("Executing AgentService for goal: '%s' (thread_id: %s, max_iterations: %d)", prompt, thread_id, max_iterations)
        
        result = await self.agent.ainvoke(
            prompt=prompt,
            thread_id=thread_id,
            system_prompt=system_prompt,
            max_iterations=max_iterations
        )

        final_state = result["state"]
        active_thread_id = result["thread_id"]

        # Extract last assistant message content
        all_messages = final_state.get("messages", [])
        last_message = all_messages[-1] if all_messages else None
        
        if last_message is not None:
            final_answer = last_message.get("content", "") if isinstance(last_message, dict) else getattr(last_message, "content", "")
        else:
            final_answer = ""

        # Format tool calls history for this turn
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
            thread_id=active_thread_id,
            final_answer=final_answer,
            tool_calls=tool_records,
            iterations=final_state.get("iteration_count", 1),
            messages_count=len(all_messages),
            prompt_tokens=acc_tokens.get("prompt_tokens", 0),
            completion_tokens=acc_tokens.get("completion_tokens", 0),
            total_tokens=acc_tokens.get("total_tokens", 0),
            cost=final_state.get("accumulated_cost", 0.0)
        )

    async def get_thread_history(self, thread_id: str) -> Optional[ThreadHistoryResponse]:
        """Retrieves chronological message history and accumulated metrics for an active thread."""
        logger.info("Fetching thread history for thread_id: '%s'", thread_id)
        state_dict = await self.agent.aget_thread_history(thread_id)
        if not state_dict or "messages" not in state_dict:
            return None

        records: List[ThreadMessageRecord] = []
        for msg in state_dict.get("messages", []):
            if isinstance(msg, dict):
                role = msg.get("role", "user")
                content = msg.get("content", "")
                t_calls = msg.get("tool_calls")
                tc_id = msg.get("tool_call_id")
                t_name = msg.get("name")
            else:
                msg_type = getattr(msg, "type", "user")
                if msg_type == "human":
                    role = "human"
                elif msg_type == "ai":
                    role = "ai"
                elif msg_type == "system":
                    role = "system"
                elif msg_type == "tool":
                    role = "tool"
                else:
                    role = msg_type

                content = getattr(msg, "content", "") or ""
                t_calls = getattr(msg, "tool_calls", None)
                tc_id = getattr(msg, "tool_call_id", None)
                t_name = getattr(msg, "name", None)

            records.append(
                ThreadMessageRecord(
                    role=role,
                    content=content,
                    tool_calls=t_calls,
                    tool_call_id=tc_id,
                    name=t_name
                )
            )

        acc_tokens = state_dict.get("accumulated_tokens", {})
        acc_cost = state_dict.get("accumulated_cost", 0.0)

        return ThreadHistoryResponse(
            thread_id=thread_id,
            messages=records,
            total_messages=len(records),
            accumulated_tokens=acc_tokens,
            accumulated_cost=acc_cost
        )

    async def clear_thread(self, thread_id: str) -> ThreadClearResponse:
        """Resets/clears short-term state memory for a given conversation thread."""
        logger.info("Clearing memory checkpoint for thread_id: '%s'", thread_id)
        cleared = await self.agent.aclear_thread(thread_id)
        return ThreadClearResponse(
            thread_id=thread_id,
            cleared=cleared,
            message=f"Memory state checkpoint for thread '{thread_id}' cleared successfully."
        )
