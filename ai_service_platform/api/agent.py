from fastapi import APIRouter, Depends, HTTPException

from ai_service_platform.schemas.agent import (
    AgentRunRequest, 
    AgentRunResponse, 
    ThreadHistoryResponse, 
    ThreadClearResponse
)
from ai_service_platform.services.agent import AgentService

router = APIRouter(tags=["LangGraph Agent"])

_agent_service_instance: AgentService | None = None


def get_agent_service() -> AgentService:
    global _agent_service_instance
    if _agent_service_instance is None:
        _agent_service_instance = AgentService()
    return _agent_service_instance


@router.post("/agent/run", response_model=AgentRunResponse)
async def run_agent_endpoint(
    request: AgentRunRequest,
    service: AgentService = Depends(get_agent_service)
):
    """Executes the autonomous LangGraph state machine agent with tool execution and short-term conversation memory."""
    try:
        return await service.run_agent(
            prompt=request.prompt,
            thread_id=request.thread_id,
            system_prompt=request.system_prompt,
            max_iterations=request.max_iterations
        )
    except Exception as e:
        raise e


@router.get("/agent/threads/{thread_id}", response_model=ThreadHistoryResponse)
async def get_thread_history_endpoint(
    thread_id: str,
    service: AgentService = Depends(get_agent_service)
):
    """Retrieves chronological conversational message history and cumulative token/cost metrics for a thread."""
    history = await service.get_thread_history(thread_id)
    if history is None:
        raise HTTPException(status_code=404, detail=f"Thread '{thread_id}' was not found or has no active state.")
    return history


@router.delete("/agent/threads/{thread_id}", response_model=ThreadClearResponse)
async def clear_thread_endpoint(
    thread_id: str,
    service: AgentService = Depends(get_agent_service)
):
    """Resets/deletes the short-term conversation memory checkpoint for a given thread."""
    return await service.clear_thread(thread_id)
