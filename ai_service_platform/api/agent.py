from fastapi import APIRouter, Depends

from ai_service_platform.schemas.agent import AgentRunRequest, AgentRunResponse
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
    """Executes the autonomous LangGraph state machine agent with tool execution and multi-turn loops."""
    try:
        return await service.run_agent(
            prompt=request.prompt,
            system_prompt=request.system_prompt,
            max_iterations=request.max_iterations
        )
    except Exception as e:
        raise e
