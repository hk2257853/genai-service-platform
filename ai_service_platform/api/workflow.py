from fastapi import APIRouter, Depends

from ai_service_platform.schemas.workflow import (
    WorkflowStartRequest,
    WorkflowResumeRequest,
    WorkflowRunResponse,
    WorkflowStateResponse,
    WorkflowTopologyResponse
)
from ai_service_platform.services.workflow import WorkflowService

router = APIRouter(tags=["LangGraph Workflow & HITL"])

_workflow_service_instance: WorkflowService | None = None


def get_workflow_service() -> WorkflowService:
    global _workflow_service_instance
    if _workflow_service_instance is None:
        _workflow_service_instance = WorkflowService()
    return _workflow_service_instance


@router.post("/workflow/start", response_model=WorkflowRunResponse)
async def start_workflow_endpoint(
    request: WorkflowStartRequest,
    service: WorkflowService = Depends(get_workflow_service)
):
    """Executes a multi-step stateful workflow with policy guardrails, conditional routing, and HITL interrupt on high-risk operations."""
    return await service.start_workflow(
        prompt=request.prompt,
        thread_id=request.thread_id,
        customer_id=request.customer_id
    )


@router.post("/workflow/resume", response_model=WorkflowRunResponse)
async def resume_workflow_endpoint(
    request: WorkflowResumeRequest,
    service: WorkflowService = Depends(get_workflow_service)
):
    """Resumes an interrupted/paused workflow thread with a human reviewer decision and RBAC authorization."""
    return await service.resume_workflow(
        thread_id=request.thread_id,
        approved=request.approved,
        reviewer=request.reviewer,
        reviewer_role=request.reviewer_role,
        feedback=request.feedback
    )


@router.get("/workflow/topology", response_model=WorkflowTopologyResponse)
async def get_workflow_topology_endpoint(
    service: WorkflowService = Depends(get_workflow_service)
):
    """Exports the static nodes, edges, and conditional branch topology of the compiled LangGraph state machine in native Mermaid syntax."""
    mermaid_code = service.get_topology_mermaid()
    return WorkflowTopologyResponse(mermaid=mermaid_code)


@router.get("/workflow/state/{thread_id}", response_model=WorkflowStateResponse)
async def get_workflow_state_endpoint(
    thread_id: str,
    service: WorkflowService = Depends(get_workflow_service)
):
    """Inspects the checkpointed StateSnapshot of an active or paused workflow thread."""
    return await service.get_workflow_state(thread_id=thread_id)
