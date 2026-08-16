import logging
from typing import Optional
from fastapi import HTTPException

from ai_service_platform.agents.workflow_graph import MultiStepWorkflowAgent
from ai_service_platform.schemas.workflow import (
    WorkflowRunResponse,
    WorkflowStateResponse,
    ApprovalInterruptPayload,
    StepExecutionRecord
)

logger = logging.getLogger("workflow_service")


class WorkflowService:
    """Service layer orchestrating multi-step LangGraph workflows, HITL pause/resume, and state introspection."""

    def __init__(self, agent: Optional[MultiStepWorkflowAgent] = None):
        self.agent = agent or MultiStepWorkflowAgent()

    def _format_execution_steps(self, steps_data: list) -> list[StepExecutionRecord]:
        return [
            StepExecutionRecord(
                step_name=step["step_name"],
                status=step["status"],
                details=step.get("details", {}),
                timestamp=step["timestamp"]
            )
            for step in steps_data
        ]

    async def start_workflow(
        self, 
        prompt: str, 
        thread_id: Optional[str] = None, 
        customer_id: str = "CUST-DEFAULT"
    ) -> WorkflowRunResponse:
        """Starts a workflow graph execution. Returns immediately if completed or paused with PENDING_APPROVAL."""
        logger.info("Starting workflow for prompt: '%s' | customer_id: %s", prompt, customer_id)
        
        final_state, is_interrupted, interrupt_payload = await self.agent.astart_workflow(
            prompt=prompt,
            thread_id=thread_id,
            customer_id=customer_id
        )
        
        tid = final_state.get("thread_id", thread_id or "wf-unknown")
        steps = self._format_execution_steps(final_state.get("execution_steps", []))
        acc_tokens = final_state.get("accumulated_tokens", {})
        cost = final_state.get("accumulated_cost", 0.0)
        
        if is_interrupted:
            logger.info("[Thread %s] Workflow PAUSED. Awaiting human approval.", tid)
            pending_obj = None
            if interrupt_payload and isinstance(interrupt_payload, dict):
                pending_obj = ApprovalInterruptPayload(
                    action_type=interrupt_payload.get("action_type", "action"),
                    order_id=interrupt_payload.get("order_id"),
                    amount=interrupt_payload.get("amount"),
                    risk_level=interrupt_payload.get("risk_level", "high"),
                    justification=interrupt_payload.get("justification", "Manager review required."),
                    required_role=interrupt_payload.get("required_role", "manager")
                )
                
            return WorkflowRunResponse(
                thread_id=tid,
                status="PENDING_APPROVAL",
                final_answer="Your request involves a sensitive operation and is pending supervisor approval.",
                pending_approval=pending_obj,
                execution_steps=steps,
                action_result=None,
                prompt_tokens=acc_tokens.get("prompt_tokens", 0),
                completion_tokens=acc_tokens.get("completion_tokens", 0),
                total_tokens=acc_tokens.get("total_tokens", 0),
                cost=cost
            )
            
        logger.info("[Thread %s] Workflow COMPLETED without interruption.", tid)
        return WorkflowRunResponse(
            thread_id=tid,
            status="COMPLETED",
            final_answer=final_state.get("final_answer"),
            pending_approval=None,
            execution_steps=steps,
            action_result=final_state.get("action_result"),
            prompt_tokens=acc_tokens.get("prompt_tokens", 0),
            completion_tokens=acc_tokens.get("completion_tokens", 0),
            total_tokens=acc_tokens.get("total_tokens", 0),
            cost=cost
        )

    async def resume_workflow(
        self, 
        thread_id: str, 
        approved: bool, 
        reviewer: str, 
        reviewer_role: str = "manager", 
        feedback: Optional[str] = None
    ) -> WorkflowRunResponse:
        """Resumes a paused workflow thread with human approval decision."""
        logger.info("[Thread %s] Resuming workflow with decision approved=%s by reviewer=%s (role=%s)",
                    thread_id, approved, reviewer, reviewer_role)
        try:
            final_state, is_still_interrupted = await self.agent.aresume_workflow(
                thread_id=thread_id,
                approved=approved,
                reviewer=reviewer,
                reviewer_role=reviewer_role,
                feedback=feedback
            )
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
            
        steps = self._format_execution_steps(final_state.get("execution_steps", []))
        acc_tokens = final_state.get("accumulated_tokens", {})
        cost = final_state.get("accumulated_cost", 0.0)
        
        status = "PENDING_APPROVAL" if is_still_interrupted else "COMPLETED"
        
        return WorkflowRunResponse(
            thread_id=thread_id,
            status=status,
            final_answer=final_state.get("final_answer"),
            pending_approval=None,
            execution_steps=steps,
            action_result=final_state.get("action_result"),
            prompt_tokens=acc_tokens.get("prompt_tokens", 0),
            completion_tokens=acc_tokens.get("completion_tokens", 0),
            total_tokens=acc_tokens.get("total_tokens", 0),
            cost=cost
        )

    async def get_workflow_state(self, thread_id: str) -> WorkflowStateResponse:
        """Inspects current checkpoint snapshot for thread_id."""
        state_data = await self.agent.aget_workflow_state(thread_id)
        if state_data is None:
            raise HTTPException(status_code=404, detail=f"Thread '{thread_id}' not found in checkpointer.")
            
        return WorkflowStateResponse(
            thread_id=state_data["thread_id"],
            status=state_data["status"],
            next_nodes=state_data["next_nodes"],
            pending_interrupt=state_data["pending_interrupt"],
            values=state_data["values"]
        )

    def get_topology_mermaid(self) -> str:
        """Returns the native Mermaid diagram syntax representing the static compiled graph topology."""
        return self.agent.get_graph_mermaid()
