from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field

from ai_service_platform.schemas.common import BaseLLMUsage


class WorkflowStartRequest(BaseModel):
    prompt: str = Field(..., min_length=2, description="Customer inquiry or action request.")
    thread_id: Optional[str] = Field(default=None, description="Optional unique thread ID for tracking state. Auto-generated if not provided.")
    customer_id: Optional[str] = Field(default="CUST-DEFAULT", description="Identifier of the customer making the request.")


class WorkflowResumeRequest(BaseModel):
    thread_id: str = Field(..., min_length=1, description="Thread ID of the paused workflow.")
    approved: bool = Field(..., description="Decision from the reviewer: True to approve, False to reject.")
    reviewer: str = Field(..., min_length=1, description="Name or ID of the human reviewer.")
    reviewer_role: str = Field(default="manager", description="Role of the reviewer (e.g. 'manager', 'admin', 'support_agent').")
    feedback: Optional[str] = Field(default=None, description="Optional notes or rejection rationale from the reviewer.")


class ApprovalInterruptPayload(BaseModel):
    action_type: str = Field(..., description="Sensitive action requiring human approval (e.g. 'refund', 'cancel_order').")
    order_id: Optional[str] = Field(default=None, description="Target order ID for the action.")
    amount: Optional[float] = Field(default=None, description="Monetary value involved in the action.")
    risk_level: Literal["low", "high"] = Field(default="high", description="Assessed risk level.")
    justification: str = Field(..., description="Explanation of why human review is required.")
    required_role: str = Field(default="manager", description="Minimum role authorized to approve.")


class StepExecutionRecord(BaseModel):
    step_name: str = Field(..., description="Identifier of the workflow graph node.")
    status: str = Field(..., description="Execution status: 'completed', 'interrupted', 'skipped', 'rejected', 'failed'.")
    details: Dict[str, Any] = Field(default_factory=dict, description="Metadata or outputs emitted by this step.")
    timestamp: str = Field(..., description="ISO 8601 UTC timestamp when this step executed.")


class WorkflowRunResponse(BaseLLMUsage):
    """Standard response model for multi-step LangGraph workflows."""
    thread_id: str = Field(..., description="Thread identifier for this workflow execution.")
    status: Literal["COMPLETED", "PENDING_APPROVAL"] = Field(..., description="Execution status.")
    final_answer: Optional[str] = Field(default=None, description="Final synthesized response to the customer/caller.")
    pending_approval: Optional[ApprovalInterruptPayload] = Field(default=None, description="Details of pending human approval if status is PENDING_APPROVAL.")
    execution_steps: List[StepExecutionRecord] = Field(default_factory=list, description="Chronological record of graph nodes executed.")
    action_result: Optional[Dict[str, Any]] = Field(default=None, description="Result metadata if a mutation action was executed.")


class WorkflowStateResponse(BaseModel):
    """Inspection response showing the snapshot of a workflow thread in the checkpointer."""
    thread_id: str = Field(..., description="Thread ID.")
    status: Literal["ACTIVE", "INTERRUPTED", "COMPLETED"] = Field(..., description="Current checkpoint status.")
    next_nodes: List[str] = Field(default_factory=list, description="Next pending nodes to execute in the graph.")
    pending_interrupt: Optional[Dict[str, Any]] = Field(default=None, description="Pending interrupt payload if paused.")
    values: Dict[str, Any] = Field(default_factory=dict, description="State values stored in the checkpoint.")


class WorkflowTopologyResponse(BaseModel):
    """Raw Mermaid diagram representing the static compiled graph topology."""
    mermaid: str = Field(..., description="Native Mermaid diagram syntax of the static compiled graph topology.")
