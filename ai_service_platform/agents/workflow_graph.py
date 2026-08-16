import json
import logging
import uuid
from datetime import datetime, timezone
from typing import TypedDict, Annotated, List, Dict, Any, Optional, Literal

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from ai_service_platform import config
from ai_service_platform.core.client import AsyncLLMClient
from ai_service_platform.core.utils import extract_usage_and_cost
from ai_service_platform.agents.langgraph_agent import serialize_messages_for_llm

logger = logging.getLogger("workflow_graph")


class WorkflowState(TypedDict):
    messages: Annotated[list, add_messages]
    prompt: str
    thread_id: str
    customer_id: str
    intent: Literal["query", "action", "unsupported"]
    action_type: Optional[str]
    extracted_entities: Dict[str, Any]
    risk_level: Literal["low", "high"]
    requires_human_approval: bool
    policy_verdict: Literal["approved", "needs_approval", "rejected"]
    policy_reason: Optional[str]
    approval_status: Optional[Literal["approved", "rejected", "auto_approved"]]
    reviewer: Optional[str]
    reviewer_role: Optional[str]
    reviewer_notes: Optional[str]
    action_result: Optional[Dict[str, Any]]
    final_answer: Optional[str]
    execution_steps: List[Dict[str, Any]]
    accumulated_tokens: Dict[str, int]
    accumulated_cost: float


CLASSIFICATION_TOOL_SPEC = [
    {
        "type": "function",
        "function": {
            "name": "classify_order_request",
            "description": "Parses and classifies customer request into intent, action type, and extracted entities.",
            "parameters": {
                "type": "object",
                "properties": {
                    "intent": {
                        "type": "string",
                        "enum": ["query", "action", "unsupported"],
                        "description": "Category of request: 'query' for read-only questions, 'action' for state mutations (refunds, cancellations, address changes), or 'unsupported'."
                    },
                    "action_type": {
                        "type": "string",
                        "enum": ["refund", "cancel_order", "change_address", "general_inquiry", "disallowed"],
                        "description": "Specific action type requested."
                    },
                    "order_id": {
                        "type": "string",
                        "description": "Order identifier mentioned in request (e.g. ORD-1001), if applicable."
                    },
                    "amount": {
                        "type": "number",
                        "description": "Monetary dollar amount mentioned in request (e.g. 25.0, 150.0), if applicable."
                    },
                    "reason": {
                        "type": "string",
                        "description": "Customer provided rationale or context."
                    }
                },
                "required": ["intent", "action_type"]
            }
        }
    }
]


class MultiStepWorkflowAgent:
    """Production Multi-Step LangGraph Agent featuring deterministic policy evaluation,

    conditional branch routing, state checkpointing, and Human-in-the-Loop (HITL) interrupt/resumption.
    """

    def __init__(self, llm_client: Optional[AsyncLLMClient] = None, checkpointer: Optional[MemorySaver] = None):
        self.llm_client = llm_client or AsyncLLMClient(default_model=config.MODEL_ID)
        self.checkpointer = checkpointer or MemorySaver()
        self.graph = self._build_graph()

    def _record_step(self, state: WorkflowState, step_name: str, status: str, details: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        steps = list(state.get("execution_steps", []))
        steps.append({
            "step_name": step_name,
            "status": status,
            "details": details or {},
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        return steps

    def _update_tokens_and_cost(self, state: WorkflowState, metrics: Dict[str, Any]) -> tuple[Dict[str, int], float]:
        acc_tokens = state.get("accumulated_tokens", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}).copy()
        acc_tokens["prompt_tokens"] += metrics.get("prompt_tokens", 0)
        acc_tokens["completion_tokens"] += metrics.get("completion_tokens", 0)
        acc_tokens["total_tokens"] += metrics.get("total_tokens", 0)
        new_cost = state.get("accumulated_cost", 0.0) + metrics.get("cost", 0.0)
        return acc_tokens, new_cost

    async def _classify_intent_node(self, state: WorkflowState) -> Dict[str, Any]:
        """Stage 1: Analyzes user prompt using LLM function calling to extract intent and entities."""
        logger.info("[Thread %s] Step 1: Classifying Intent & Extracting Entities...", state["thread_id"])
        
        system_prompt = (
            "You are an intent classification and entity extraction engine for an e-commerce platform. "
            "Analyze the customer prompt and extract intent ('query', 'action', or 'unsupported'), "
            "action_type ('refund', 'cancel_order', 'change_address', 'general_inquiry', or 'disallowed'), "
            "and any order_id, amount, or reason provided."
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": state["prompt"]}
        ]
        
        response = await self.llm_client.acompletion(
            action_name="classify_intent",
            messages=messages,
            tools=CLASSIFICATION_TOOL_SPEC,
            tool_choice={"type": "function", "function": {"name": "classify_order_request"}},
            temperature=0.0
        )
        
        metrics = extract_usage_and_cost(response)
        acc_tokens, new_cost = self._update_tokens_and_cost(state, metrics)
        
        msg = response.choices[0].message
        intent = "query"
        action_type = "general_inquiry"
        entities = {}
        
        if msg.tool_calls and len(msg.tool_calls) > 0:
            raw_args = msg.tool_calls[0].function.arguments
            try:
                parsed_args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                intent = parsed_args.get("intent", "query")
                action_type = parsed_args.get("action_type", "general_inquiry")
                entities = {
                    "order_id": parsed_args.get("order_id"),
                    "amount": parsed_args.get("amount"),
                    "reason": parsed_args.get("reason")
                }
            except Exception as e:
                logger.warning("Failed to parse tool call arguments: %s", e)
                
        details = {
            "intent": intent,
            "action_type": action_type,
            "entities": entities
        }
        steps = self._record_step(state, "classify_intent", "completed", details)
        
        return {
            "intent": intent,
            "action_type": action_type,
            "extracted_entities": entities,
            "execution_steps": steps,
            "accumulated_tokens": acc_tokens,
            "accumulated_cost": new_cost
        }

    async def _evaluate_policy_node(self, state: WorkflowState) -> Dict[str, Any]:
        """Stage 2: Deterministic policy evaluation enforcing compliance, risk tiering, and approval thresholds."""
        logger.info("[Thread %s] Step 2: Evaluating Enterprise Policy Guardrails...", state["thread_id"])
        
        intent = state.get("intent", "query")
        action_type = state.get("action_type", "general_inquiry")
        entities = state.get("extracted_entities", {})
        
        # Policy Rules:
        # 1. Unsupported or malicious actions -> Rejected
        if intent == "unsupported" or action_type == "disallowed":
            verdict = "rejected"
            risk_level = "high"
            needs_approval = False
            reason = "Requested operation is unsupported or prohibited by security guardrails."
            approval_status = "rejected"
            
        # 2. Informational queries -> Low Risk, Auto-Approved
        elif intent == "query":
            verdict = "approved"
            risk_level = "low"
            needs_approval = False
            reason = "Standard informational inquiry auto-approved for resolution."
            approval_status = "auto_approved"
            
        # 3. Actions (Mutations)
        elif intent == "action":
            if action_type == "refund":
                amount = float(entities.get("amount") or 0.0)
                if amount <= 50.0:
                    verdict = "approved"
                    risk_level = "low"
                    needs_approval = False
                    reason = f"Refund amount (${amount:.2f}) is within the <= $50 auto-approval threshold."
                    approval_status = "auto_approved"
                else:
                    verdict = "needs_approval"
                    risk_level = "high"
                    needs_approval = True
                    reason = f"Refund amount (${amount:.2f}) exceeds the $50 threshold. Manager review required."
                    approval_status = None
            elif action_type in ["cancel_order", "change_address"]:
                verdict = "needs_approval"
                risk_level = "high"
                needs_approval = True
                reason = f"Action '{action_type}' modifies core order state. Requires manager review."
                approval_status = None
            else:
                verdict = "rejected"
                risk_level = "high"
                needs_approval = False
                reason = f"Unrecognized action type '{action_type}'."
                approval_status = "rejected"
        else:
            verdict = "rejected"
            risk_level = "high"
            needs_approval = False
            reason = "Invalid intent classification."
            approval_status = "rejected"

        details = {
            "policy_verdict": verdict,
            "risk_level": risk_level,
            "requires_human_approval": needs_approval,
            "policy_reason": reason
        }
        steps = self._record_step(state, "evaluate_policy", "completed", details)

        return {
            "policy_verdict": verdict,
            "risk_level": risk_level,
            "requires_human_approval": needs_approval,
            "policy_reason": reason,
            "approval_status": approval_status,
            "execution_steps": steps
        }

    async def _query_resolver_node(self, state: WorkflowState) -> Dict[str, Any]:
        """Stage 3a: Resolves informational queries cleanly using LLM synthesis without heavy RAG overhead."""
        logger.info("[Thread %s] Step 3a: Resolving Informational Query...", state["thread_id"])
        
        system_prompt = (
            "You are a helpful customer support agent for GenAI Service Platform Store. "
            "Answer customer inquiries politely and concisely. "
            "Known policies: "
            "- Return window: 30 days from delivery for unopened items. "
            "- Standard shipping: 3-5 business days; Express shipping: 1-2 business days. "
            "- Support hours: 24/7 via chat and email."
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": state["prompt"]}
        ]
        
        response = await self.llm_client.acompletion(
            action_name="query_resolver",
            messages=messages,
            temperature=0.1
        )
        
        metrics = extract_usage_and_cost(response)
        acc_tokens, new_cost = self._update_tokens_and_cost(state, metrics)
        
        answer = response.choices[0].message.content or "Thank you for your inquiry. Please let us know if you need further assistance."
        steps = self._record_step(state, "query_resolver", "completed", {"answer_preview": answer[:80]})
        
        return {
            "final_answer": answer,
            "execution_steps": steps,
            "accumulated_tokens": acc_tokens,
            "accumulated_cost": new_cost
        }

    async def _human_review_node(self, state: WorkflowState) -> Dict[str, Any]:
        """Stage 3c: Pauses execution via LangGraph interrupt() and yields approval payload to human supervisor."""
        logger.info("[Thread %s] Step 3c: Pausing execution for Human-in-the-Loop review...", state["thread_id"])
        
        interrupt_payload = {
            "action_type": state.get("action_type", "action"),
            "order_id": state.get("extracted_entities", {}).get("order_id"),
            "amount": state.get("extracted_entities", {}).get("amount"),
            "risk_level": "high",
            "justification": state.get("policy_reason", "Action requires supervisor approval."),
            "required_role": "manager"
        }
        
        # Record step as interrupted before yielding
        steps = self._record_step(state, "human_review", "interrupted", interrupt_payload)
        
        # Freeze graph execution and persist state snapshot in checkpointer
        decision = interrupt(interrupt_payload)
        
        # Resumed from Command(resume=decision)
        logger.info("[Thread %s] Step 3c: Workflow resumed by human reviewer: %s", state["thread_id"], decision)
        
        reviewer = decision.get("reviewer", "system_admin")
        reviewer_role = str(decision.get("reviewer_role", "manager")).lower()
        notes = decision.get("feedback")
        is_approved = bool(decision.get("approved", False))
        
        # RBAC validation: Enforce required role ('manager' or 'admin')
        if is_approved and reviewer_role not in ["manager", "admin"]:
            logger.warning("[Thread %s] Reviewer '%s' with role '%s' lacks required manager authorization.", 
                           state["thread_id"], reviewer, reviewer_role)
            is_approved = False
            status_verdict = "rejected"
            notes = f"Approval rejected: Reviewer role '{reviewer_role}' does not have required 'manager' authorization."
        else:
            status_verdict = "approved" if is_approved else "rejected"
            
        resumed_details = {
            "approved": is_approved,
            "reviewer": reviewer,
            "reviewer_role": reviewer_role,
            "notes": notes
        }
        steps = self._record_step(state, "human_review", "completed", resumed_details)
        
        return {
            "approval_status": status_verdict,
            "reviewer": reviewer,
            "reviewer_role": reviewer_role,
            "reviewer_notes": notes,
            "execution_steps": steps
        }

    async def _action_executor_node(self, state: WorkflowState) -> Dict[str, Any]:
        """Stage 3b: Executes approved business mutation and records transaction ID and audit trail."""
        logger.info("[Thread %s] Step 3b: Executing Approved Action '%s'...", state["thread_id"], state.get("action_type"))
        
        action = state.get("action_type", "action")
        entities = state.get("extracted_entities", {})
        order_id = entities.get("order_id") or "ORD-UNKNOWN"
        amount = entities.get("amount") or 0.0
        
        txn_id = f"TXN-{action.upper()}-{uuid.uuid4().hex[:8].upper()}"
        
        action_result = {
            "transaction_id": txn_id,
            "action_type": action,
            "order_id": order_id,
            "amount": amount,
            "executed_at": datetime.now(timezone.utc).isoformat(),
            "status": "SUCCESS",
            "approval_mode": state.get("approval_status", "auto_approved")
        }
        
        steps = self._record_step(state, "action_executor", "completed", action_result)
        
        return {
            "action_result": action_result,
            "execution_steps": steps
        }

    async def _rejection_handler_node(self, state: WorkflowState) -> Dict[str, Any]:
        """Stage 3d: Formats clear, policy-compliant rejection explanation for denied requests."""
        logger.info("[Thread %s] Step 3d: Formatting Request Rejection...", state["thread_id"])
        
        reason = state.get("reviewer_notes") or state.get("policy_reason") or "Request could not be approved under current operating guidelines."
        
        rejection_result = {
            "status": "REJECTED",
            "reason": reason,
            "reviewer": state.get("reviewer"),
            "policy_reason": state.get("policy_reason")
        }
        
        steps = self._record_step(state, "rejection_handler", "completed", rejection_result)
        
        return {
            "action_result": rejection_result,
            "execution_steps": steps
        }

    async def _synthesizer_node(self, state: WorkflowState) -> Dict[str, Any]:
        """Stage 4: Synthesizes final user-facing response incorporating execution audit trail."""
        logger.info("[Thread %s] Step 4: Synthesizing Final Workflow Response...", state["thread_id"])
        
        # If query_resolver already formulated the final answer, preserve it
        if state.get("final_answer"):
            steps = self._record_step(state, "synthesizer", "completed", {"mode": "query_direct"})
            return {"execution_steps": steps}
            
        action = state.get("action_type", "request")
        entities = state.get("extracted_entities", {})
        order_id = entities.get("order_id") or "your order"
        result = state.get("action_result", {})
        status = result.get("status", "COMPLETED")
        
        if status == "SUCCESS":
            txn = result.get("transaction_id", "N/A")
            mode = result.get("approval_mode", "auto_approved")
            mode_text = "automatically approved" if mode == "auto_approved" else f"approved by manager ({state.get('reviewer', 'Manager')})"
            if action == "refund":
                amt = entities.get("amount", 0.0)
                answer = f"Your refund request for ${amt:.2f} on order {order_id} has been {mode_text}. Transaction ID: {txn}."
            elif action == "cancel_order":
                answer = f"Your cancellation request for order {order_id} has been {mode_text}. Transaction ID: {txn}."
            else:
                answer = f"Your request for {action} on order {order_id} has been successfully processed. Transaction ID: {txn}."
        else:
            reason = result.get("reason") or state.get("policy_reason") or "Policy criteria not met."
            answer = f"We are unable to process your request for {action} on {order_id}. Reason: {reason}."
            
        steps = self._record_step(state, "synthesizer", "completed", {"answer_preview": answer[:80]})
        
        return {
            "final_answer": answer,
            "execution_steps": steps
        }

    def _route_after_policy(self, state: WorkflowState) -> str:
        """Dynamic conditional router evaluating policy verdict, risk level, and intent."""
        verdict = state.get("policy_verdict")
        intent = state.get("intent")
        needs_approval = state.get("requires_human_approval", False)
        
        if verdict == "rejected":
            logger.info("[Thread %s] Routing -> rejection_handler (Policy Rejected)", state["thread_id"])
            return "rejection_handler"
            
        if intent == "query":
            logger.info("[Thread %s] Routing -> query_resolver (Informational Query)", state["thread_id"])
            return "query_resolver"
            
        if needs_approval:
            logger.info("[Thread %s] Routing -> human_review (High Risk / Manager Approval Needed)", state["thread_id"])
            return "human_review"
            
        logger.info("[Thread %s] Routing -> action_executor (Auto-Approved Mutation)", state["thread_id"])
        return "action_executor"

    def _route_after_human_review(self, state: WorkflowState) -> str:
        """Dynamic conditional router following human review decision."""
        approval = state.get("approval_status")
        if approval == "approved":
            logger.info("[Thread %s] Routing -> action_executor (Human Approved)", state["thread_id"])
            return "action_executor"
        else:
            logger.info("[Thread %s] Routing -> rejection_handler (Human Rejected / Unauthorized)", state["thread_id"])
            return "rejection_handler"

    def _build_graph(self):
        """Constructs and compiles the multi-step LangGraph state graph with checkpointer."""
        workflow = StateGraph(WorkflowState)
        
        # 1. Register Nodes
        workflow.add_node("classify_intent", self._classify_intent_node)
        workflow.add_node("evaluate_policy", self._evaluate_policy_node)
        workflow.add_node("query_resolver", self._query_resolver_node)
        workflow.add_node("human_review", self._human_review_node)
        workflow.add_node("action_executor", self._action_executor_node)
        workflow.add_node("rejection_handler", self._rejection_handler_node)
        workflow.add_node("synthesizer", self._synthesizer_node)
        
        # 2. Add Fixed and Conditional Edges
        workflow.add_edge(START, "classify_intent")
        workflow.add_edge("classify_intent", "evaluate_policy")
        
        workflow.add_conditional_edges(
            "evaluate_policy",
            self._route_after_policy,
            {
                "query_resolver": "query_resolver",
                "human_review": "human_review",
                "action_executor": "action_executor",
                "rejection_handler": "rejection_handler"
            }
        )
        
        workflow.add_conditional_edges(
            "human_review",
            self._route_after_human_review,
            {
                "action_executor": "action_executor",
                "rejection_handler": "rejection_handler"
            }
        )
        
        workflow.add_edge("query_resolver", "synthesizer")
        workflow.add_edge("action_executor", "synthesizer")
        workflow.add_edge("rejection_handler", "synthesizer")
        workflow.add_edge("synthesizer", END)
        
        return workflow.compile(checkpointer=self.checkpointer)

    async def astart_workflow(
        self, 
        prompt: str, 
        thread_id: Optional[str] = None, 
        customer_id: str = "CUST-DEFAULT"
    ) -> tuple[WorkflowState, bool, Optional[Dict[str, Any]]]:
        """Initiates a workflow run. Returns (state, is_interrupted, interrupt_payload)."""
        tid = thread_id or f"wf-{uuid.uuid4().hex[:10]}"
        config_run = {"configurable": {"thread_id": tid}}
        
        initial_state: WorkflowState = {
            "messages": [HumanMessage(content=prompt)],
            "prompt": prompt,
            "thread_id": tid,
            "customer_id": customer_id,
            "intent": "query",
            "action_type": None,
            "extracted_entities": {},
            "risk_level": "low",
            "requires_human_approval": False,
            "policy_verdict": "approved",
            "policy_reason": None,
            "approval_status": None,
            "reviewer": None,
            "reviewer_role": None,
            "reviewer_notes": None,
            "action_result": None,
            "final_answer": None,
            "execution_steps": [],
            "accumulated_tokens": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "accumulated_cost": 0.0
        }
        
        final_state = await self.graph.ainvoke(initial_state, config=config_run)
        
        # Check if the graph execution was paused by an interrupt
        state_snap = await self.graph.aget_state(config_run)
        is_interrupted = len(state_snap.tasks) > 0 and bool(state_snap.next)
        
        interrupt_payload = None
        if is_interrupted:
            # Extract interrupt details from tasks
            for task in state_snap.tasks:
                if hasattr(task, "interrupts") and task.interrupts:
                    interrupt_payload = task.interrupts[0].value
                    break
                    
        return final_state, is_interrupted, interrupt_payload

    async def aresume_workflow(
        self, 
        thread_id: str, 
        approved: bool, 
        reviewer: str, 
        reviewer_role: str = "manager", 
        feedback: Optional[str] = None
    ) -> tuple[WorkflowState, bool]:
        """Resumes a paused workflow run by sending a Command(resume=decision) to the checkpointer."""
        config_run = {"configurable": {"thread_id": thread_id}}
        
        state_snap = await self.graph.aget_state(config_run)
        if not state_snap.values:
            raise KeyError(f"Thread '{thread_id}' not found in checkpointer.")
            
        if not state_snap.next or len(state_snap.tasks) == 0:
            raise ValueError(f"Thread '{thread_id}' is not in an interrupted / pending approval state.")
            
        decision = {
            "approved": approved,
            "reviewer": reviewer,
            "reviewer_role": reviewer_role,
            "feedback": feedback
        }
        
        final_state = await self.graph.ainvoke(Command(resume=decision), config=config_run)
        
        # Verify if any subsequent interrupts exist
        post_snap = await self.graph.aget_state(config_run)
        is_still_interrupted = len(post_snap.tasks) > 0 and bool(post_snap.next)
        
        return final_state, is_still_interrupted

    async def aget_workflow_state(self, thread_id: str) -> Optional[Dict[str, Any]]:
        """Inspects and returns the StateSnapshot of a workflow thread."""
        config_run = {"configurable": {"thread_id": thread_id}}
        state_snap = await self.graph.aget_state(config_run)
        if not state_snap.values:
            return None
            
        pending_interrupt = None
        for task in state_snap.tasks:
            if hasattr(task, "interrupts") and task.interrupts:
                pending_interrupt = task.interrupts[0].value
                break
                
        status = "INTERRUPTED" if (len(state_snap.tasks) > 0 and bool(state_snap.next)) else "COMPLETED"
        
        return {
            "thread_id": thread_id,
            "status": status,
            "next_nodes": list(state_snap.next or []),
            "pending_interrupt": pending_interrupt,
            "values": state_snap.values
        }

    def get_graph_mermaid(self) -> str:
        """Exports the static topology of the compiled state graph in native Mermaid syntax."""
        return self.graph.get_graph().draw_mermaid()
