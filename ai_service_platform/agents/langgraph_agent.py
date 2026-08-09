import os
import json
import logging
from datetime import datetime, timezone
from typing import TypedDict, Annotated, List, Dict, Any, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

from ai_service_platform import config
from ai_service_platform.core.client import AsyncLLMClient
from ai_service_platform.core.utils import extract_usage_and_cost

logger = logging.getLogger("langgraph_agent")


# --- 1. Tools Definition & Registry ---

def calculate_order_discount(order_amount: float, discount_code: str) -> str:
    """Calculates discounted order price based on promo codes: VIP50 (50%), NEW20 (20%), SAVE10 ($10 off)."""
    code = str(discount_code).strip().upper()
    amt = float(order_amount)
    if code == "VIP50":
        final_price = amt * 0.5
        discount_saved = amt * 0.5
    elif code == "NEW20":
        final_price = amt * 0.8
        discount_saved = amt * 0.2
    elif code == "SAVE10":
        discount_saved = min(10.0, amt)
        final_price = max(0.0, amt - 10.0)
    else:
        raise ValueError(f"Invalid discount code '{discount_code}'. Valid codes are VIP50, NEW20, SAVE10.")
        
    return json.dumps({
        "original_amount": amt,
        "discount_code": code,
        "discount_saved": discount_saved,
        "final_price": final_price
    })


def check_service_status(service_name: str) -> str:
    """Checks the live health and status of microservices in the platform."""
    known_services = {
        "payment_service": "HEALTHY (latency: 14ms, uptime: 99.98%)",
        "inventory_service": "HEALTHY (latency: 22ms, uptime: 99.95%)",
        "auth_service": "HEALTHY (latency: 8ms, uptime: 100%)",
        "notification_service": "DEGRADED (high queue delay: 350ms)"
    }
    key = str(service_name).strip().lower()
    if key not in known_services:
        raise ValueError(f"Microservice '{service_name}' not found. Available services: {list(known_services.keys())}")
    
    return json.dumps({
        "service_name": key,
        "status_report": known_services[key]
    })


def get_server_time() -> str:
    """Returns the current UTC ISO timestamp of the service platform."""
    return json.dumps({
        "current_utc_time": datetime.now(timezone.utc).isoformat()
    })


# Tool Schema Registry for LiteLLM tool-calling format
TOOLS_SPEC = [
    {
        "type": "function",
        "function": {
            "name": "calculate_order_discount",
            "description": "Calculates discounted order price based on promo codes (VIP50, NEW20, SAVE10).",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_amount": {"type": "number", "description": "Original order amount in dollars."},
                    "discount_code": {"type": "string", "description": "Promo discount code (e.g. VIP50)."}
                },
                "required": ["order_amount", "discount_code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "check_service_status",
            "description": "Checks the live health and status of microservices in the platform.",
            "parameters": {
                "type": "object",
                "properties": {
                    "service_name": {"type": "string", "description": "Name of the service (e.g. payment_service, auth_service)."}
                },
                "required": ["service_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_server_time",
            "description": "Returns the current UTC ISO timestamp of the service platform.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    }
]

TOOL_CALLABLE_MAP = {
    "calculate_order_discount": calculate_order_discount,
    "check_service_status": check_service_status,
    "get_server_time": get_server_time
}


def serialize_messages_for_llm(messages: list) -> list:
    """Converts LangChain message objects from LangGraph add_messages into LiteLLM compatible dicts."""
    serialized = []
    for msg in messages:
        if isinstance(msg, dict):
            serialized.append(msg)
            continue
        
        msg_type = getattr(msg, "type", "user")
        if msg_type == "human":
            role = "user"
        elif msg_type == "ai":
            role = "assistant"
        elif msg_type == "system":
            role = "system"
        elif msg_type == "tool":
            role = "tool"
        else:
            role = "user"
            
        entry: Dict[str, Any] = {"role": role, "content": msg.content or ""}
        
        # Preserve tool calls on AI messages
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            entry["tool_calls"] = [
                {
                    "id": tc.get("id", "call_default") if isinstance(tc, dict) else getattr(tc, "id", "call_default"),
                    "type": "function",
                    "function": {
                        "name": tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", ""),
                        "arguments": json.dumps(tc.get("args", {})) if isinstance(tc, dict) and isinstance(tc.get("args"), dict) else (
                            json.dumps(getattr(tc, "args", {})) if hasattr(tc, "args") else "{}"
                        )
                    }
                }
                for tc in msg.tool_calls
            ]
            
        if hasattr(msg, "tool_call_id") and msg.tool_call_id:
            entry["tool_call_id"] = msg.tool_call_id
            
        serialized.append(entry)
    return serialized


# --- 2. LangGraph State Definition ---

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    tool_calls: List[Dict[str, Any]]
    iteration_count: int
    max_iterations: int
    accumulated_tokens: Dict[str, int]
    accumulated_cost: float


# --- 3. LangGraph Node Functions ---

class LangGraphAgent:
    """Production LangGraph agent with tool execution, error handling, and multi-turn cost accumulation."""

    def __init__(self, llm_client: Optional[AsyncLLMClient] = None):
        self.llm_client = llm_client or AsyncLLMClient(default_model=config.MODEL_ID)
        self.graph = self._build_graph()

    async def _agent_node(self, state: AgentState) -> Dict[str, Any]:
        """Invokes LLM with bound tools, tracks token usage and cost, and records assistant output."""
        logger.info("Agent node executing turn %d (max: %d)", state["iteration_count"] + 1, state["max_iterations"])
        
        llm_messages = serialize_messages_for_llm(state["messages"])
        
        response = await self.llm_client.acompletion(
            action_name="langgraph_agent_step",
            messages=llm_messages,
            tools=TOOLS_SPEC,
            tool_choice="auto",
            temperature=0.1
        )
        
        message = response.choices[0].message
        metrics = extract_usage_and_cost(response)
        
        acc_tokens = state.get("accumulated_tokens", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}).copy()
        acc_tokens["prompt_tokens"] += metrics["prompt_tokens"]
        acc_tokens["completion_tokens"] += metrics["completion_tokens"]
        acc_tokens["total_tokens"] += metrics["total_tokens"]
        
        new_cost = state.get("accumulated_cost", 0.0) + metrics["cost"]
        
        # Build LangChain AIMessage for LangGraph state reducer
        tool_calls_payload = []
        if message.tool_calls:
            for tc in message.tool_calls:
                try:
                    args_dict = json.loads(tc.function.arguments) if isinstance(tc.function.arguments, str) else tc.function.arguments
                except Exception:
                    args_dict = {}
                tool_calls_payload.append({
                    "id": tc.id,
                    "name": tc.function.name,
                    "args": args_dict
                })
        
        ai_msg = AIMessage(
            content=message.content or "",
            tool_calls=tool_calls_payload
        )

        return {
            "messages": [ai_msg],
            "iteration_count": state["iteration_count"] + 1,
            "accumulated_tokens": acc_tokens,
            "accumulated_cost": new_cost
        }

    async def _tool_node(self, state: AgentState) -> Dict[str, Any]:
        """Safely executes tool calls, catches tool exceptions, and returns ToolMessages back to state."""
        last_message = state["messages"][-1]
        tool_calls = getattr(last_message, "tool_calls", None) or []
        
        new_tool_messages = []
        recorded_calls = list(state.get("tool_calls", []))

        for tc in tool_calls:
            fn_name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", "")
            args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", {})
            call_id = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", "call_default")
            
            logger.info("Executing tool '%s' with args: %s", fn_name, args)
            
            if fn_name in TOOL_CALLABLE_MAP:
                try:
                    result = TOOL_CALLABLE_MAP[fn_name](**args)
                    status = "success"
                except Exception as e:
                    logger.warning("Tool '%s' execution threw an error: %s", fn_name, e)
                    result = json.dumps({"error": f"Tool execution failed: {str(e)}"})
                    status = "error"
            else:
                result = json.dumps({"error": f"Tool '{fn_name}' is not recognized."})
                status = "error"

            recorded_calls.append({
                "tool_name": fn_name,
                "tool_args": args,
                "tool_output": result,
                "status": status
            })

            new_tool_messages.append(
                ToolMessage(
                    content=result,
                    tool_call_id=call_id,
                    name=fn_name
                )
            )

        return {
            "messages": new_tool_messages,
            "tool_calls": recorded_calls
        }

    def _should_continue(self, state: AgentState) -> str:
        """Conditional routing edge checking for tool calls and enforcing the max_iterations guard."""
        if state["iteration_count"] >= state["max_iterations"]:
            logger.warning("Agent reached max_iterations (%d). Halting loop.", state["max_iterations"])
            return "end"

        last_message = state["messages"][-1]
        tool_calls = getattr(last_message, "tool_calls", None) or []

        if tool_calls and len(tool_calls) > 0:
            return "tools"
        
        return "end"

    def _build_graph(self):
        """Constructs and compiles the cyclic LangGraph state machine."""
        workflow = StateGraph(AgentState)
        
        workflow.add_node("agent", self._agent_node)
        workflow.add_node("tools", self._tool_node)
        
        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges(
            "agent", 
            self._should_continue, 
            {"tools": "tools", "end": END}
        )
        workflow.add_edge("tools", "agent")
        
        return workflow.compile()

    async def ainvoke(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None, 
        max_iterations: int = 5
    ) -> AgentState:
        """Executes the agent state machine until completion or max_iterations is reached."""
        sys_content = system_prompt or (
            "You are an autonomous AI assistant for the GenAI Service Platform. "
            "You have access to tools for discounts, microservice health, and server time. "
            "Use tools when required to provide factual, accurate answers. "
            "If a tool returns an error, explain the issue clearly to the user."
        )

        initial_state: AgentState = {
            "messages": [
                SystemMessage(content=sys_content),
                HumanMessage(content=prompt)
            ],
            "tool_calls": [],
            "iteration_count": 0,
            "max_iterations": max_iterations,
            "accumulated_tokens": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "accumulated_cost": 0.0
        }

        final_state = await self.graph.ainvoke(initial_state)
        return final_state
