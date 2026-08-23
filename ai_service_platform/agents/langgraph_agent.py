import os
import json
import uuid
import logging
from datetime import datetime, timezone
from typing import TypedDict, Annotated, List, Dict, Any, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

from ai_service_platform import config
from ai_service_platform.core.client import AsyncLLMClient
from ai_service_platform.core.utils import extract_usage_and_cost
from ai_service_platform.data import (
    ORDER_DATABASE,
    PRODUCT_DATABASE,
    MICROSERVICE_REGISTRY,
    PROMO_CODES
)

logger = logging.getLogger("langgraph_agent")


# --- 1. Tool Implementations ---

def calculate_order_discount(order_amount: float, discount_code: str) -> str:
    """Calculates discounted order price based on promo codes: VIP50 (50%), NEW20 (20%), SAVE10 ($10 off)."""
    code = str(discount_code).strip().upper()
    amt = float(order_amount)
    if code not in PROMO_CODES:
        raise ValueError(f"Invalid discount code '{discount_code}'. Valid codes are {list(PROMO_CODES.keys())}.")
        
    promo = PROMO_CODES[code]
    if promo["type"] == "percentage":
        discount_saved = amt * promo["value"]
        final_price = amt - discount_saved
    elif promo["type"] == "fixed":
        discount_saved = min(promo["value"], amt)
        final_price = max(0.0, amt - promo["value"])
    else:
        raise ValueError(f"Unknown discount calculation type for '{code}'.")
        
    return json.dumps({
        "original_amount": amt,
        "discount_code": code,
        "discount_saved": discount_saved,
        "final_price": final_price
    })


def check_service_status(service_name: str) -> str:
    """Checks the live health and status of microservices in the platform."""
    key = str(service_name).strip().lower()
    if key not in MICROSERVICE_REGISTRY:
        raise ValueError(f"Microservice '{service_name}' not found. Available services: {list(MICROSERVICE_REGISTRY.keys())}")
    
    return json.dumps({
        "service_name": key,
        "status_report": MICROSERVICE_REGISTRY[key]
    })


def get_server_time() -> str:
    """Returns the current UTC ISO timestamp of the service platform."""
    return json.dumps({
        "current_utc_time": datetime.now(timezone.utc).isoformat()
    })


def check_order_status(order_id: str) -> str:
    """Checks the real-time shipping, tracking, and fulfillment status of a customer order."""
    normalized_id = str(order_id).strip().upper()
    if normalized_id not in ORDER_DATABASE:
        available = list(ORDER_DATABASE.keys())
        raise ValueError(f"Order '{order_id}' was not found in the database. Available orders: {available}")
    return json.dumps(ORDER_DATABASE[normalized_id])


def get_product_info(product_id_or_name: str) -> str:
    """Retrieves product specifications, live stock availability, pricing, and warranty details by SKU or name."""
    query = str(product_id_or_name).strip()
    query_upper = query.upper()
    
    # 1. Exact SKU match
    if query_upper in PRODUCT_DATABASE:
        return json.dumps(PRODUCT_DATABASE[query_upper])
        
    # 2. Case-insensitive substring match
    query_lower = query.lower()
    for prod in PRODUCT_DATABASE.values():
        if query_lower in prod["name"].lower() or prod["name"].lower() in query_lower:
            return json.dumps(prod)
            
    # 3. Token overlap match (at least 2 matching words)
    query_tokens = set(query_lower.split())
    for prod in PRODUCT_DATABASE.values():
        prod_tokens = set(prod["name"].lower().split())
        if len(query_tokens.intersection(prod_tokens)) >= 2:
            return json.dumps(prod)
            
    available_skus = list(PRODUCT_DATABASE.keys())
    raise ValueError(f"Product '{product_id_or_name}' not found. Available product SKUs: {available_skus}")


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
    },
    {
        "type": "function",
        "function": {
            "name": "check_order_status",
            "description": "Checks real-time fulfillment, tracking, carrier, and shipping status of an order by order ID (e.g. ORD-1001).",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string", "description": "Unique order identifier (e.g. ORD-1001, ORD-2002)."}
                },
                "required": ["order_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_product_info",
            "description": "Retrieves product specs, price, stock/inventory availability, warranty, and return policy by product SKU or name (e.g. PROD-101, Sony WH-1000XM5, Logitech MX Master 3S).",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id_or_name": {"type": "string", "description": "Product SKU (e.g. PROD-101) or product name/model."}
                },
                "required": ["product_id_or_name"]
            }
        }
    }
]

TOOL_CALLABLE_MAP = {
    "calculate_order_discount": calculate_order_discount,
    "check_service_status": check_service_status,
    "get_server_time": get_server_time,
    "check_order_status": check_order_status,
    "get_product_info": get_product_info
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
            
        if hasattr(msg, "name") and msg.name:
            entry["name"] = msg.name
            
        serialized.append(entry)
    return serialized


# --- 3. LangGraph State Definition ---

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    tool_calls: List[Dict[str, Any]]
    iteration_count: int
    max_iterations: int
    accumulated_tokens: Dict[str, int]
    accumulated_cost: float


# --- 4. LangGraph Node Functions ---

class LangGraphAgent:
    """Production LangGraph agent with tool execution, checkpointer-backed short-term memory, and multi-turn cost accumulation."""

    def __init__(self, llm_client: Optional[AsyncLLMClient] = None, checkpointer: Optional[Any] = None):
        self.llm_client = llm_client or AsyncLLMClient(default_model=config.MODEL_ID)
        self.checkpointer = checkpointer if checkpointer is not None else MemorySaver()
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
        """Constructs and compiles the cyclic LangGraph state machine with attached checkpointer."""
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
        
        return workflow.compile(checkpointer=self.checkpointer)

    async def ainvoke(
        self, 
        prompt: str, 
        thread_id: Optional[str] = None,
        system_prompt: Optional[str] = None, 
        max_iterations: int = 5
    ) -> Dict[str, Any]:
        """Executes the agent state machine with persistent thread checkpointing."""
        active_thread_id = thread_id or uuid.uuid4().hex
        config_run = {"configurable": {"thread_id": active_thread_id}}

        # Inspect current state for this thread in checkpointer
        current_snapshot = await self.graph.aget_state(config_run)
        has_existing_messages = bool(
            current_snapshot.values 
            and "messages" in current_snapshot.values 
            and len(current_snapshot.values["messages"]) > 0
        )

        if not has_existing_messages:
            sys_content = system_prompt or (
                "You are an autonomous AI assistant for the GenAI Service Platform with short-term conversational memory. "
                "You have access to tools for discounts, service status, order tracking (check_order_status), "
                "and product catalog info (get_product_info). "
                "Use tools when required to provide factual, accurate answers. "
                "Always remember and refer back to context from previous turns in the conversation when answering follow-up questions."
            )
            input_state: Dict[str, Any] = {
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
        else:
            # Continuing thread: append HumanMessage and reset per-turn tool calls and iterations
            input_state = {
                "messages": [HumanMessage(content=prompt)],
                "tool_calls": [],
                "iteration_count": 0,
                "max_iterations": max_iterations
            }

        final_state = await self.graph.ainvoke(input_state, config=config_run)

        return {
            "state": final_state,
            "thread_id": active_thread_id
        }

    async def aget_thread_history(self, thread_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves stored state snapshot and message history for a specific thread."""
        config_run = {"configurable": {"thread_id": thread_id}}
        snapshot = await self.graph.aget_state(config_run)
        if not snapshot.values or "messages" not in snapshot.values:
            return None
        return snapshot.values

    async def aclear_thread(self, thread_id: str) -> bool:
        """Clears/resets state checkpoint for the specified thread."""
        if hasattr(self.checkpointer, "adelete_thread"):
            await self.checkpointer.adelete_thread(thread_id)
            return True
        elif hasattr(self.checkpointer, "delete_thread"):
            self.checkpointer.delete_thread(thread_id)
            return True
        elif hasattr(self.checkpointer, "storage") and thread_id in self.checkpointer.storage:
            del self.checkpointer.storage[thread_id]
            return True
        return True
