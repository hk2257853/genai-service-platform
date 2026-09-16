"""Production LangGraph Agent integrated with an external MCP Server over STDIO.

Demonstrates:
1. Dynamic Tool Discovery: Tools are negotiated at startup via MCP JSON-RPC protocol without hardcoded schemas.
2. Process Isolation: MCP server executes as an independent subprocess communicating over stdin/stdout.
3. Multi-Turn State Machine: Maintains conversation memory and context across turns using LangGraph checkpointer.
4. Resilient Error Handling: Structured tool errors feed back into the ReAct loop for self-correction.
"""

import sys
import json
import uuid
import logging
from typing import TypedDict, Annotated, List, Dict, Any, Optional

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_mcp_adapters.client import MultiServerMCPClient

from ai_service_platform import config
from ai_service_platform.core.client import AsyncLLMClient
from ai_service_platform.core.utils import extract_usage_and_cost

logger = logging.getLogger("mcp_langgraph_agent")


def format_tool_result(result: Any) -> str:
    """Formats raw MCP tool results (content blocks or dictionaries) into a clean string for ToolMessage."""
    if isinstance(result, str):
        return result
    if isinstance(result, list):
        texts = []
        for item in result:
            if isinstance(item, dict) and "text" in item:
                texts.append(item["text"])
            elif hasattr(item, "text"):
                texts.append(getattr(item, "text"))
            else:
                texts.append(str(item))
        return "\n".join(texts)
    return json.dumps(result, default=str)


def serialize_messages_for_llm(messages: list) -> list:
    """Converts LangChain message instances from LangGraph state into LiteLLM compatible dicts."""
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


# --- 1. LangGraph State Definition ---

class MCPAgentState(TypedDict):
    messages: Annotated[list, add_messages]
    tool_calls: List[Dict[str, Any]]
    iteration_count: int
    max_iterations: int
    accumulated_tokens: Dict[str, int]
    accumulated_cost: float


# --- 2. MCP LangGraph Agent ---

class MCPLangGraphAgent:
    """Autonomous LangGraph agent utilizing external MCP servers for tool execution."""

    def __init__(
        self,
        mcp_servers: Optional[Dict[str, Dict[str, Any]]] = None,
        llm_client: Optional[AsyncLLMClient] = None,
        checkpointer: Optional[Any] = None
    ):
        self.llm_client = llm_client or AsyncLLMClient(default_model=config.MODEL_ID)
        self.checkpointer = checkpointer if checkpointer is not None else MemorySaver()

        # Default MCP server: Ecommerce FastMCP Server running over STDIO
        default_servers = {
            "ecommerce": {
                "command": sys.executable,
                "args": ["-m", "ai_service_platform.mcp_servers.ecommerce_server"],
                "transport": "stdio"
            }
        }
        self.mcp_servers_config = mcp_servers or default_servers
        self.mcp_client = MultiServerMCPClient(self.mcp_servers_config)

        self.tools_by_name: Dict[str, Any] = {}
        self.tools_spec: List[Dict[str, Any]] = []
        self._initialized = False

        self.graph = self._build_graph()

    async def initialize(self) -> None:
        """Connects to MCP servers, negotiates protocol, and dynamically discovers available tools."""
        if self._initialized:
            return

        logger.info("Discovering tools from MCP servers...")
        tools = await self.mcp_client.get_tools()
        self.tools_by_name = {tool.name: tool for tool in tools}
        self.tools_spec = [convert_to_openai_tool(t) for t in tools]
        self._initialized = True
        logger.info("Successfully registered %d MCP tools: %s", len(self.tools_by_name), list(self.tools_by_name.keys()))

    async def _agent_node(self, state: MCPAgentState) -> Dict[str, Any]:
        """Invokes LLM with dynamically bound MCP tool specs, accumulating tokens and cost."""
        if not self._initialized:
            await self.initialize()

        logger.info("Agent node executing turn %d (max: %d)", state["iteration_count"] + 1, state["max_iterations"])
        llm_messages = serialize_messages_for_llm(state["messages"])

        response = await self.llm_client.acompletion(
            action_name="mcp_agent_reasoning",
            messages=llm_messages,
            tools=self.tools_spec,
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

        # Parse tool calls for LangChain AIMessage
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

    async def _tool_node(self, state: MCPAgentState) -> Dict[str, Any]:
        """Dispatches tool calls to the external MCP server over JSON-RPC, catching errors safely."""
        last_message = state["messages"][-1]
        tool_calls = getattr(last_message, "tool_calls", None) or []

        new_tool_messages = []
        recorded_calls = list(state.get("tool_calls", []))

        for tc in tool_calls:
            fn_name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", "")
            args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", {})
            call_id = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", "call_default")

            logger.info("Dispatching tool '%s' via MCP with args: %s", fn_name, args)

            if fn_name in self.tools_by_name:
                try:
                    mcp_tool = self.tools_by_name[fn_name]
                    raw_result = await mcp_tool.ainvoke(args)
                    result_str = format_tool_result(raw_result)
                    status = "success"
                except Exception as e:
                    logger.warning("MCP tool '%s' execution threw exception: %s", fn_name, e)
                    result_str = json.dumps({"error": f"MCP execution failed: {str(e)}"})
                    status = "error"
            else:
                result_str = json.dumps({"error": f"MCP tool '{fn_name}' is not registered on the server."})
                status = "error"

            recorded_calls.append({
                "tool_name": fn_name,
                "tool_args": args,
                "tool_output": result_str,
                "status": status
            })

            new_tool_messages.append(
                ToolMessage(
                    content=result_str,
                    tool_call_id=call_id,
                    name=fn_name
                )
            )

        return {
            "messages": new_tool_messages,
            "tool_calls": recorded_calls
        }

    def _should_continue(self, state: MCPAgentState) -> str:
        """Evaluates whether to cycle back to tools or finalize response."""
        if state["iteration_count"] >= state["max_iterations"]:
            logger.warning("Agent reached maximum allowed iterations (%d). Ending turn.", state["max_iterations"])
            return "end"

        last_message = state["messages"][-1]
        tool_calls = getattr(last_message, "tool_calls", None) or []

        if tool_calls and len(tool_calls) > 0:
            return "tools"

        return "end"

    def _build_graph(self):
        """Constructs and compiles the cyclic ReAct state graph."""
        workflow = StateGraph(MCPAgentState)

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
        """Executes a turn in the conversation state machine, preserving state across turns with checkpointer."""
        if not self._initialized:
            await self.initialize()

        active_thread_id = thread_id or uuid.uuid4().hex
        config_run = {"configurable": {"thread_id": active_thread_id}}

        current_snapshot = await self.graph.aget_state(config_run)
        has_existing_messages = bool(
            current_snapshot.values
            and "messages" in current_snapshot.values
            and len(current_snapshot.values["messages"]) > 0
        )

        if not has_existing_messages:
            sys_content = system_prompt or (
                "You are an intelligent e-commerce customer support and data assistant. "
                "You have access to real store data tools provided via an external MCP (Model Context Protocol) server. "
                "Always check order and product details using tools when answering customer inquiries. "
                "Remember details from previous turns in the conversation to seamlessly answer follow-up questions."
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
