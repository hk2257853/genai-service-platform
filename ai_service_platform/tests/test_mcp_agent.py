"""Integration tests for MCPLangGraphAgent connecting to FastMCP Server over STDIO."""

import os
import asyncio
import pytest
import logging

from langchain_core.messages import AIMessage
from ai_service_platform.agents.mcp_agent import MCPLangGraphAgent

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("test_mcp_agent")


@pytest.fixture(scope="module")
def event_loop():
    """Module-level event loop for async test suite."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="module")
def mcp_agent():
    """Initializes a shared MCPLangGraphAgent instance."""
    agent = MCPLangGraphAgent()
    return agent


@pytest.mark.asyncio
async def test_01_mcp_tool_discovery(mcp_agent):
    """Test 1: Dynamic tool discovery from FastMCP server over STDIO."""
    print("\n" + "=" * 70)
    print("TEST 1: FastMCP Dynamic Tool Discovery via Protocol")
    print("=" * 70)

    await mcp_agent.initialize()

    tool_names = list(mcp_agent.tools_by_name.keys())
    print(f"Discovered MCP Tools: {tool_names}")
    print(f"Generated Tool Specs: {[spec['function']['name'] for spec in mcp_agent.tools_spec]}")

    assert "get_order_details" in tool_names, "Expected 'get_order_details' tool from MCP server"
    assert "get_product_details" in tool_names, "Expected 'get_product_details' tool from MCP server"
    assert len(mcp_agent.tools_spec) >= 2


@pytest.mark.asyncio
async def test_02_mcp_single_turn_tool_call(mcp_agent):
    """Test 2: Single-turn agent reasoning and tool execution over MCP."""
    print("\n" + "=" * 70)
    print("TEST 2: Single-Turn Query (Order Lookup via MCP)")
    print("=" * 70)

    prompt = "What is the status, carrier, and item for order ORD-1001?"
    print(f"INPUT PROMPT: '{prompt}'")

    result = await mcp_agent.ainvoke(prompt=prompt, thread_id="thread-test-single")
    final_state = result["state"]
    last_msg = final_state["messages"][-1]

    print("\n--- AGENT RESPONSE ---")
    print(f"Thread ID      : {result['thread_id']}")
    print(f"Final Answer   : {last_msg.content}")
    print(f"Tool Calls Exec: {final_state['tool_calls']}")
    print(f"Total Tokens   : {final_state['accumulated_tokens']}")
    print(f"Cost in USD    : ${final_state['accumulated_cost']:.6f}")

    assert len(final_state["tool_calls"]) >= 1, "Agent should have invoked get_order_details tool"
    assert final_state["tool_calls"][0]["tool_name"] == "get_order_details"
    assert final_state["tool_calls"][0]["status"] == "success"
    answer_lower = last_msg.content.lower()
    assert any(term in answer_lower for term in ["shipped", "fedex", "sony", "wh-1000xm5"])

    await asyncio.sleep(7.0)


@pytest.mark.asyncio
async def test_03_mcp_multi_turn_coreference(mcp_agent):
    """Test 3: Multi-turn dialog with memory, resolving coreference from previous turn."""
    print("\n" + "=" * 70)
    print("TEST 3: Multi-Turn Conversation with State Memory (Thread Continuity)")
    print("=" * 70)

    thread_id = "thread-test-multiturn-101"

    # Turn 1: Lookup order
    prompt_1 = "Hi! Can you check order ORD-2002 for me?"
    print(f"\n[TURN 1 PROMPT]: '{prompt_1}'")
    res_1 = await mcp_agent.ainvoke(prompt=prompt_1, thread_id=thread_id)
    answer_1 = res_1["state"]["messages"][-1].content
    print(f"[TURN 1 ANSWER]: {answer_1}")

    assert any(term in answer_1.lower() for term in ["processing", "bob smith", "logitech", "mx master 3s"])

    await asyncio.sleep(7.0)

    # Turn 2: Follow-up relying on memory ("that product")
    prompt_2 = "Is that product still in stock, and how many are available?"
    print(f"\n[TURN 2 PROMPT]: '{prompt_2}'")
    res_2 = await mcp_agent.ainvoke(prompt=prompt_2, thread_id=thread_id)
    final_state_2 = res_2["state"]
    answer_2 = final_state_2["messages"][-1].content
    print(f"[TURN 2 ANSWER]: {answer_2}")
    print(f"[TURN 2 TOOL CALLS]: {final_state_2['tool_calls']}")

    # Agent should have invoked get_product_details for Logitech MX Master 3S
    tool_names = [tc["tool_name"] for tc in final_state_2["tool_calls"]]
    assert "get_product_details" in tool_names, "Expected agent to call get_product_details in turn 2"
    assert any(term in answer_2.lower() for term in ["stock", "out of stock", "0", "restock"])

    await asyncio.sleep(7.0)


@pytest.mark.asyncio
async def test_04_mcp_tool_error_resilience(mcp_agent):
    """Test 4: Agent handling structured errors returned by MCP tool gracefully."""
    print("\n" + "=" * 70)
    print("TEST 4: MCP Tool Error Resilience & Self-Correction")
    print("=" * 70)

    prompt = "Please look up order status for ORD-9999."
    print(f"INPUT PROMPT: '{prompt}'")

    result = await mcp_agent.ainvoke(prompt=prompt, thread_id="thread-test-error")
    final_state = result["state"]
    last_msg = final_state["messages"][-1]

    print("\n--- AGENT RESPONSE ---")
    print(f"Final Answer   : {last_msg.content}")
    print(f"Tool Calls     : {final_state['tool_calls']}")

    assert len(final_state["tool_calls"]) >= 1
    # Check that tool output contained the error message from the MCP server
    tool_output = final_state["tool_calls"][0]["tool_output"]
    assert "not found" in tool_output.lower() or "error" in tool_output.lower()
    # The agent should explain to the user that the order was not found
    assert any(term in last_msg.content.lower() for term in ["not found", "not be found", "could not be found", "couldn't find", "could not find", "doesn't exist"])
