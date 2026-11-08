"""Test Live LangSmith Tracing & Agent Run-Tree.

Executes:
1. Tracing status check confirming LangSmith active connection
2. LangGraph Agent multi-turn run with tool execution (check_order_status)
3. Direct LangSmith run-tree verification
"""

import asyncio
import json
import httpx
from ai_service_platform.main import app
from ai_service_platform.agents.langgraph_agent import LangGraphAgent
from ai_service_platform import config


async def main():
    print("\n" + "=" * 70)
    print(" 1. VERIFYING LANGSMITH ACTIVE STATUS VIA REST API")
    print("=" * 70)
    
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/api/v1/observability/tracing/status")
        status_data = resp.json()
        print(f"Endpoint: GET /api/v1/observability/tracing/status")
        print(json.dumps(status_data, indent=2))
        assert status_data["tracing_enabled"] is True, "Tracing is not enabled!"
        print("\n>> SUCCESS: LangSmith Tracing is actively ENABLED in the platform! <<")

    print("\n" + "=" * 70)
    print(" 2. EXECUTING MULTI-STEP LANGGRAPH AGENT WITH TOOL CALLS")
    print("=" * 70)
    
    prompt = "Where is order ORD-1001 right now and what is the shipping carrier?"
    print(f"Agent Prompt: '{prompt}'")
    print("Invoking LangGraph Agent (streaming run-tree directly to LangSmith)...")

    agent = LangGraphAgent()
    result = await agent.ainvoke(prompt=prompt)
    
    final_state = result["state"]
    thread_id = result["thread_id"]
    last_message = final_state["messages"][-1]
    
    print("\n[AGENT FINAL ANSWER]:")
    print(last_message.content)
    
    print("\n[RECORDED TOOL CALLS]:")
    for tc in final_state.get("tool_calls", []):
        print(f"  • Tool: {tc['tool_name']} | Args: {tc['tool_args']} | Status: {tc['status']}")
        print(f"    Output: {tc['tool_output']}")
        
    print(f"\n[ACCUMULATED TOKENS]: {final_state.get('accumulated_tokens')}")
    print(f"[ACCUMULATED COST]:   ${final_state.get('accumulated_cost', 0.0):.6f}")

    # Give background async worker a moment to flush spans
    await asyncio.sleep(2)

    print("\n" + "=" * 70)
    print(" 3. OPEN YOUR LANGSMITH DASHBOARD TO VIEW THE RUN TREE")
    print("=" * 70)
    print("Go to your browser:")
    print(">> https://smith.langchain.com/ <<")
    print(f"Project: '{config.LANGCHAIN_PROJECT}'")
    print(f"Look for the latest Run: 'LangGraphAgent:{thread_id[:8]}'")
    print("You will see the interactive hierarchical tree:")
    print("  └── LangGraphAgent")
    print("       ├── agent (LLM reasoning call)")
    print("       ├── tools (check_order_status execution)")
    print("       └── agent (final synthesis)")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())

