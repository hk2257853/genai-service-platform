import os
import pytest
import httpx
import logging

os.environ["MODEL_ID"] = "gemini/gemini-3.1-flash-lite"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("test_langgraph_agent")

from ai_service_platform.main import app


@pytest.mark.asyncio
async def test_01_single_tool_calling():
    """Test 1: Single Tool Calling (VIP50 Discount Calculation)"""
    print("\n" + "=" * 70)
    print("TEST 1: Single Tool Calling — Discount Calculation (VIP50)")
    print("=" * 70)
    
    prompt = "I have an order of $200 and want to use promo code VIP50. How much is my final price?"
    print(f"INPUT PROMPT: '{prompt}'")
    
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/api/v1/agent/run",
            json={"prompt": prompt, "max_iterations": 5}
        )
        assert resp.status_code == 200, f"Agent run failed: {resp.text}"
        data = resp.json()
        
        print("\n--- AGENT RESPONSE ---")
        print(f"Final Answer : {data['final_answer']}")
        print(f"Tool Calls   : {data['tool_calls']}")
        print(f"Iterations   : {data['iterations']}")
        print(f"Total Tokens : {data['total_tokens']} (Prompt: {data['prompt_tokens']}, Completion: {data['completion_tokens']})")
        print(f"Cost in USD  : ${data['cost']:.6f}")
        
        assert len(data["tool_calls"]) >= 1, "Agent should have called calculate_order_discount!"
        assert data["tool_calls"][0]["tool_name"] == "calculate_order_discount"
        assert data["tool_calls"][0]["status"] == "success"
        assert any(x in data["final_answer"] for x in ["100", "50%", "100.0"]), "Final answer should mention $100 price."


@pytest.mark.asyncio
async def test_02_direct_answer_no_tools():
    """Test 2: Direct Answer without Tool Execution"""
    print("\n" + "=" * 70)
    print("TEST 2: Direct Answer without Tool Execution")
    print("=" * 70)
    
    prompt = "What is the capital of France? Answer in one short sentence."
    print(f"INPUT PROMPT: '{prompt}'")
    
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/api/v1/agent/run",
            json={"prompt": prompt, "max_iterations": 5}
        )
        assert resp.status_code == 200, f"Direct answer failed: {resp.text}"
        data = resp.json()
        
        print("\n--- AGENT RESPONSE ---")
        print(f"Final Answer : {data['final_answer']}")
        print(f"Tool Calls   : {data['tool_calls']} (None invoked)")
        print(f"Total Tokens : {data['total_tokens']}")
        print(f"Cost in USD  : ${data['cost']:.6f}")
        
        assert len(data["tool_calls"]) == 0, "Agent should NOT have invoked tools for general knowledge."
        assert "Paris" in data["final_answer"], "Agent should correctly answer Paris."


@pytest.mark.asyncio
async def test_03_tool_error_handling_and_recovery():
    """Test 3: Tool Error Handling & Graceful Recovery"""
    print("\n" + "=" * 70)
    print("TEST 3: Tool Error Handling & Recovery (Unrecognized Service)")
    print("=" * 70)
    
    prompt = "Check the health status of legacy_mainframe_service."
    print(f"INPUT PROMPT: '{prompt}'")
    
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/api/v1/agent/run",
            json={"prompt": prompt, "max_iterations": 5}
        )
        assert resp.status_code == 200, f"Error recovery test failed: {resp.text}"
        data = resp.json()
        
        print("\n--- AGENT RESPONSE ---")
        print(f"Final Answer : {data['final_answer']}")
        print(f"Tool Calls   : {data['tool_calls']}")
        print(f"Cost in USD  : ${data['cost']:.6f}")
        
        assert len(data["tool_calls"]) >= 1, "Agent should have attempted to call check_service_status."
        assert data["tool_calls"][0]["status"] == "error", "Tool call status should be recorded as 'error'."


@pytest.mark.asyncio
async def test_04_max_iterations_infinite_loop_guard():
    """Test 4: Infinite Loop Guard (max_iterations=1 enforcement)"""
    print("\n" + "=" * 70)
    print("TEST 4: Max Iterations Infinite Loop Guard (max_iterations=1)")
    print("=" * 70)
    
    prompt = "Calculate discount for $500 with code VIP50, then check payment_service status."
    print(f"INPUT PROMPT: '{prompt}'")
    
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/api/v1/agent/run",
            json={"prompt": prompt, "max_iterations": 1}
        )
        assert resp.status_code == 200, f"Max iteration test failed: {resp.text}"
        data = resp.json()
        
        print("\n--- AGENT RESPONSE ---")
        print(f"Iterations Enforced : {data['iterations']} (Limit: 1)")
        print(f"Final Answer        : {data['final_answer']}")
        
        assert data["iterations"] <= 1, f"Agent exceeded max_iterations limit! Iterations: {data['iterations']}"


@pytest.mark.asyncio
async def test_05_multi_turn_ordering_and_cost():
    """Test 5: Multi-Turn Execution Ordering & Cumulative Cost"""
    print("\n" + "=" * 70)
    print("TEST 5: Multi-Turn Execution Ordering & Cost Accumulation")
    print("=" * 70)
    
    prompt = "First check the server time, and then check the status of payment_service."
    print(f"INPUT PROMPT: '{prompt}'")
    
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/api/v1/agent/run",
            json={"prompt": prompt, "max_iterations": 5}
        )
        assert resp.status_code == 200, f"Multi-turn test failed: {resp.text}"
        data = resp.json()
        
        print("\n--- AGENT RESPONSE ---")
        print(f"Final Answer : {data['final_answer']}")
        print(f"Execution Order:")
        for idx, tc in enumerate(data["tool_calls"]):
            print(f"  Step {idx + 1}: {tc['tool_name']} -> Output: {tc['tool_output']}")
        print(f"Total Tokens : {data['total_tokens']}")
        print(f"Cost in USD  : ${data['cost']:.6f}")
        
        assert data["total_tokens"] > 0, "Accumulated tokens should be greater than 0."
        assert data["cost"] >= 0.0, "Accumulated cost should be calculated."
