import os
import asyncio
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
        print(f"Thread ID    : {data['thread_id']}")
        print(f"Final Answer : {data['final_answer']}")
        print(f"Tool Calls   : {data['tool_calls']}")
        print(f"Iterations   : {data['iterations']}")
        print(f"Messages Cnt : {data['messages_count']}")
        print(f"Total Tokens : {data['total_tokens']} (Prompt: {data['prompt_tokens']}, Completion: {data['completion_tokens']})")
        print(f"Cost in USD  : ${data['cost']:.6f}")
        
        assert len(data["tool_calls"]) >= 1, "Agent should have called calculate_order_discount!"
        assert data["tool_calls"][0]["tool_name"] == "calculate_order_discount"
        assert data["tool_calls"][0]["status"] == "success"
        assert any(x in data["final_answer"] for x in ["100", "50%", "100.0"]), "Final answer should mention $100 price."
    
    await asyncio.sleep(1.5)


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

    await asyncio.sleep(1.5)


@pytest.mark.asyncio
async def test_03_check_order_status_tool():
    """Test 3: check_order_status tool execution for real-time tracking"""
    print("\n" + "=" * 70)
    print("TEST 3: Domain Tool — check_order_status (ORD-1001)")
    print("=" * 70)
    
    prompt = "What is the status of order ORD-1001?"
    print(f"INPUT PROMPT: '{prompt}'")
    
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/api/v1/agent/run",
            json={"prompt": prompt, "max_iterations": 5}
        )
        assert resp.status_code == 200, f"Order status lookup failed: {resp.text}"
        data = resp.json()
        
        print("\n--- AGENT RESPONSE ---")
        print(f"Final Answer : {data['final_answer']}")
        print(f"Tool Calls   : {data['tool_calls']}")
        
        assert len(data["tool_calls"]) >= 1, "Agent should invoke check_order_status!"
        assert data["tool_calls"][0]["tool_name"] == "check_order_status"
        assert data["tool_calls"][0]["status"] == "success"
        answer_lower = data["final_answer"].lower()
        assert "shipped" in answer_lower or "fedex" in answer_lower, "Response must include shipping status or carrier."

    await asyncio.sleep(1.5)


@pytest.mark.asyncio
async def test_04_get_product_info_tool():
    """Test 4: get_product_info tool execution for specs and stock"""
    print("\n" + "=" * 70)
    print("TEST 4: Domain Tool — get_product_info (Sony WH-1000XM5)")
    print("=" * 70)
    
    prompt = "Give me information, pricing, and stock status for Sony WH-1000XM5."
    print(f"INPUT PROMPT: '{prompt}'")
    
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/api/v1/agent/run",
            json={"prompt": prompt, "max_iterations": 5}
        )
        assert resp.status_code == 200, f"Product info lookup failed: {resp.text}"
        data = resp.json()
        
        print("\n--- AGENT RESPONSE ---")
        print(f"Final Answer : {data['final_answer']}")
        print(f"Tool Calls   : {data['tool_calls']}")
        
        assert len(data["tool_calls"]) >= 1, "Agent should invoke get_product_info!"
        assert data["tool_calls"][0]["tool_name"] == "get_product_info"
        assert data["tool_calls"][0]["status"] == "success"
        answer = data["final_answer"]
        assert "399" in answer or "stock" in answer.lower(), "Response must mention product price or stock availability."

    await asyncio.sleep(1.5)


@pytest.mark.asyncio
async def test_05_multi_turn_short_term_memory_order_coreference():
    """Test 5: Multi-Turn Conversation Memory & Coreference Resolution (Order Lookup)"""
    print("\n" + "=" * 70)
    print("TEST 5: Multi-Turn Memory & Coreference Resolution (Order Follow-up)")
    print("=" * 70)
    
    thread_id = "test-thread-order-memory-01"
    
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        # Turn 1: Ask about order ORD-1001
        turn1_prompt = "Hello! Can you check the status of my order ORD-1001?"
        print(f"TURN 1 PROMPT: '{turn1_prompt}' (thread_id={thread_id})")
        resp1 = await client.post(
            "/api/v1/agent/run",
            json={"prompt": turn1_prompt, "thread_id": thread_id, "max_iterations": 5}
        )
        assert resp1.status_code == 200, f"Turn 1 failed: {resp1.text}"
        data1 = resp1.json()
        print(f"TURN 1 ANSWER: {data1['final_answer']}")
        print(f"TURN 1 MESSAGES COUNT: {data1['messages_count']}")
        assert len(data1["tool_calls"]) >= 1
        
        await asyncio.sleep(2.0)
        
        # Turn 2: Coreference follow-up without mentioning ORD-1001
        turn2_prompt = "What was the carrier for that order, and when will it arrive?"
        print(f"\nTURN 2 PROMPT: '{turn2_prompt}' (thread_id={thread_id})")
        resp2 = await client.post(
            "/api/v1/agent/run",
            json={"prompt": turn2_prompt, "thread_id": thread_id, "max_iterations": 5}
        )
        assert resp2.status_code == 200, f"Turn 2 failed: {resp2.text}"
        data2 = resp2.json()
        print(f"TURN 2 ANSWER: {data2['final_answer']}")
        print(f"TURN 2 MESSAGES COUNT: {data2['messages_count']}")
        
        # Verify memory retention
        assert data2["messages_count"] > data1["messages_count"], "Messages count should increase across turns."
        answer2_lower = data2["final_answer"].lower()
        assert "fedex" in answer2_lower or "2026-08-25" in answer2_lower or "august 25" in answer2_lower, (
            "Agent should answer with FedEx / delivery date using conversational memory from Turn 1!"
        )

    await asyncio.sleep(1.5)


@pytest.mark.asyncio
async def test_06_multi_turn_short_term_memory_product_coreference():
    """Test 6: Multi-Turn Conversation Memory for Product Catalog Inquiries"""
    print("\n" + "=" * 70)
    print("TEST 6: Multi-Turn Memory for Product Catalog (Logitech Mouse)")
    print("=" * 70)
    
    thread_id = "test-thread-product-memory-02"
    
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        # Turn 1: Inquire about Logitech MX Master 3S
        turn1_prompt = "What is the price of the Logitech MX Master 3S Wireless Mouse?"
        print(f"TURN 1 PROMPT: '{turn1_prompt}'")
        resp1 = await client.post(
            "/api/v1/agent/run",
            json={"prompt": turn1_prompt, "thread_id": thread_id, "max_iterations": 5}
        )
        assert resp1.status_code == 200
        data1 = resp1.json()
        print(f"TURN 1 ANSWER: {data1['final_answer']}")
        assert "99" in data1["final_answer"]
        
        await asyncio.sleep(2.0)
        
        # Turn 2: Follow-up using anaphoric pronoun "it"
        turn2_prompt = "Is it currently in stock, and what is its return policy window?"
        print(f"\nTURN 2 PROMPT: '{turn2_prompt}'")
        resp2 = await client.post(
            "/api/v1/agent/run",
            json={"prompt": turn2_prompt, "thread_id": thread_id, "max_iterations": 5}
        )
        assert resp2.status_code == 200
        data2 = resp2.json()
        print(f"TURN 2 ANSWER: {data2['final_answer']}")
        answer2_lower = data2["final_answer"].lower()
        assert "stock" in answer2_lower or "30" in answer2_lower, "Agent should address stock/return policy using thread memory."

    await asyncio.sleep(1.5)


@pytest.mark.asyncio
async def test_07_thread_isolation():
    """Test 7: Thread Isolation (Separate conversation contexts remain completely segregated)"""
    print("\n" + "=" * 70)
    print("TEST 7: Thread Isolation (Independent Thread Contexts)")
    print("=" * 70)
    
    thread_a = "thread-alice-1001"
    thread_b = "thread-bob-2002"
    
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        # Thread A talks about ORD-1001
        await client.post("/api/v1/agent/run", json={"prompt": "Track my order ORD-1001.", "thread_id": thread_a})
        await asyncio.sleep(2.0)
        
        # Thread B talks about ORD-2002
        await client.post("/api/v1/agent/run", json={"prompt": "Track my order ORD-2002.", "thread_id": thread_b})
        await asyncio.sleep(2.0)
        
        # Ask Thread A who the customer is
        resp_a = await client.post("/api/v1/agent/run", json={"prompt": "Who was the customer for my order?", "thread_id": thread_a})
        assert resp_a.status_code == 200, f"Thread A query failed: {resp_a.text}"
        data_a = resp_a.json()
        print(f"THREAD A ANSWER: {data_a['final_answer']}")
        
        await asyncio.sleep(2.0)
        
        # Ask Thread B who the customer is
        resp_b = await client.post("/api/v1/agent/run", json={"prompt": "Who was the customer for my order?", "thread_id": thread_b})
        assert resp_b.status_code == 200, f"Thread B query failed: {resp_b.text}"
        data_b = resp_b.json()
        print(f"THREAD B ANSWER: {data_b['final_answer']}")
        
        assert "alice" in data_a["final_answer"].lower(), "Thread A should remember Alice Johnson."
        assert "bob" in data_b["final_answer"].lower(), "Thread B should remember Bob Smith."

    await asyncio.sleep(1.5)


@pytest.mark.asyncio
async def test_08_thread_history_and_clear_api():
    """Test 8: Thread History Retrieval and Thread Reset / Clear API"""
    print("\n" + "=" * 70)
    print("TEST 8: Thread History & Clear API (GET & DELETE /threads/{thread_id})")
    print("=" * 70)
    
    thread_id = "test-thread-history-api"
    
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        # 1. Run a request to populate thread state
        await client.post(
            "/api/v1/agent/run",
            json={"prompt": "Check server time.", "thread_id": thread_id}
        )
        
        await asyncio.sleep(1.5)
        
        # 2. Retrieve thread history
        hist_resp = await client.get(f"/api/v1/agent/threads/{thread_id}")
        assert hist_resp.status_code == 200, f"History fetch failed: {hist_resp.text}"
        hist_data = hist_resp.json()
        print(f"FETCHED HISTORY: total_messages={hist_data['total_messages']}")
        assert hist_data["total_messages"] >= 2, "Thread history must contain messages."
        assert hist_data["thread_id"] == thread_id
        
        # 3. Clear the thread
        del_resp = await client.delete(f"/api/v1/agent/threads/{thread_id}")
        assert del_resp.status_code == 200, f"Thread clear failed: {del_resp.text}"
        del_data = del_resp.json()
        print(f"CLEAR RESPONSE: {del_data}")
        assert del_data["cleared"] is True

    await asyncio.sleep(1.5)


@pytest.mark.asyncio
async def test_09_tool_error_handling_and_recovery():
    """Test 9: Tool Error Handling & Graceful Recovery for Invalid Order/Product IDs"""
    print("\n" + "=" * 70)
    print("TEST 9: Tool Error Handling & Recovery (Unrecognized Order ID)")
    print("=" * 70)
    
    prompt = "Check the status of order ORD-9999."
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
        
        assert len(data["tool_calls"]) >= 1, "Agent should have attempted to call check_order_status."
        assert data["tool_calls"][0]["status"] == "error", "Tool call status should be recorded as 'error'."
        assert "ORD-9999" in data["final_answer"] or "not found" in data["final_answer"].lower() or "unable" in data["final_answer"].lower() or "invalid" in data["final_answer"].lower() or "valid" in data["final_answer"].lower(), (
            "Agent should explain that the order was not found."
        )

    await asyncio.sleep(1.5)


@pytest.mark.asyncio
async def test_10_max_iterations_infinite_loop_guard():
    """Test 10: Infinite Loop Guard (max_iterations=1 enforcement)"""
    print("\n" + "=" * 70)
    print("TEST 10: Max Iterations Infinite Loop Guard (max_iterations=1)")
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
