import asyncio
import os
import httpx
import logging

os.environ["MODEL_ID"] = "gemini/gemini-3.1-flash-lite"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("test_langgraph_agent")

from ai_service_platform.main import app


async def run_langgraph_agent_tests():
    logger.info("Initializing LangGraph Autonomous Agent Integration Test Suite...")

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        
        # 1. Single Tool Calling Test (Discount Calculation)
        logger.info("1. Testing Single Tool Calling: Discount Calculation (VIP50)...")
        resp1 = await client.post(
            "/api/v1/agent/run",
            json={
                "prompt": "I have an order of $200 and want to use promo code VIP50. How much is my final price?",
                "max_iterations": 5
            }
        )
        assert resp1.status_code == 200, f"Agent run failed: {resp1.text}"
        data1 = resp1.json()
        
        logger.info("Final Answer: %s", data1["final_answer"])
        logger.info("Tool Calls Executed: %s", data1["tool_calls"])
        logger.info("Iterations: %d | Tokens: %d | Cost: $%0.6f", 
                    data1["iterations"], data1["total_tokens"], data1["cost"])
        
        assert len(data1["tool_calls"]) >= 1, "Agent should have called calculate_order_discount!"
        assert data1["tool_calls"][0]["tool_name"] == "calculate_order_discount"
        assert data1["tool_calls"][0]["status"] == "success"
        assert "100" in data1["final_answer"] or "50%" in data1["final_answer"] or "100.0" in data1["final_answer"], \
            "Agent should mention the $100 final price."
        print("=" * 60)

        # 2. Direct Answer Test (No Tools Required)
        logger.info("2. Testing Direct Answer without Tool Execution...")
        resp2 = await client.post(
            "/api/v1/agent/run",
            json={
                "prompt": "What is the capital of France? Answer in one short sentence.",
                "max_iterations": 5
            }
        )
        assert resp2.status_code == 200, f"Direct answer failed: {resp2.text}"
        data2 = resp2.json()
        
        logger.info("Final Answer: %s", data2["final_answer"])
        logger.info("Tool Calls: %s (should be empty)", data2["tool_calls"])
        assert len(data2["tool_calls"]) == 0, "Agent should NOT have invoked any tools for simple general knowledge."
        assert "Paris" in data2["final_answer"], "Agent should correctly answer Paris."
        print("=" * 60)

        # 3. Tool Error Handling & Recovery Test
        logger.info("3. Testing Tool Error Handling with unrecognized service...")
        resp3 = await client.post(
            "/api/v1/agent/run",
            json={
                "prompt": "Check the health status of legacy_mainframe_service.",
                "max_iterations": 5
            }
        )
        assert resp3.status_code == 200, f"Error recovery test failed: {resp3.text}"
        data3 = resp3.json()
        
        logger.info("Final Answer: %s", data3["final_answer"])
        logger.info("Tool Calls: %s", data3["tool_calls"])
        assert len(data3["tool_calls"]) >= 1, "Agent should have attempted to call check_service_status."
        assert data3["tool_calls"][0]["status"] == "error", "Tool call status should be recorded as 'error'."
        print("=" * 60)

        # 4. Infinite Loop / Max Iterations Guard Test
        logger.info("4. Testing Max Iterations Guard (max_iterations=1)...")
        resp4 = await client.post(
            "/api/v1/agent/run",
            json={
                "prompt": "Calculate discount for $500 with code VIP50, then check payment_service status.",
                "max_iterations": 1
            }
        )
        assert resp4.status_code == 200, f"Max iteration test failed: {resp4.text}"
        data4 = resp4.json()
        logger.info("Max Iterations Enforced: %d iterations executed.", data4["iterations"])
        assert data4["iterations"] <= 1, f"Agent exceeded max_iterations limit! Iterations: {data4['iterations']}"
        print("=" * 60)

        # 5. Multi-Turn Ordering & Cost Verification
        logger.info("5. Testing Multi-Turn Ordering & Multi-Turn Cost Accumulation...")
        resp5 = await client.post(
            "/api/v1/agent/run",
            json={
                "prompt": "First check the server time, and then check the status of payment_service.",
                "max_iterations": 5
            }
        )
        assert resp5.status_code == 200, f"Multi-turn test failed: {resp5.text}"
        data5 = resp5.json()
        
        logger.info("Final Answer: %s", data5["final_answer"])
        logger.info("Tool Execution Order:")
        for idx, tc in enumerate(data5["tool_calls"]):
            logger.info("  Step %d: %s -> Output: %s", idx + 1, tc["tool_name"], tc["tool_output"])
        
        assert data5["total_tokens"] > 0, "Accumulated tokens should be greater than 0."
        assert data5["cost"] >= 0.0, "Accumulated cost should be calculated."
        print("=" * 60)

        logger.info("ALL LANGGRAPH AGENT INTEGRATION TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    asyncio.run(run_langgraph_agent_tests())
