import asyncio
import httpx
import logging

# Set up logging for test runner
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("test_api")

# Import the FastAPI app
from day02.main import app
# Import mock customer feedback for sentiment analysis test
from day01.data import HYPERCHARGE_FEEDBACK

async def run_tests():
    logger.info("Initializing in-memory integration tests using httpx.AsyncClient...")

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        # 1. Test Health Endpoint
        logger.info("Testing Endpoint: GET /api/v1/health")
        resp = await client.get("/api/v1/health")
        assert resp.status_code == 200, f"Health check failed: {resp.status_code}"
        logger.info("Response: %s", resp.json())
        print("-" * 50)

        # 2. Test Chat Endpoint
        prompt = "Explain in one sentence why clean architecture prevents code rotting."
        logger.info("Testing Endpoint: POST /api/v1/chat")
        logger.info("Payload: %s", prompt)
        resp = await client.post("/api/v1/chat", json={"prompt": prompt})
        assert resp.status_code == 200, f"Chat failed: {resp.text}"
        data = resp.json()
        logger.info("Chat Response: %s", data["content"])
        logger.info("Token Usage: Prompt=%d, Completion=%d, Total=%d", 
                    data["prompt_tokens"], data["completion_tokens"], data["total_tokens"])
        logger.info("Estimated Cost: $%.8f", data["cost"])
        print("-" * 50)

        # 3. Test Sentiment Analysis Endpoint
        logger.info("Testing Endpoint: POST /api/v1/sentiment")
        logger.info("Feedback: %s...", HYPERCHARGE_FEEDBACK[:100])
        resp = await client.post("/api/v1/sentiment", json={"feedback": HYPERCHARGE_FEEDBACK})
        assert resp.status_code == 200, f"Sentiment analysis failed: {resp.text}"
        data = resp.json()
        analysis = data["analysis"]
        logger.info("Sentiment Analysis Result:")
        logger.info("  Product:         %s", analysis["product_name"])
        logger.info("  Sentiment:       %s", analysis["sentiment"])
        logger.info("  Rating:          %d/5", analysis["rating"])
        logger.info("  Pros:            %s", ", ".join(analysis["pros"]))
        logger.info("  Cons:            %s", ", ".join(analysis["cons"]))
        logger.info("  Refund Demanded? %s", "Yes" if analysis["requires_refund"] else "No")
        logger.info("Token Usage: Prompt=%d, Completion=%d, Total=%d", 
                    data["prompt_tokens"], data["completion_tokens"], data["total_tokens"])
        logger.info("Estimated Cost: $%.8f", data["cost"])
        print("-" * 50)

        # 4. Test Streaming Endpoint
        logger.info("Testing Endpoint: POST /api/v1/stream")
        stream_prompt = "Write a 3-line poem about async Python."
        logger.info("Payload: %s", stream_prompt)
        
        async with client.stream("POST", "/api/v1/stream", json={"prompt": stream_prompt}) as response:
            assert response.status_code == 200, f"Streaming failed: {response.status_code}"
            logger.info("Streaming Response:")
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    token = line[6:]
                    print(token, end="", flush=True)
            print()
        print("-" * 50)
        
        logger.info("All tests completed successfully!")

if __name__ == "__main__":
    asyncio.run(run_tests())
