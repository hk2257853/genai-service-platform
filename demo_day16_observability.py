"""Day 16: Live AI Observability & Tracing Showcase.

Demonstrates:
1. Non-Streaming Call with automatic token & cost telemetry
2. Streaming Call measuring real-time Time-to-First-Token (TTFT)
3. Direct queries to the /api/v1/observability REST endpoints:
   - /api/v1/observability/costs (Cumulative platform billing & token metrics)
   - /api/v1/observability/costs/recent (Granular per-call traces)
   - /api/v1/observability/tracing/status (LangSmith integration status)
"""

import asyncio
import json
import httpx
from ai_service_platform.main import app
from ai_service_platform.core.cost_tracker import cost_tracker


def header(title: str):
    print("\n" + "=" * 70)
    print(f" {title.upper()}")
    print("=" * 70)


async def main():
    # 0. Clean slate for demo
    cost_tracker.reset()

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        # ---------------------------------------------------------
        # 1. Tracing Status Check
        # ---------------------------------------------------------
        header("1. LangSmith Tracing Status Check")
        status_resp = await client.get("/api/v1/observability/tracing/status")
        print(f"GET /api/v1/observability/tracing/status -> Status {status_resp.status_code}")
        print(json.dumps(status_resp.json(), indent=2))

        # ---------------------------------------------------------
        # 2. Non-Streaming LLM Invocations (Chat & Structured Sentiment)
        # ---------------------------------------------------------
        header("2. Non-Streaming Invocation: Text Generation")
        chat_prompt = "Explain in 2 sentences why Time-to-First-Token (TTFT) matters for AI UX."
        print(f"[INPUT PROMPT]: {chat_prompt}\n")
        
        chat_resp = await client.post("/api/v1/chat", json={"prompt": chat_prompt})
        chat_data = chat_resp.json()
        print(f"[OUTPUT]:\n{chat_data['content']}\n")
        print(f"[TELEMETRY]:")
        print(f"  • Prompt Tokens:     {chat_data['prompt_tokens']}")
        print(f"  • Completion Tokens: {chat_data['completion_tokens']}")
        print(f"  • Total Tokens:      {chat_data['total_tokens']}")
        print(f"  • USD Cost:          ${chat_data['cost']:.8f}")

        header("3. Non-Streaming Invocation: Structured Sentiment Analysis")
        feedback = "The battery charges my phone from 0 to 80% in 25 minutes, but gets quite warm. Overall love it!"
        print(f"[INPUT FEEDBACK]: {feedback}\n")
        sentiment_resp = await client.post("/api/v1/sentiment", json={"feedback": feedback})
        sentiment_data = sentiment_resp.json()
        print(f"[OUTPUT STRUCTURED JSON]:")
        print(json.dumps(sentiment_data["analysis"], indent=2))
        print(f"\n[TELEMETRY]:")
        print(f"  • Prompt Tokens:     {sentiment_data['prompt_tokens']}")
        print(f"  • Completion Tokens: {sentiment_data['completion_tokens']}")
        print(f"  • USD Cost:          ${sentiment_data['cost']:.8f}")

        # ---------------------------------------------------------
        # 3. Streaming Invocation with TTFT Measurement
        # ---------------------------------------------------------
        header("4. Streaming Invocation with Time-to-First-Token (TTFT)")
        stream_prompt = "Write a haiku about microservice observability."
        print(f"[INPUT PROMPT]: {stream_prompt}\n")
        print("[STREAMING TOKENS LIVE]: ", end="", flush=True)

        async with client.stream("POST", "/api/v1/stream", json={"prompt": stream_prompt}) as stream_resp:
            metrics_payload = None
            async for line in stream_resp.aiter_lines():
                if not line:
                    continue
                if line.startswith("data: "):
                    token = line[len("data: "):]
                    print(token, end="", flush=True)
                elif line.startswith("event: metrics"):
                    # Next line will be data: with json
                    continue
                elif line.startswith("data: {") and "ttft_ms" in line:
                    metrics_payload = json.loads(line[len("data: "):])

        print("\n")
        if metrics_payload:
            print("[TRAILING SSE METRICS EVENT]:")
            print(f"  • Time-to-First-Token (TTFT): {metrics_payload['ttft_ms']} ms")
            print(f"  • Total Duration:             {metrics_payload['total_duration_ms']} ms")
            print(f"  • Stream Chunks Received:     {metrics_payload['total_chunks']}")
            print(f"  • Input / Output Tokens:      {metrics_payload['prompt_tokens']} in / {metrics_payload['completion_tokens']} out")
            print(f"  • Stream USD Cost:            ${metrics_payload['cost_usd']:.8f}")

        # ---------------------------------------------------------
        # 4. Observability Dashboard Query: /api/v1/observability/costs
        # ---------------------------------------------------------
        header("5. Querying Platform-Wide Cost & Telemetry Dashboard")
        cost_resp = await client.get("/api/v1/observability/costs")
        cost_summary = cost_resp.json()
        print(f"GET /api/v1/observability/costs -> 200 OK\n")
        print(f"┌{'─' * 45}┬{'─' * 22}┐")
        print(f"│ Metric                                      │ Value                │")
        print(f"├{'─' * 45}┼{'─' * 22}┤")
        print(f"│ Total LLM Invocations                       │ {cost_summary['total_calls']:<20} │")
        print(f"│ Total Prompt Tokens                         │ {cost_summary['total_prompt_tokens']:<20} │")
        print(f"│ Total Completion Tokens                     │ {cost_summary['total_completion_tokens']:<20} │")
        print(f"│ Total Tokens Consumed                       │ {cost_summary['total_tokens']:<20} │")
        print(f"│ Total USD Spend                             │ ${cost_summary['total_cost_usd']:<19.8f} │")
        print(f"│ Average Total Latency                       │ {cost_summary['avg_latency_ms']} ms{' ' * (17 - len(str(cost_summary['avg_latency_ms'])))}│")
        print(f"│ Average Time-to-First-Token (TTFT)          │ {cost_summary['avg_ttft_ms']} ms{' ' * (17 - len(str(cost_summary['avg_ttft_ms'])))}│")
        print(f"└{'─' * 45}┴{'─' * 22}┘")

        print("\n[PER-MODEL BREAKDOWN]:")
        for model_name, model_stats in cost_summary["per_model"].items():
            print(f"  • Model: {model_name}")
            print(f"      Invocations: {model_stats['total_calls']}")
            print(f"      Tokens:      {model_stats['total_tokens']} ({model_stats['prompt_tokens']} in / {model_stats['completion_tokens']} out)")
            print(f"      Total Spend: ${model_stats['total_cost_usd']:.8f}")
            print(f"      Avg Latency: {model_stats['avg_latency_ms']:.1f} ms")

        # ---------------------------------------------------------
        # 5. Granular Recent Trace Records: /api/v1/observability/costs/recent
        # ---------------------------------------------------------
        header("6. Querying Granular Call Traces (/api/v1/observability/costs/recent)")
        recent_resp = await client.get("/api/v1/observability/costs/recent?limit=5")
        recent_calls = recent_resp.json()
        print(f"GET /api/v1/observability/costs/recent?limit=5 -> {len(recent_calls)} traces found\n")
        
        for i, call in enumerate(recent_calls, 1):
            stream_tag = f"[STREAM | TTFT: {call['ttft_ms']} ms]" if call['stream'] else "[NON-STREAM]"
            print(f"Trace #{i}: {call['action_name']} {stream_tag}")
            print(f"  Call ID:    {call['call_id']}")
            print(f"  Timestamp:  {call['timestamp']}")
            print(f"  Tokens:     {call['total_tokens']} (prompt: {call['prompt_tokens']}, comp: {call['completion_tokens']})")
            print(f"  Latency:    {call['latency_ms']:.1f} ms")
            print(f"  Cost:       ${call['cost_usd']:.8f}")
            print(f"  Status:     {call['status']}")
            print("-" * 50)


if __name__ == "__main__":
    asyncio.run(main())

