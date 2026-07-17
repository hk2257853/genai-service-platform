# This file is for basic text generation and streaming.
# Streaming sends responses chunk-by-chunk using Server-Sent Events (SSE).

from utils import execute_completion, execute_completion_stream
import config

def main():
    model = config.MODEL_ID
    prompt = "Explain why API design is important for SDEs in one paragraph."
    
    response = execute_completion(
        action_name="basic_gen",
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    print(response.choices[0].message.content)
    print()

    stream_prompt = "Write a short 3-line poem about a database query finding its target index."
    response_stream = execute_completion_stream(
        action_name="stream_gen",
        model=model,
        messages=[{"role": "user", "content": stream_prompt}]
    )
    for chunk in response_stream:
        content = chunk.choices[0].delta.content
        if content:
            print(content, end="", flush=True)
    print()

if __name__ == "__main__":
    main()
