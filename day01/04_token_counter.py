# This file demonstrates counting tokens of a prompt and calculating API costs.
# Pre-flight counting uses a local tokenizer model before sending to the API.
# Post-call details inspect the response metadata to get exact token usage.
# Then calculates the total cost locally

from litellm import token_counter, completion_cost
from utils import execute_completion, handle_litellm_exceptions
import config

def main():
    model = config.MODEL_ID
    prompt = (
        "Write a comprehensive explanation of how indices speed up SELECT queries in databases, "
        "including B-Tree structure and the difference between clustered and non-clustered indices."
    )

    with handle_litellm_exceptions("token_counter"):
        messages = [{"role": "user", "content": prompt}]
        tokens = token_counter(model=model, messages=messages)
        print(f"Pre-flight Token Count: {tokens}")

    response = execute_completion(
        action_name="token_cost_check",
        model=model,
        messages=messages
    )
    
    usage = response.usage
    print(f"Prompt Tokens:     {usage.prompt_tokens}")
    print(f"Completion Tokens: {usage.completion_tokens}")
    print(f"Total Tokens:      {usage.total_tokens}")
    
    cost = completion_cost(completion_response=response)
    if cost is not None:
        print(f"Estimated Cost:    ${cost:.8f}")
    else:
        print("Estimated Cost:    Unknown")

if __name__ == "__main__":
    main()
