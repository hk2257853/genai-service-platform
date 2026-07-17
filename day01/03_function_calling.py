# This file is for function calling / tool use.
# Tool definitions are passed to the model, which decides when to call them.
# The actual execution happens locally, and results are sent back to the model.
# Same concept used in MCP

import json
from utils import execute_completion
import config
from day01.data import MOCK_PRODUCT_DB, MOCK_TICKET_ID

def get_product_stock(product_id: str) -> int:
    print(f"[Tool Call] get_product_stock({product_id})")
    return MOCK_PRODUCT_DB.get(product_id, -1)


def create_support_ticket(product_id: str, issue_description: str) -> str:
    print(f"[Tool Call] create_support_ticket({product_id}, {issue_description})")
    return f"Success: Created support ticket {MOCK_TICKET_ID} for product {product_id}."

# Registry mapping tool names to python callables (eliminates fragile if-else branches)
TOOL_MAP = {
    "get_product_stock": get_product_stock,
    "create_support_ticket": create_support_ticket
}

# LiteLLM translates this standard schema to the provider-specific layout at execution time.
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_product_stock",
            "description": "Get the current stock quantity for a given product ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {
                        "type": "string",
                        "description": "The unique product code, e.g. PROD-101, PROD-202"
                    }
                },
                "required": ["product_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_support_ticket",
            "description": "Create a customer support ticket for a product issue.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {
                        "type": "string",
                        "description": "The product code experiencing issues."
                    },
                    "issue_description": {
                        "type": "string",
                        "description": "Descriptive summary of the issue."
                    }
                },
                "required": ["product_id", "issue_description"]
            }
        }
    }
]

def main():
    model = config.MODEL_ID
    user_prompt = "Check stock for PROD-202. If it's out of stock, create a support ticket stating 'Out of stock in database'."
    
    messages = [
        {"role": "user", "content": user_prompt}
    ]
    
    MAX_TURNS = 5
    for turn in range(MAX_TURNS):
        response = execute_completion(
            action_name=f"tool_turn_{turn + 1}",
            model=model,
            messages=messages,
            tools=TOOLS_SCHEMA,
            temperature=config.TOOL_TEMPERATURE
        )
        
        message = response.choices[0].message
        messages.append(message)
        
        # The LLM decides which tool to call and with what arguments
        if message.tool_calls:
            for tool_call in message.tool_calls:
                func_name = tool_call.function.name
                func_args = json.loads(tool_call.function.arguments)
                
                if func_name in TOOL_MAP:
                    result = TOOL_MAP[func_name](**func_args)
                else:
                    result = f"Error: Tool '{func_name}' is not registered."
                
                messages.append({
                    "role": "tool",
                    "name": func_name,
                    "tool_call_id": tool_call.id,
                    "content": json.dumps({"result": result})
                })
        else:
            print(message.content)
            break

if __name__ == "__main__":
    main()
