import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Model identifier for LiteLLM routing (e.g. "gemini/gemini-flash-latest", "openai/gpt-4o", "anthropic/claude-3-5-sonnet")
MODEL_ID: str = os.getenv("MODEL_ID", "gemini/gemini-flash-latest")

# Default generation parameters
STRUCTURED_TEMPERATURE: float = 0.1  # Lower temperature is recommended for structured output schema tasks
TOOL_TEMPERATURE: float = 0.0        # Deterministic temperature for tool invocation / function calling
DEFAULT_RETRIES: int = 3             # Number of retries for transient API errors
