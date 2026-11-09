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

# Database configuration
DB_HOST: str = os.getenv("DB_HOST", "127.0.0.1")
DB_PORT: int = int(os.getenv("DB_PORT", "5432"))
DB_NAME: str = os.getenv("DB_NAME", "ai_project")
DB_USER: str = os.getenv("DB_USER", "postgres")
DB_PASSWORD: str = os.getenv("DB_PASSWORD", "pass")

# Observability & Tracing Configuration (Day 16)
LANGCHAIN_TRACING_V2: bool = os.getenv("LANGCHAIN_TRACING_V2", "false").strip().lower() in ("true", "1", "yes")
LANGCHAIN_API_KEY: str | None = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_PROJECT: str = os.getenv("LANGCHAIN_PROJECT", "ai-service-platform")
LANGCHAIN_ENDPOINT: str = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")

# Production AI Patterns Configuration (Day 17)
# 1. Semantic Caching
SEMANTIC_CACHE_ENABLED: bool = os.getenv("SEMANTIC_CACHE_ENABLED", "true").strip().lower() in ("true", "1", "yes")
SEMANTIC_CACHE_THRESHOLD: float = float(os.getenv("SEMANTIC_CACHE_THRESHOLD", "0.92"))
SEMANTIC_CACHE_TTL_SECONDS: int = int(os.getenv("SEMANTIC_CACHE_TTL_SECONDS", "3600"))
SEMANTIC_CACHE_MAX_ENTRIES: int = int(os.getenv("SEMANTIC_CACHE_MAX_ENTRIES", "1000"))

# 2. Token-Per-Minute (TPM) Budgeting
TPM_LIMIT: int = int(os.getenv("TPM_LIMIT", "100000"))

# 3. Fallback Model Routing & LiteLLM Cooldown / Circuit Breakers
FALLBACK_MODELS: list[str] = [
    m.strip() for m in os.getenv("FALLBACK_MODELS", "openai/gpt-4o-mini,anthropic/claude-3-5-haiku").split(",") if m.strip()
]
ROUTER_ALLOWED_FAILS: int = int(os.getenv("ROUTER_ALLOWED_FAILS", "3"))
ROUTER_COOLDOWN_TIME: int = int(os.getenv("ROUTER_COOLDOWN_TIME", "30"))

