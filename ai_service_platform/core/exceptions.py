class LLMException(Exception):
    """Base exception for all LLM client errors."""
    def __init__(self, message: str, original_exception: Exception | None = None):
        super().__init__(message)
        self.original_exception = original_exception


class LLMAuthenticationError(LLMException):
    """Raised when authentication with the provider fails (e.g. invalid API key)."""
    pass


class LLMRateLimitError(LLMException):
    """Raised when the provider rate limit is exceeded."""
    pass


class LLMTPMRateLimitError(LLMRateLimitError):
    """Raised when client-side or gateway Token-Per-Minute (TPM) budget is exceeded."""
    def __init__(self, message: str, retry_after_seconds: float = 60.0, original_exception: Exception | None = None):
        super().__init__(message, original_exception)
        self.retry_after_seconds = retry_after_seconds


class LLMConnectionError(LLMException):
    """Raised when there is a network connectivity issue or the service is unavailable."""
    pass


class LLMTimeoutError(LLMException):
    """Raised when the request to the provider times out."""
    pass


class LLMAPIError(LLMException):
    """Raised for general or unknown provider API errors."""
    pass


class AllProvidersUnavailableError(LLMException):
    """Raised when all primary and fallback providers in the router have failed or are in cooldown."""
    def __init__(self, message: str, tried_models: list[str] | None = None, original_exception: Exception | None = None):
        super().__init__(message, original_exception)
        self.tried_models = tried_models or []

