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


class LLMConnectionError(LLMException):
    """Raised when there is a network connectivity issue or the service is unavailable."""
    pass


class LLMTimeoutError(LLMException):
    """Raised when the request to the provider times out."""
    pass


class LLMAPIError(LLMException):
    """Raised for general or unknown provider API errors."""
    pass
