from pydantic import BaseModel, Field


# Structured output validation schemas
class ProductSentimentAnalysis(BaseModel):
    """Pydantic schema for structured output sentiment analysis."""
    product_name: str = Field(description="The name of the product mentioned.")
    sentiment: str = Field(description="The sentiment of the user. Options: Positive, Negative, Neutral.")
    rating: int = Field(description="An estimated star rating from 1 to 5 based on the text.")
    pros: list[str] = Field(description="Key strengths/features liked by the customer.")
    cons: list[str] = Field(description="Key complaints or pain points mentioned.")
    requires_refund: bool = Field(description="True if the user is demanding a refund, else False.")


# API Request schemas
class ChatRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="Prompt message to send to the LLM.")


class SentimentRequest(BaseModel):
    feedback: str = Field(..., min_length=5, description="Feedback text to perform sentiment analysis on.")


# API Response schemas
class LLMResponse(BaseModel):
    """Standardized response model for general generation requests."""
    content: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost: float


class SentimentAnalysisResponse(BaseModel):
    """Standardized response model for structured sentiment analysis."""
    analysis: ProductSentimentAnalysis
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost: float
