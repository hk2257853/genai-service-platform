# This file is for structured output.
# Temperature helps for this.
# Schema validation we are doing just for a fail-safe circuit rail.

import json
from pydantic import BaseModel, Field
from utils import execute_completion
import config
from day01.data import HYPERCHARGE_FEEDBACK

class ProductSentimentAnalysis(BaseModel):
    product_name: str = Field(description="The name of the product mentioned.")
    sentiment: str = Field(description="The sentiment of the user. Options: Positive, Negative, Neutral.")
    rating: int = Field(description="An estimated star rating from 1 to 5 based on the text.")
    pros: list[str] = Field(description="Key strengths/features liked by the customer.")
    cons: list[str] = Field(description="Key complaints or pain points mentioned.")
    requires_refund: bool = Field(description="True if the user is demanding a refund, else False.")

def main():
    model = config.MODEL_ID
    feedback_text = HYPERCHARGE_FEEDBACK

    # Executes the API completion call wrapped in the centralized exception handler
    response = execute_completion(
        action_name="structured_output",
        model=model,
        messages=[{"role": "user", "content": f"Analyze this customer feedback:\n\n{feedback_text}"}],
        response_format=ProductSentimentAnalysis,
        temperature=config.STRUCTURED_TEMPERATURE
    )

    raw_json_str = response.choices[0].message.content

    # Catch JSON decoding errors locally since output validation is script-specific
    try:
        analysis = ProductSentimentAnalysis.model_validate_json(raw_json_str)
    except json.JSONDecodeError as e:
        print(f"[FATAL] Failed to decode JSON returned by model: {e}")
        raise
        
    print(f"Product:         {analysis.product_name}")
    print(f"Sentiment:       {analysis.sentiment}")
    print(f"Rating:          {analysis.rating}/5")
    print(f"Pros:            {', '.join(analysis.pros)}")
    print(f"Cons:            {', '.join(analysis.cons)}")
    print(f"Refund Claimed?  {'Yes' if analysis.requires_refund else 'No'}")

if __name__ == "__main__":
    main()
