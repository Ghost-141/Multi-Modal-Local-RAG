"""Primary QA prompt for the backend plus notebook prompts exports."""

PROMPT = """Use the provided context to answer the question truthfully.
If the answer is not contained in the context, respond with 'I do not know.'

Context:
{context}

Question: {question}
"""

# Re-export notebook prompts so callers can access all prompts from one place.
from backend.system_prompts.notebook_prompts import (
    TEXT_SUMMARY_PROMPT,
    IMAGE_DESCRIPTION_PROMPT,
)

__all__ = ["PROMPT", "TEXT_SUMMARY_PROMPT", "IMAGE_DESCRIPTION_PROMPT"]
