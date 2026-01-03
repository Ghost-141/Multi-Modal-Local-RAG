"""Prompts originally used in the exploratory notebook."""

# Text/table summarization prompt from the notebook (prompts.py)
TEXT_SUMMARY_PROMPT = """
You are an assistant that returns only the concise summary of the provided table or text.

Rules:
- Respond with the summary only; no prefaces, labels, meta commentary, or extra sentences.
- Respond with the summary of the table given in html format.
- Do not mention the act of summarizing or refer to the source (e.g., no "Here is a summary" or "From the table...").

Table or text chunk:
{element}
"""

# Image description prompt from the notebook (prompts.py)
IMAGE_DESCRIPTION_PROMPT = """
Describe the image in detail.
Be specific about notable elements such as architecture, graphs, or plots like bar charts.
Respond with the description only, with no preface.
"""
