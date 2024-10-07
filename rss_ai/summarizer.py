import os
from typing import Optional

from openai import OpenAI

from rss_ai.logger import setup_logger  # Import the logger setup function

logger = setup_logger("summarizer", "summarizer.log")  # Initialize the logger

client = OpenAI()


def get_cached_summary(article_url: str) -> Optional[str]:
    # TODO: Implement caching logic
    return None


def cache_summary(article_url: str, summary: str) -> None:
    # TODO: Implement caching logic
    pass


def generate_summary(article_text: str) -> str:
    client = OpenAI()  # Assuming openai is already initialized
    completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": f"Please summarize the following article:\n\n{article_text}",
            }
        ],
        model="gpt-3.5-turbo",  # Use the specified model
    )
    summary_content = completion.choices[0].message.content  # Store the summary content
    logger.info(summary_content)  # Log the generated summary
    return summary_content  # Updated to match new API


def summarize_article(article_url: str, article_text: str) -> str:
    # Check for cached summary
    cached_summary = get_cached_summary(article_url)
    if cached_summary:
        return cached_summary

    # Generate new summary
    summary = generate_summary(article_text)

    # Cache the summary
    cache_summary(article_url, summary)

    return summary
