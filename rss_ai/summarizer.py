import os
from typing import Optional

from openai import OpenAI

from rss_ai.error_handler import (  # Import error handling components
    RSSAIError,
    handle_error,
)
from rss_ai.logger import setup_logger  # Import the logger setup function

logger = setup_logger("summarizer", "summarizer.log")  # Initialize the logger

client = OpenAI()

CACHE_DIR = "data/output"  # Define the cache directory


def get_cached_summary(article_url: str) -> Optional[str]:
    try:
        # Construct the cache file path based on the article URL
        cache_file_path = os.path.join(
            CACHE_DIR, f"{article_url.replace('/', '_')}.txt"
        )
        if os.path.exists(cache_file_path):
            with open(cache_file_path, "r") as file:
                return file.read()  # Return the cached summary
        return None
    except Exception as e:
        handle_error(
            e, RSSAIError, f"Error retrieving cached summary for: {article_url}"
        )


def generate_cache_filename(article_url: str) -> str:
    from urllib.parse import urlparse  # Importing urlparse to parse the URL

    parsed_url = urlparse(article_url)  # Parse the URL
    domain = parsed_url.netloc.replace(
        ".", "_"
    )  # Replace dots in domain with underscores
    last_part = parsed_url.path.split("/")[-1].split(".")[
        0
    ]  # Get the last part of the URL without extension
    return f"{domain}_{last_part}"  # Return the formatted cache filename


def cache_summary(article_url: str, summary: str) -> None:
    try:
        # Ensure the cache directory exists
        os.makedirs(CACHE_DIR, exist_ok=True)
        # Construct the cache file path based on the article URL using the new helper method
        cache_file_path = os.path.join(
            CACHE_DIR, f"{generate_cache_filename(article_url)}.txt"
        )
        with open(cache_file_path, "w") as file:
            file.write(summary)  # Write the summary to the cache file
    except Exception as e:
        handle_error(e, RSSAIError, f"Error caching summary for: {article_url}")


def generate_summary(article_text: str) -> str:
    try:
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
        summary_content = completion.choices[
            0
        ].message.content  # Store the summary content
        logger.info(summary_content)  # Log the generated summary
        return summary_content  # Updated to match new API
    except Exception as e:
        handle_error(e, RSSAIError, "Error generating summary")


def summarize_article(article_url: str, article_text: str) -> str:
    try:
        # Check for cached summary
        cached_summary = get_cached_summary(article_url)
        if cached_summary:
            logger.info(
                f"Cached summary found for URL: {article_url}"
            )  # Log if cached summary is found
            return cached_summary

        # Generate new summary
        summary = generate_summary(article_text)

        # Cache the summary
        cache_summary(article_url, summary)

        return summary
    except Exception as e:
        handle_error(e, RSSAIError, f"Error summarizing article: {article_url}")
