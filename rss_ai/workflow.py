import os
from typing import Dict, List

from rss_ai.error_handler import (  # Import the error handling functions and exceptions
    FeedFetchError,
    ParseError,
    handle_error,
)
from rss_ai.fetcher import get_recent_articles
from rss_ai.formatter import format_summary
from rss_ai.logger import setup_logger
from rss_ai.scraper import scrape_url  # Import the scrape_url function
from rss_ai.summarizer import (  # Import the function
    generate_cache_filename,
    generate_summary,
)

# Setup logger
logger = setup_logger("workflow", "workflow.log")


def process_feeds(feed_file: str, output_dir: str, days: int = 1):
    """
    Process RSS feeds, fetch recent articles, and write formatted summaries to output directory.

    Args:
        feed_file (str): Path to the YAML file containing RSS feed URLs.
        output_dir (str): Path to the directory where formatted summaries will be saved.
        days (int): Number of days to look back for recent articles. Defaults to 1.
    """
    logger.info(f"Starting feed processing with feed file: {feed_file}")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Get recent articles
    try:
        articles = get_recent_articles(feed_file, days)  # Attempt to fetch articles
    except Exception as e:
        handle_error(
            e, FeedFetchError, "Failed to fetch recent articles"
        )  # Handle fetch errors

    logger.info(f"Retrieved {len(articles)} recent articles")

    # Process each article
    for article in articles:
        process_article(article, output_dir)  # Call the new helper function

    logger.info("Finished processing feeds")


def process_article(article: Dict, output_dir: str):
    """
    Process a single article, scrape its content, and write the formatted summary to a file.

    Args:
        article (Dict): The article data.
        output_dir (str): Path to the directory where formatted summaries will be saved.
    """
    try:
        # Prepare article data
        article_data = {
            "title": article.get("title", "No title available"),
            "source": article.get("feed_name", "Unknown source"),
            "date": article.get("published_parsed", "No date available"),
            "url": article.get("link", "#"),
        }

        # For this example, we'll use the article's summary as our "AI-generated" summary
        # summary = article.get("summary", "No summary available")

        # Scrape the full content of the article
        try:
            full_content = scrape_url(article_data["url"])  # Get full content
        except Exception as e:
            handle_error(
                e, ParseError, f"Failed to scrape URL: {article_data['url']}"
            )  # Handle scraping errors

        summary = generate_summary(full_content)

        # Format the summary and include full content
        formatted_content = format_summary(article_data, summary)

        # Generate a filename using the generate_cache_filename function
        filename = generate_cache_filename(article_data["url"]) + ".md"
        file_path = os.path.join(output_dir, filename)

        # Write to file
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(formatted_content)

        logger.info(f"Wrote article summary to {file_path}")

    except Exception as e:
        handle_error(
            e,
            ParseError,
            f"Error processing article: {article.get('title', 'Unknown')}",
        )  # Handle article processing errors
