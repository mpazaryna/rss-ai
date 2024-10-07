import os
from typing import Dict, List

from rss_ai.formatter import format_summary
from rss_ai.logger import setup_logger
from rss_ai.rss_fetcher import get_recent_articles
from rss_ai.summarizer import generate_cache_filename  # Import the function

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
    articles = get_recent_articles(feed_file, days)
    logger.info(f"Retrieved {len(articles)} recent articles")

    # Process each article
    for article in articles:
        try:
            # Prepare article data
            article_data = {
                "title": article.get("title", "No title available"),
                "source": article.get("feed_name", "Unknown source"),
                "date": article.get("published_parsed", "No date available"),
                "url": article.get("link", "#"),
            }

            # For this example, we'll use the article's summary as our "AI-generated" summary
            summary = article.get("summary", "No summary available")

            # Format the summary
            formatted_content = format_summary(article_data, summary)

            # Generate a filename using the generate_cache_filename function
            filename = generate_cache_filename(article_data["url"]) + ".md"
            file_path = os.path.join(output_dir, filename)

            # Write to file
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(formatted_content)

            logger.info(f"Wrote article summary to {file_path}")

        except Exception as e:
            logger.error(
                f"Error processing article {article.get('title', 'Unknown')}: {str(e)}"
            )
            continue

    logger.info("Finished processing feeds")
