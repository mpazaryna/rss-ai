"""
RSS Fetcher Module

This module provides functionality to fetch and process RSS feeds.
It includes functions to read feed URLs from a YAML file, fetch RSS content,
and filter articles based on recency.

Functions:
    load_feeds(file_path: str) -> dict
    fetch_rss(url: str) -> List[Dict]
    filter_recent_articles(articles: List[Dict], days: int) -> List[Dict]
    get_recent_articles(file_path: str, days: int) -> List[Dict]

Dependencies:
    - PyYAML
    - feedparser
    - pytz
    - datetime
"""

import logging  # Add this import at the top of your file
from datetime import datetime, timedelta
from typing import Dict, List

import feedparser
import pytz
import yaml

from rss_ai.logger import setup_logger

# Setup logger
logger = setup_logger("rss_fetcher", "rss_fetcher.log")


def load_feeds(file_path: str) -> dict:
    """
    Load RSS feed URLs from a YAML file.

    Args:
        file_path (str): Path to the YAML file containing RSS feed URLs.

    Returns:
        dict: A dictionary of feed names and their corresponding URLs.

    Raises:
        FileNotFoundError: If the specified file is not found.
        yaml.YAMLError: If there's an error parsing the YAML file.
    """
    try:
        with open(file_path, "r") as file:
            data = yaml.safe_load(file)
        feeds = data.get("feeds", {})
        logger.info(f"Successfully loaded {len(feeds)} feeds from {file_path}")
        return feeds
    except FileNotFoundError:
        logger.error(f"Feed file not found: {file_path}")
        raise FileNotFoundError(f"Feed file not found: {file_path}")
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file: {e}")
        raise yaml.YAMLError(f"Error parsing YAML file: {e}")


def fetch_rss(url: str) -> List[Dict]:
    """
    Fetch and parse an RSS feed.

    Args:
        url (str): URL of the RSS feed.

    Returns:
        List[Dict]: A list of dictionaries, each containing article metadata.

    Raises:
        Exception: If there's an error fetching or parsing the RSS feed.
    """
    try:
        # Use feedparser to fetch and parse the RSS feed directly
        feed_data = feedparser.parse(url)
        if feed_data.bozo:
            raise ValueError(f"Error parsing feed: {feed_data.bozo_exception}")

        # Extract articles from the feed
        articles = []
        for entry in feed_data.entries:
            articles.append(
                {
                    "title": entry.title,
                    "link": entry.link,
                    "published_parsed": entry.published_parsed,
                    # Add any other fields you need from the entry
                }
            )
        return articles

    except Exception as e:
        logging.error(f"Error fetching RSS feed from {url}: {e}")
        # Handle the error appropriately
    # ... existing code ...


def filter_recent_articles(articles: List[Dict], days: int) -> List[Dict]:
    """
    Filter articles based on recency.

    Args:
        articles (List[Dict]): List of article dictionaries.
        days (int): Number of days to look back.

    Returns:
        List[Dict]: Filtered list of recent articles.
    """
    cutoff_date = datetime.now(pytz.utc) - timedelta(days=days)
    recent_articles = [
        article
        for article in articles
        if "published_parsed" in article
        and datetime(*article["published_parsed"][:6], tzinfo=pytz.utc) > cutoff_date
    ]
    logger.info(
        f"Filtered {len(recent_articles)} recent articles out of {len(articles)} total articles"
    )
    return recent_articles


def get_recent_articles(file_path: str, days: int) -> List[Dict]:
    """
    Get recent articles from all feeds specified in the YAML file.

    Args:
        file_path (str): Path to the YAML file containing RSS feed URLs.
        days (int): Number of days to look back for recent articles.

    Returns:
        List[Dict]: A list of recent articles from all feeds.

    Raises:
        Exception: If there's an error processing any of the feeds.
    """
    feeds = load_feeds(file_path)
    all_articles = []

    for name, url in feeds.items():
        try:
            logger.info(f"Processing feed: {name}")
            articles = fetch_rss(url)
            recent_articles = filter_recent_articles(articles, days)
            for article in recent_articles:
                article["feed_name"] = name
            all_articles.extend(recent_articles)
        except Exception as e:
            logger.error(f"Error processing feed '{name}': {str(e)}")
            # Instead of raising an exception, we'll continue processing other feeds
            continue

    logger.info(
        f"Retrieved a total of {len(all_articles)} recent articles from all feeds"
    )
    return all_articles
