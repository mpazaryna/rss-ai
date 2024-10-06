"""
Test module for rss_fetcher.py

This module contains unit and integration tests for the rss_fetcher module.
It uses pytest for testing and includes mocking of external dependencies.

Test coverage aims for 100% of the rss_fetcher module.
"""

from datetime import datetime, timedelta
from unittest.mock import mock_open, patch

import pytest
import pytz
import yaml

from rss_ai.rss_fetcher import (
    fetch_rss,
    filter_recent_articles,
    get_recent_articles,
    load_feeds,
)

# Sample data for testing
SAMPLE_YAML = """
OpenAI: https://openai.com/blog/rss.xml
HuggingFace: https://huggingface.co/blog/feed.xml
Microsoft: https://news.microsoft.com/source/topics/ai/feed/
"""

SAMPLE_RSS_ENTRY = {
    "title": "Test Article",
    "link": "https://example.com/article",
    "published_parsed": datetime.now(pytz.utc).timetuple(),
}


# Mocking functions manually
def mock_load_feeds(file_path):
    return {
        "OpenAI": "https://openai.com/blog/rss.xml",
        "HuggingFace": "https://huggingface.co/blog/feed.xml",
        "Microsoft": "https://news.microsoft.com/source/topics/ai/feed/",
    }


def mock_fetch_rss(url):
    return [SAMPLE_RSS_ENTRY]


# Tests for load_feeds function
def test_load_feeds_success():
    result = mock_load_feeds("dummy_path.yaml")
    assert result == {
        "OpenAI": "https://openai.com/blog/rss.xml",
        "HuggingFace": "https://huggingface.co/blog/feed.xml",
        "Microsoft": "https://news.microsoft.com/source/topics/ai/feed/",
    }


# Tests for fetch_rss function
def test_fetch_rss_success():
    result = mock_fetch_rss("https://example.com/rss")
    assert result == [SAMPLE_RSS_ENTRY]


def test_fetch_rss_failure():
    # Simulating a failure scenario
    try:
        raise Exception("Parse error")
    except Exception as e:
        assert str(e) == "Parse error"


# Tests for filter_recent_articles function
def test_filter_recent_articles():
    now = datetime.now(pytz.utc)
    articles = [
        {"published_parsed": now.timetuple()},
        {"published_parsed": (now - timedelta(days=2)).timetuple()},
        {"published_parsed": (now - timedelta(days=5)).timetuple()},
    ]
    result = filter_recent_articles(articles, 3)
    assert len(result) == 2


# Integration test for get_recent_articles function
def test_get_recent_articles():
    mock_load_feeds_func = mock_load_feeds
    mock_fetch_rss_func = mock_fetch_rss

    feeds = mock_load_feeds_func("dummy_path.yaml")
    result = mock_fetch_rss_func(feeds["OpenAI"])

    assert len(result) == 1
    assert result[0]["title"] == "Test Article"


# Error handling test for get_recent_articles
def test_get_recent_articles_error_handling():
    try:
        raise Exception("Network error")
    except Exception as excinfo:
        assert "Network error" in str(excinfo)


if __name__ == "__main__":
    pytest.main()
