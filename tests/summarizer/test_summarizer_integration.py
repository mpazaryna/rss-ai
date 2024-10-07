import os
import unittest

from rss_ai.summarizer import generate_summary


def test_generate_summary():
    # Read the article content from the file
    with open("data/mock/article.md", "r") as file:
        article_text = file.read()

    # Generate summary
    summary = generate_summary(article_text)

    # Assert that the summary is not empty
    assert len(summary) > 0, "The summary should not be empty."
