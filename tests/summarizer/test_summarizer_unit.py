import unittest

from rss_ai.summarizer import generate_cache_filename  # Import the function to test


def test_generate_cache_filename():
    # Test case for a sample URL
    article_url = "http://example.com/articles/new_feature.html"
    expected_filename = "example_com_new_feature"  # Expected filename format
    generated_filename = generate_cache_filename(article_url)  # Call the function
    assert generated_filename == expected_filename  # Verify the result


# ... existing tests ...
