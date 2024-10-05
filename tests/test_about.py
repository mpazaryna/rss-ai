"""Tests for the about module in the RSS AI project."""

from rss_ai import about


def test_get_version():
    """Test that the get_version function returns the expected version string."""
    assert about.get_version() == "0.1.0"
