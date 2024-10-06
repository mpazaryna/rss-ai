import logging
import time
from typing import Optional

import requests
from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_USER_AGENT = "RSS-AI-Scraper/1.0"
RATE_LIMIT_DELAY = 1  # seconds


def scrape_url(url: str, user_agent: Optional[str] = None) -> str:
    """
    Scrape the content of a given URL using BeautifulSoup to strip HTML tags.

    Args:
        url (str): The URL to scrape.
        user_agent (Optional[str]): Custom user agent string. If None, uses default.

    Returns:
        str: The cleaned text content of the webpage.

    Raises:
        ValueError: If the URL is invalid.
        requests.RequestException: For network-related errors.
    """
    headers = {"User-Agent": user_agent or DEFAULT_USER_AGENT}

    try:
        # Implement basic rate limiting
        time.sleep(RATE_LIMIT_DELAY)

        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching URL {url}: {str(e)}")
        raise

    soup = BeautifulSoup(response.content, "html.parser")

    # Remove script and style elements
    for script_or_style in soup(["script", "style"]):
        script_or_style.decompose()

    # Get text content
    text = soup.get_text(separator="\n")

    # Break into lines and remove leading and trailing space on each
    lines = (line.strip() for line in text.splitlines())

    # Drop blank lines
    text = "\n".join(line for line in lines if line)

    logger.info(f"Successfully scraped content from {url}")
    return text


if __name__ == "__main__":
    # Example usage
    test_url = "https://example.com"
    try:
        content = scrape_url(test_url)
        print(content[:500])  # Print first 500 characters
    except Exception as e:
        print(f"An error occurred: {str(e)}")
