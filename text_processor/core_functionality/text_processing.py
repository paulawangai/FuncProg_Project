import re
from typing import List, Iterator


def _clean_text(text: str) -> str:
    """
    Removes punctuation marks and numbers from text in a functional way

    Args:
        text: Input text to clean

    Returns:
        Cleaned text
    """
    cleaned = re.sub(r'[^\w\s]|\d', '', text.lower())
    return re.sub(r'\s+', ' ', cleaned).strip()


def process_text_chunk(chunk: str) -> List[str]:

    """Pure function to process a chunk of text"""

    def tokenize(text: str) -> Iterator[str]:
        cleaned = re.sub(r'[^\w\s]|\d', '', text.lower())
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return filter(bool, cleaned.split())

    return list(set(tokenize(chunk)))
