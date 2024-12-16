from typing import NamedTuple, Optional, List
from enum import Enum, auto


class Colour(Enum):
    RED = auto()  # get assigned integer value
    BLACK = auto()


class FileContent(NamedTuple):
    """Immutable container for file content"""
    content: str
    size: int


class SystemInfo(NamedTuple):
    """Immutable container for system information"""
    cpu_count: int
    chunk_size: int


class ProcessingResult(NamedTuple):
    """Immutable container for processing results"""
    words: List[str]
    count: int
    duration: float
    success: bool
    error: Optional[str]


class IOResult(NamedTuple):
    """Immutable container for IO results"""
    success: bool
    content: Optional[str]
    error: Optional[str]
