"""
Custom exceptions for the Rizal Exploration Engine.
"""

class RizalError(Exception):
    """Base class for all Rizal engine errors."""
    pass

class QueryTooShortError(RizalError):
    """Raised when the search query is too short."""
    pass

class DataLoadingError(RizalError):
    """Raised when required data files cannot be loaded."""
    pass

class TopicNotFoundError(RizalError):
    """Raised when semantic search yields no relevant results."""
    pass
