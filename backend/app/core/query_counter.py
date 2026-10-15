"""
Database Query Counter Middleware

Tracks the number of database queries executed per request
for performance profiling.
"""

from sqlalchemy import event
from sqlalchemy.engine import Engine
import threading

# Thread-local storage for query counts
_query_counts = threading.local()


def get_query_count():
    """Get the current query count for this thread"""
    return getattr(_query_counts, 'count', 0)


def reset_query_count():
    """Reset the query count for this thread"""
    _query_counts.count = 0


def increment_query_count():
    """Increment the query count for this thread"""
    if not hasattr(_query_counts, 'count'):
        _query_counts.count = 0
    _query_counts.count += 1


# Register SQLAlchemy event listener
@event.listens_for(Engine, "before_cursor_execute")
def receive_before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
    """Intercept all SQL queries and count them"""
    increment_query_count()
