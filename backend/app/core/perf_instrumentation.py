"""
Performance instrumentation for RizalEngine.search()

This module provides timing wrappers and metadata collection
for detailed performance profiling when PERF_BENCHMARK=1.

Usage in engine.py:
    from app.core.perf_instrumentation import PerfTimer

    timer = PerfTimer()
    with timer.stage("preprocessing"):
        # ... preprocessing code ...

    # At end of search():
    return {
        "results": results,
        "metadata": {
            ...,
            "timings": timer.get_timings() if PERF_BENCHMARK else None
        }
    }
"""

import time
from typing import Dict, Any, Optional
from contextlib import contextmanager


class PerfTimer:
    """Context manager for stage-level timing collection"""

    def __init__(self):
        self.timings: Dict[str, float] = {}
        self.metadata: Dict[str, Any] = {}
        self._stage_stack = []

    @contextmanager
    def stage(self, name: str):
        """Time a specific stage of execution"""
        start = time.perf_counter()
        self._stage_stack.append(name)
        try:
            yield
        finally:
            elapsed = (time.perf_counter() - start) * 1000  # Convert to ms
            self.timings[name] = round(elapsed, 2)
            self._stage_stack.pop()

    def record_metadata(self, key: str, value: Any):
        """Record non-timing metadata (counts, flags, etc.)"""
        self.metadata[key] = value

    def get_timings(self) -> Dict[str, Any]:
        """Get all collected timings and metadata"""
        return {
            **self.timings,
            **self.metadata
        }
