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
    """Context manager for stage-level timing collection with accumulation and call counts"""

    def __init__(self):
        self.timings: Dict[str, float] = {}
        self.call_counts: Dict[str, int] = {}
        self.metadata: Dict[str, Any] = {}
        self._stage_stack = []

    @contextmanager
    def stage(self, name: str):
        """Time a specific stage of execution, accumulating elapsed ms and incrementing call count"""
        start = time.perf_counter()
        self._stage_stack.append(name)
        try:
            yield
        finally:
            elapsed = (time.perf_counter() - start) * 1000  # Convert to ms
            self.timings[name] = self.timings.get(name, 0.0) + elapsed
            self.call_counts[name] = self.call_counts.get(name, 0) + 1
            self._stage_stack.pop()

    def record_stage(self, name: str, elapsed_ms: float):
        """Record or accumulate an externally-measured stage duration with call count"""
        self.timings[name] = self.timings.get(name, 0.0) + elapsed_ms
        self.call_counts[name] = self.call_counts.get(name, 0) + 1

    def record_metadata(self, key: str, value: Any):
        """Record non-timing metadata (counts, flags, etc.)"""
        self.metadata[key] = value

    def get_timings(self) -> Dict[str, Any]:
        """Get all collected timings (rounded to 2 decimal places), call counts, and metadata"""
        # Ensure any stage recorded directly in self.timings has a call count
        for k in self.timings:
            if k not in self.call_counts:
                self.call_counts[k] = 1
        rounded_timings = {k: round(v, 2) for k, v in self.timings.items()}
        counts = {f"{k}_count": v for k, v in self.call_counts.items()}
        return {
            **rounded_timings,
            **counts,
            **self.metadata
        }
