"""
profiling_log.py — Shared file-based profiling logger.

All profiling output (StepProfiler, rollout timing, etc.) is written to
a single ``profiling.log`` file inside the run directory instead of
cluttering stdout.

Usage::

    from profiling_log import get_profiling_logger

    log = get_profiling_logger()       # uses default path
    log("⏱  Rollout profiling …")     # appends a line

    # Or redirect an entire block:
    with open(log.path, "a") as f:
        f.write(block_text)
"""

from __future__ import annotations

import os
import threading
from pathlib import Path


_DEFAULT_LOG_DIR = "runs/hover"
_LOG_FILENAME = "profiling.log"

_lock = threading.Lock()
_logger_instance: "ProfilingLogger | None" = None


class ProfilingLogger:
    """Append-only file logger for profiling messages."""

    def __init__(self, log_dir: str | Path) -> None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        self.path: Path = log_dir / _LOG_FILENAME

    def __call__(self, text: str) -> None:
        """Append *text* (with trailing newline) to the log file."""
        with _lock:
            with open(self.path, "a") as f:
                f.write(text)
                if not text.endswith("\n"):
                    f.write("\n")

    def write_lines(self, lines: list[str]) -> None:
        """Write several lines at once."""
        with _lock:
            with open(self.path, "a") as f:
                for line in lines:
                    f.write(line)
                    if not line.endswith("\n"):
                        f.write("\n")
                f.write("\n")


def init_profiling_logger(log_dir: str | Path) -> ProfilingLogger:
    """Initialise (or re-initialise) the global profiling logger."""
    global _logger_instance
    _logger_instance = ProfilingLogger(log_dir)
    return _logger_instance


def get_profiling_logger() -> ProfilingLogger:
    """Return the global logger, creating one with defaults if needed."""
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = ProfilingLogger(_DEFAULT_LOG_DIR)
    return _logger_instance
