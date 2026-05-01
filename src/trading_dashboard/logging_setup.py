"""Logging helpers."""
from __future__ import annotations

import logging
from pathlib import Path


def setup_logging(log_path: str | None = None, level: int = logging.INFO) -> None:
    """Plain-text logging suitable for journalctl AND a local file.

    No ANSI colors so output is readable when piped through systemd or ssh.
    Stream handler is unbuffered (flush=True) so `journalctl -fu` shows
    lines as they happen instead of in chunks.
    """
    handlers: list[logging.Handler] = []

    stream = logging.StreamHandler()
    stream.flush = lambda: None  # noop, but ensures no buffering shenanigans
    handlers.append(stream)

    if log_path:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path))

    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-5s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )

    # Quiet down a couple of chatty libraries.
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("yfinance").setLevel(logging.WARNING)
