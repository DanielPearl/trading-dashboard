"""Per-ticker market history reader for the in-game models.

Pulls recent ``market_views`` rows for a ticker and shapes them into
the ``[(unix_ts, yes_cents), ...]`` tuples the feature functions
expect.

All reads are best-effort: missing DB / unknown ticker / schema
gaps all return ``[]`` so callers can blend gracefully.
"""
from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple

# Same parse idiom as the dashboard's other history paths — slice
# to 19 chars to drop fractional seconds / tz offsets, then assume
# UTC. Kalshi sim.dbs are all UTC.
def _iso_to_unix(s: str | None) -> float | None:
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s[:19])
        return dt.replace(tzinfo=timezone.utc).timestamp()
    except (TypeError, ValueError):
        return None


def recent_market_history(db_path: str, ticker: str,
                            hours: int = 6,
                            limit: int = 500
                            ) -> List[Tuple[float, float]]:
    """Last N hours of ``yes_ask_cents`` snapshots for one ticker.

    Returns ``[(unix_ts, yes_cents), ...]`` sorted ascending. Empty
    list when the DB is missing, the ticker has no rows, or schema
    drift breaks the query.
    """
    if not db_path or not ticker:
        return []
    if not Path(db_path).exists():
        return []
    try:
        with closing(sqlite3.connect(db_path)) as c:
            c.row_factory = sqlite3.Row
            rows = c.execute(
                "SELECT captured_at, yes_ask_cents FROM market_views "
                "WHERE ticker = ? "
                "  AND yes_ask_cents IS NOT NULL "
                "  AND captured_at >= datetime('now', ?) "
                "ORDER BY captured_at ASC LIMIT ?",
                (ticker, f"-{int(hours)} hours", limit),
            ).fetchall()
    except (sqlite3.OperationalError, sqlite3.DatabaseError):
        return []
    out: List[Tuple[float, float]] = []
    for r in rows:
        ts = _iso_to_unix(r["captured_at"])
        if ts is None:
            continue
        try:
            out.append((ts, float(r["yes_ask_cents"])))
        except (TypeError, ValueError):
            continue
    return out


def latest_yes_cents(db_path: str, ticker: str) -> float | None:
    """The most recent ``yes_ask_cents`` for a ticker, or ``None``.

    Convenience helper used when the in-game model only needs the
    current price and doesn't care about the trajectory.
    """
    hist = recent_market_history(db_path, ticker, hours=24, limit=1)
    return hist[-1][1] if hist else None
