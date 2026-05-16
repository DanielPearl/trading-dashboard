"""Auto-pause daemon for underperforming bots.

Once every ``interval_seconds`` (default 6h), walks the registered
bot list and applies a simple discipline rule to each:

  Auto-pause when a bot has three consecutive 30-day windows of
  negative realized P&L AND at least one closed bet in each window.

The goal isn't the mechanical action — it's the forcing function.
Auto-pausing a bot that's losing money over 90 days makes the user
consciously decide to re-enable it (via the existing on/off toggle
on the home card) rather than passively letting it bleed.

Tennis-shape adapters (tennis / survivor / table-tennis / darts) are
skipped — they don't carry a sim.db with closed positions and have
their own paper-bet ledgers.

Audit trail
-----------
Every auto-pause is appended as a JSON line to
``<repo>/data/regime_notifications.jsonl``. The dashboard reads the
tail of this file to render a small "Recent auto-pauses" panel on
the Home tab. The file is append-only; never deleted automatically.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

log = logging.getLogger("dashboard.regime_monitor")

NOTIFICATIONS_PATH = (
    Path(__file__).resolve().parents[2] / "data" /
    "regime_notifications.jsonl"
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _conn(db_path: str) -> sqlite3.Connection:
    c = sqlite3.connect(db_path)
    c.row_factory = sqlite3.Row
    return c


def evaluate_bot(db_path: str) -> Dict[str, Any]:
    """Three-window check used by the daemon.

    Returns a dict shaped like:
        {"action": "auto_pause" | "ok",
         "reason": "...",
         "windows": [(label, pnl_cents, n_bets), ...]}
    """
    if not Path(db_path).exists():
        return {"action": "ok", "reason": "no db", "windows": []}
    windows: List[tuple] = []
    try:
        with closing(_conn(db_path)) as c:
            # Three rolling 30-day windows ending today, 30d ago, 60d ago.
            ranges = [(0, 30), (30, 60), (60, 90)]
            for end_days, start_days in ranges:
                row = c.execute(
                    "SELECT COALESCE(SUM(realized_pnl_cents), 0) p, "
                    "COUNT(*) n FROM positions "
                    "WHERE status = 'closed' "
                    "  AND realized_pnl_cents IS NOT NULL "
                    "  AND date(exited_at) >= date('now', ?) "
                    "  AND date(exited_at) <  date('now', ?)",
                    (f"-{start_days} days", f"-{end_days} days"),
                ).fetchone()
                label = f"{end_days}-{start_days}d"
                windows.append((label, int(row["p"] or 0), int(row["n"] or 0)))
    except (sqlite3.OperationalError, sqlite3.DatabaseError):
        return {"action": "ok", "reason": "schema error", "windows": []}
    # Need at least one closed bet in each window to apply the rule —
    # otherwise the "negative P&L" reading is just absence of activity.
    if any(w[2] == 0 for w in windows):
        return {"action": "ok",
                "reason": "insufficient activity across all 3 windows",
                "windows": windows}
    if all(w[1] < 0 for w in windows):
        loss_str = " · ".join(
            f"{w[0]}: −${abs(w[1])/100:.2f} ({w[2]} bets)"
            for w in windows
        )
        return {"action": "auto_pause",
                "reason": f"3 consecutive 30d windows all negative — {loss_str}",
                "windows": windows}
    return {"action": "ok", "reason": "edge holding", "windows": windows}


def _log_notification(entry: Dict[str, Any]) -> None:
    """Append a single notification line to the JSONL audit log."""
    try:
        NOTIFICATIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with NOTIFICATIONS_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, separators=(",", ":")) + "\n")
    except OSError as exc:
        log.warning("regime_monitor notification write failed: %s", exc)


def read_notifications(limit: int = 20) -> List[Dict[str, Any]]:
    """Tail the audit log — newest first. Used by the Home tab's
    'Recent auto-pauses' panel. Returns ``[]`` when the file is
    missing or unreadable.
    """
    if not NOTIFICATIONS_PATH.exists():
        return []
    try:
        with NOTIFICATIONS_PATH.open("r", encoding="utf-8") as f:
            lines = f.readlines()
    except OSError:
        return []
    out: List[Dict[str, Any]] = []
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
        if len(out) >= limit:
            break
    return out


def tick(bots: List[dict]) -> None:
    """One pass across the bot list. Idempotent — a bot that's already
    paused gets skipped so we don't double-log on every cycle.
    """
    from . import bot_state
    for b in bots:
        bot_key = b.get("key")
        if not bot_key:
            continue
        if b.get("dashboard_type") in ("tennis", "survivor"):
            continue
        db_path = b.get("db_path") or ""
        if not db_path:
            continue
        if not bot_state.is_bot_enabled(bot_key):
            continue  # already paused — manual or prior auto-pause
        result = evaluate_bot(db_path)
        if result.get("action") != "auto_pause":
            continue
        bot_state.set_bot_enabled(bot_key, False)
        _log_notification({
            "ts": _now_iso(),
            "bot_key": bot_key,
            "bot_name": b.get("name", bot_key),
            "action": "auto_pause",
            "reason": result.get("reason", ""),
            "windows": result.get("windows", []),
        })
        log.info("regime_monitor: auto-paused bot=%s reason=%s",
                  bot_key, result.get("reason", ""))


def start_daemon(bots: List[dict],
                  interval_seconds: int = 6 * 3600) -> threading.Thread:
    """Spawn the regime-monitor background thread. Daemon = True so
    SIGINT on the dashboard tears it down cleanly. Sleeps before the
    first tick so the dashboard's startup logs aren't competing.
    """

    def _loop() -> None:
        log.info("regime_monitor started (interval=%ds)", interval_seconds)
        # Wait one cycle before the first run — gives the bot service
        # time to come up and write its first day's worth of data
        # before we render any judgement on it.
        time.sleep(interval_seconds)
        while True:
            try:
                tick(bots)
            except Exception:  # noqa: BLE001
                log.exception("regime_monitor loop iteration failed")
            time.sleep(interval_seconds)

    t = threading.Thread(target=_loop, daemon=True,
                          name="regime-monitor")
    t.start()
    return t
