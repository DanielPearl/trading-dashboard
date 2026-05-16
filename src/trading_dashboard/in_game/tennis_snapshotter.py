"""Per-poll tennis odds snapshotter.

Runs in the dashboard process as a background thread. Every
``interval_seconds`` (default 60s) it walks every tennis-shape
bot's ``watchlist.json`` and appends each match's current
``live_prob_a`` / ``live_prob_b`` / ``yes_ask_cents_a`` /
``yes_ask_cents_b`` / ``current_score`` to a per-match JSONL log:

    data/tennis_history/<bot_key>__<match_id>.jsonl

These files give ``in_game/tennis.py`` the time-series substrate
it needs for the cross-sport velocity / volatility / divergence
features — same shape NBA gets from ``market_views``.

The snapshotter never writes to bot data, only to the dashboard's
own ``data/`` dir. Append-only; never rewritten. Stale entries
naturally drop out of the in-game-features window (last 6h).
"""
from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("dashboard.in_game.tennis_snapshotter")


HISTORY_DIR = (Path(__file__).resolve().parents[3]
               / "data" / "tennis_history")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _safe_filename(s: str) -> str:
    """Make a watchlist match_id safe for a filename. match_ids are
    already alphanumeric+dash but we defensively strip anything else.
    """
    return "".join(c if c.isalnum() or c in "-_." else "_"
                    for c in (s or "match"))


def _snapshot_path(bot_key: str, match_id: str) -> Path:
    return HISTORY_DIR / f"{_safe_filename(bot_key)}__{_safe_filename(match_id)}.jsonl"


def snapshot_bot(bot: Dict[str, Any]) -> int:
    """Append one snapshot per watchlist row for this bot. Returns
    the number of rows written.
    """
    if (bot.get("dashboard_type") or "") not in ("tennis", "survivor"):
        return 0
    path = bot.get("watchlist_json_path")
    if not path:
        return 0
    p = Path(path)
    if not p.exists():
        return 0
    try:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        log.debug("snapshotter watchlist read failed (%s): %s", path, exc)
        return 0
    bot_key = bot.get("key") or "tennis"
    written = 0
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    for row in (data.get("rows") or []):
        match_id = row.get("match_id")
        if not match_id:
            continue
        # Skip rows that have no live signal yet — pre-match polls
        # would just spam the log with constant pre_match priors.
        if not (row.get("current_score") or "").strip():
            continue
        entry = {
            "ts": _now_iso(),
            "live_prob_a": row.get("live_prob_a"),
            "live_prob_b": row.get("live_prob_b"),
            "pre_match_prob_a": row.get("pre_match_prob_a"),
            "yes_ask_cents_a": row.get("yes_ask_cents_a"),
            "yes_ask_cents_b": row.get("yes_ask_cents_b"),
            "current_score": row.get("current_score"),
            "round_label": row.get("round_label"),
            "confidence_score": row.get("confidence_score"),
            "volatility_score": row.get("volatility_score"),
        }
        try:
            with _snapshot_path(bot_key, match_id).open(
                    "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, separators=(",", ":")) + "\n")
            written += 1
        except OSError as exc:
            log.debug("snapshot write failed (%s): %s", match_id, exc)
            continue
    return written


def _iso_to_unix(s: str | None) -> Optional[float]:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s[:19]).replace(
            tzinfo=timezone.utc).timestamp()
    except (TypeError, ValueError):
        return None


def read_history(bot_key: str, match_id: str,
                   hours: int = 6) -> List[Tuple[float, Dict[str, Any]]]:
    """Return [(ts, snapshot_dict), ...] for one match, sorted
    ascending. Filtered to the last ``hours`` of entries. Empty
    list when the file doesn't exist or is unreadable.
    """
    if not bot_key or not match_id:
        return []
    path = _snapshot_path(bot_key, match_id)
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8") as f:
            lines = f.readlines()
    except OSError:
        return []
    out: List[Tuple[float, Dict[str, Any]]] = []
    cutoff = time.time() - hours * 3600
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        ts = _iso_to_unix(d.get("ts"))
        if ts is None or ts < cutoff:
            continue
        out.append((ts, d))
    out.sort(key=lambda r: r[0])
    return out


def yes_cents_history(bot_key: str, match_id: str, side_label: str,
                       hours: int = 6) -> List[Tuple[float, float]]:
    """Return [(ts, yes_cents), ...] for the side we hold. Used by
    the cross-sport features module's velocity / volatility helpers.

    ``side_label`` is "A" or "B" — tells us which yes_ask column to
    read on each snapshot row. The b-side YES price implicitly
    equals 100 − yes_ask_a if only one is recorded, but we prefer
    the explicit column when both are present.
    """
    out: List[Tuple[float, float]] = []
    for ts, d in read_history(bot_key, match_id, hours=hours):
        col = ("yes_ask_cents_a" if side_label == "A"
                else "yes_ask_cents_b")
        v = d.get(col)
        if v is None:
            # Fall back to the opposite side flipped if available.
            other = ("yes_ask_cents_b" if side_label == "A"
                      else "yes_ask_cents_a")
            ov = d.get(other)
            if ov is None:
                continue
            try:
                v = 100 - int(ov)
            except (TypeError, ValueError):
                continue
        try:
            out.append((ts, float(v)))
        except (TypeError, ValueError):
            continue
    return out


def start_daemon(bots: List[Dict[str, Any]],
                  interval_seconds: int = 60) -> threading.Thread:
    """Spawn the snapshotter background thread. Walks every
    tennis-shape bot once per interval. Daemon=True so SIGINT
    tears it down with the dashboard.
    """

    def _loop() -> None:
        log.info(
            "tennis_snapshotter started (interval=%ds, dir=%s)",
            interval_seconds, HISTORY_DIR,
        )
        # Wait one full interval before first run — dashboard
        # startup logs are already noisy.
        time.sleep(interval_seconds)
        while True:
            try:
                total = 0
                for b in bots:
                    total += snapshot_bot(b)
                if total > 0:
                    log.debug("tennis_snapshotter: %d snapshots", total)
            except Exception:  # noqa: BLE001
                log.exception("tennis_snapshotter tick failed")
            time.sleep(interval_seconds)

    t = threading.Thread(target=_loop, daemon=True,
                          name="tennis-snapshotter")
    t.start()
    return t
