"""Game-start gate used by the hedge monitor on sport bots.

Pre-tipoff prices on sports markets swing wildly — injury news,
starting-lineup leaks, weather, all of it. The hedge monitor's
profit-lock and stop-loss thresholds shouldn't fire during that
window because the move that triggered them is rarely a settlement
signal; it's noise that often reverses by tip-off.

This module gates hedge actions on sport positions on two checks:

  1. Has the match actually started? (per an external feed)
  2. Has it been live for at least a per-sport buffer?

External feeds
--------------
  - NBA: ESPN's free public scoreboard JSON. Cached 60s in-process so
    a 30s hedge tick doesn't hammer it.
  - Tennis-style (tennis / table-tennis / darts): the bot's own
    ``watchlist.json``. Each row carries ``current_score``; when it's
    non-empty the match is live.

For any bot key that isn't a sport, ``is_hedge_allowed`` returns
``(True, ...)`` unconditionally — non-sport hedge logic is unchanged.
"""
from __future__ import annotations

import json
import logging
import re
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

log = logging.getLogger("dashboard.sport_game_state")


# Minutes the match must have been observed live before the hedge
# monitor is allowed to act. Tuned per sport — basketball quarters
# are 12 game-minutes (~30 wall-minutes); tennis sets take longer to
# settle into "predictive" patterns, so we wait further into a set.
DEFAULT_BUFFER_MINUTES: Dict[str, int] = {
    "nba": 10,
    "tennis": 15,
    "table-tennis": 8,
    "darts": 10,
}


# In-memory map of "first observed live" timestamps, keyed by
# ``{bot_key}:{ticker_or_match_id}``. Survives across hedge ticks
# inside one process; resets on dashboard restart. Lock guards
# concurrent reads/writes from hedge ticks.
_first_live_seen: Dict[str, float] = {}
_seen_lock = threading.Lock()

# ESPN scoreboard cache (single key per sport endpoint).
_espn_cache: Dict[str, Tuple[float, Optional[dict]]] = {}
_ESPN_TTL_SECONDS = 60.0


def _mark_seen_live(key: str) -> float:
    """Record the first time we saw this match as live. Subsequent
    calls return the same timestamp so the buffer measures wall-time
    since first-observation, not the most recent tick.
    """
    now = time.time()
    with _seen_lock:
        if key not in _first_live_seen:
            _first_live_seen[key] = now
        return _first_live_seen[key]


def _fetch_espn(sport_path: str) -> Optional[dict]:
    """Return cached ESPN scoreboard JSON for the given sport path
    (e.g. ``basketball/nba``). 60-second TTL. ``None`` on any failure
    so callers fall back to "not live" rather than crashing the tick.
    """
    now = time.time()
    cached = _espn_cache.get(sport_path)
    if cached and (now - cached[0]) < _ESPN_TTL_SECONDS:
        return cached[1]
    url = (f"https://site.api.espn.com/apis/site/v2/sports/"
            f"{sport_path}/scoreboard")
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "kalshi-dashboard/1.0"}
        )
        with urllib.request.urlopen(req, timeout=5) as r:
            data = json.load(r)
    except Exception as exc:  # noqa: BLE001
        log.debug("ESPN fetch failed (%s): %s", url, exc)
        _espn_cache[sport_path] = (now, None)
        return None
    _espn_cache[sport_path] = (now, data)
    return data


# Kalshi NBA tickers look like KXNBAGAME-26MAY18SASOKC-SAS. We need
# to split the "<DATE><AWAY><HOME>" middle segment back into team
# codes. Date is the leading 7 chars (YYMMMDD). Team abbreviations
# are typically 3 chars each on ESPN; some legacy markets use 2-4
# chars. We accept 2-4 and try the obvious splits.
_NBA_MID_RE = re.compile(r"^\d{2}[A-Z]{3}\d{1,2}(?P<teams>[A-Z]+)$")


def _nba_team_codes_from_ticker(ticker: str) -> Optional[Tuple[str, str]]:
    if not ticker:
        return None
    parts = ticker.split("-")
    if len(parts) < 2:
        return None
    m = _NBA_MID_RE.match(parts[1])
    if not m:
        return None
    teams = m.group("teams")
    # Try 3+3 first (most common ESPN abbreviation length).
    for split_at in (3, 2, 4):
        if 2 * split_at == len(teams):
            return (teams[:split_at], teams[split_at:])
    if len(teams) >= 6:
        return (teams[:3], teams[3:])
    return None


def _nba_game_state(ticker: str) -> Tuple[bool, str]:
    """Returns (is_live, reason). Reason is a debug string we surface
    in the hedge log so it's clear why a position was skipped."""
    teams = _nba_team_codes_from_ticker(ticker)
    if not teams:
        return (False, "could not parse team codes from ticker")
    data = _fetch_espn("basketball/nba")
    if not data:
        return (False, "ESPN unreachable; defaulting to not-live")
    a_upper = teams[0].upper()
    b_upper = teams[1].upper()
    for ev in (data.get("events") or []):
        for comp in (ev.get("competitions") or []):
            competitors = comp.get("competitors") or []
            names = {
                ((c.get("team") or {}).get("abbreviation") or "").upper()
                for c in competitors
            }
            if a_upper not in names or b_upper not in names:
                continue
            status = (comp.get("status") or {}).get("type") or {}
            state = (status.get("state") or "").lower()
            if state == "in":
                return (True, "ESPN reports game in progress")
            return (False, f"ESPN state={state!r}")
    return (False, "no matching ESPN event found")


def _tennis_match_state(watchlist_path: Optional[str],
                          match_id: str) -> Tuple[bool, str]:
    """Returns (is_live, reason) for a tennis-shape match. Reads the
    bot's watchlist.json — non-empty / non-placeholder current_score
    means the match has started."""
    if not watchlist_path:
        return (False, "no watchlist path on this bot")
    if not match_id:
        return (False, "no match_id on position")
    p = Path(watchlist_path)
    if not p.exists():
        return (False, "watchlist.json missing")
    try:
        with p.open("r", encoding="utf-8") as f:
            d = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        return (False, f"watchlist read failed: {exc}")
    for row in (d.get("rows") or []):
        if row.get("match_id") != match_id:
            continue
        score = (row.get("current_score") or "").strip()
        if not score or score in {"-", "—", "0-0", "0:0"}:
            return (False, "current_score empty — match not started")
        return (True, f"current_score={score!r}")
    return (False, "match_id not found in current watchlist")


def is_hedge_allowed(bot: Dict[str, Any], ticker: Optional[str],
                       buffer_minutes_override: Optional[int] = None
                       ) -> Tuple[bool, str]:
    """Should the hedge monitor close this position right now?

    For non-sport bots: always ``(True, ...)``. For sport bots: only
    once the match is live AND we've observed it live for the
    sport-specific buffer.

    ``ticker`` is the position's market ticker (or match_id for
    tennis-shape bots). Pass the same value the hedge close path
    would log on its action.
    """
    bot_key = bot.get("key") or ""
    dashboard_type = bot.get("dashboard_type") or "standard"
    if bot_key not in DEFAULT_BUFFER_MINUTES:
        return (True, "non-sport bot")
    if not ticker:
        return (True, "no ticker; default-allow")
    if bot_key == "nba":
        live, reason = _nba_game_state(ticker)
    elif dashboard_type == "tennis":
        live, reason = _tennis_match_state(
            bot.get("watchlist_json_path"), ticker,
        )
    else:
        return (True, "unknown sport bot; default-allow")
    if not live:
        return (False, f"pre-game: {reason}")
    first_seen = _mark_seen_live(f"{bot_key}:{ticker}")
    elapsed_min = (time.time() - first_seen) / 60.0
    buf = (buffer_minutes_override
            if buffer_minutes_override is not None
            else DEFAULT_BUFFER_MINUTES.get(bot_key, 10))
    if elapsed_min < buf:
        return (False,
                 f"live for {elapsed_min:.1f}m; buffer {buf}m "
                 f"({reason})")
    return (True, f"live {elapsed_min:.0f}m+, past {buf}m buffer "
                  f"({reason})")
