"""Per-bot enable/disable state persistence.

The dashboard's homepage card grid carries a toggle per bot. Toggling
it writes the new state to a JSON file the dashboard reads on every
page render and that the individual bot services can poll to decide
whether to keep placing new bets.

State file lives at ``data/bot_states.json`` next to the dashboard.
Schema:

    {
      "states": {
        "<bot_key>": {
          "enabled": true,
          "updated_at": "2026-05-15T06:30:00Z"
        },
        …
      }
    }

Bots that don't yet check this file just keep running as before — the
toggle then acts as a UI status flag the user maintains. To make a
bot honor the toggle, call ``is_bot_enabled(bot_key)`` at the top of
its buy/sell decision loop and short-circuit when False.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

log = logging.getLogger("dashboard.bot_state")


# Resolved at import time so callers don't need to know the path. The
# dashboard service writes here; bot services on the same host read.
#
# ``KALSHI_BOT_STATES_PATH`` (env) overrides the default. This is the
# isolation knob for the live/sim split: the sim systemd unit leaves
# it unset (default file = ``<repo>/data/bot_states.json``) while the
# live systemd unit sets it to ``<repo>/data/bot_states_live.json``,
# so toggling a bot on in one dashboard never flips it on in the
# other. Without this isolation a stray sim click could enable a
# real-money trader.
#
# Also used by unit tests to point at a tempfile per run.
_STATE_PATH = Path(
    os.environ.get(
        "KALSHI_BOT_STATES_PATH",
        # Default: alongside the trading-dashboard repo's data dir
        # on the droplet, or local data/ during development.
        Path(__file__).resolve().parents[2] / "data" / "bot_states.json",
    )
).resolve()


def _read() -> Dict[str, Any]:
    """Load the state document. Missing file = "everyone enabled"."""
    if not _STATE_PATH.exists():
        return {"states": {}}
    try:
        with _STATE_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f) or {}
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("bot_states read failed (%s) — treating all as enabled",
                     exc)
        return {"states": {}}
    if "states" not in data or not isinstance(data["states"], dict):
        data["states"] = {}
    return data


def _atomic_write(payload: Dict[str, Any]) -> None:
    """Write the state JSON atomically — concurrent readers on the
    droplet (each bot service) should never see a half-written file.
    """
    _STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8",
        dir=str(_STATE_PATH.parent),
        prefix=".bot_states.", suffix=".tmp",
        delete=False,
    )
    try:
        json.dump(payload, tmp, indent=2)
        tmp.flush()
        os.fsync(tmp.fileno())
    finally:
        tmp.close()
    os.replace(tmp.name, _STATE_PATH)


def is_bot_enabled(bot_key: str) -> bool:
    """Public read API the individual bot services use.

    Defaults to True when the state file is missing OR the bot is not
    listed — keeps fresh-clone bot deployments running without the
    user having to touch the toggle first.
    """
    data = _read()
    entry = data["states"].get(bot_key)
    if not entry:
        return True
    return bool(entry.get("enabled", True))


def get_all_states() -> Dict[str, Dict[str, Any]]:
    """Return the full bot -> {enabled, updated_at} map. Used by the
    dashboard renderer to colour the cards."""
    return dict(_read()["states"])


def set_bot_enabled(bot_key: str, enabled: bool) -> Dict[str, Any]:
    """Write ``enabled`` for ``bot_key`` and return the new entry.
    Stamps ``updated_at`` with the current UTC ISO timestamp."""
    data = _read()
    entry = {
        "enabled": bool(enabled),
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    data["states"][bot_key] = entry
    _atomic_write(data)
    return entry


def toggle_bot(bot_key: str) -> Dict[str, Any]:
    """Flip the bot's enabled flag and return the new entry. Used by
    the POST /api/bot/toggle endpoint behind the homepage toggle UI."""
    current = is_bot_enabled(bot_key)
    return set_bot_enabled(bot_key, not current)


def bootstrap_disabled(bot_keys: list[str]) -> bool:
    """Write a state file with every bot in ``bot_keys`` explicitly
    set to ``enabled: false`` — but ONLY if the file doesn't exist
    yet. Returns True if a fresh file was written, False if one was
    already present (in which case the caller's preferences win and
    nothing is touched).

    This is the "live dashboard starts with every toggle OFF"
    enforcement. Without it, the UI would default to "enabled" for
    unlisted bots (see ``is_bot_enabled`` — that default makes sense
    for sim where the implicit answer is "yes please trade", but is
    the wrong default for live where the implicit answer is "no
    real money until I explicitly opt in"). Called from the live
    dashboard's main() at startup.
    """
    if _STATE_PATH.exists():
        return False
    now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
    payload: Dict[str, Any] = {
        "states": {
            key: {"enabled": False, "updated_at": now_iso}
            for key in bot_keys
        },
    }
    _atomic_write(payload)
    log.info("bootstrap_disabled — wrote %s with %d bots all OFF",
              _STATE_PATH, len(bot_keys))
    return True
