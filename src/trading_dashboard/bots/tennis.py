"""Tennis bot — runs the Baseline Break (ATP / WTA match) loop inside
the dashboard process.

This module is the first of the "folded-in" bots. The tennis-forecast
project at ``/root/tennis-forecast`` was previously its own systemd
service (``baseline-break-monitor.service``) running a 60-second poll
loop and writing ``data/raw/live_state.json`` + ``data/outputs/*``.
That service is now disabled; this thread does the same work from
inside the dashboard process so:

  * The bot toggle on the Home tab actually controls trading (the
    previous standalone service never read ``bot_states.json``).
  * pandas / numpy / scikit-learn / joblib only get imported once for
    the dashboard process instead of once per bot service.
  * State is reloaded by the dashboard's reader without crossing a
    process boundary.

How it works
------------
The tennis-forecast repo isn't pip-installable (no setup.py); it relies
on a ``sys.path`` insert inside ``scripts/run_live_monitor.py`` to find
its ``src/`` package. We mirror that here: inject the configured
``tennis_repo_path`` onto ``sys.path`` once, then import the same
modules that the original script imported. Each tick re-uses
upstream's pure functions — no logic was reimplemented.

Bot toggle behaviour
--------------------
Each iteration ALWAYS fetches markets, updates the watchlist, and runs
the simulator's mark-to-market / settlement step. When
``bot_state.is_bot_enabled("tennis")`` returns False, the simulator is
called with an empty watchlist so it can keep settling existing
positions but never opens new ones. This matches what the user
expects from the "model toggle card" on the Home tab — paused means
no new exposure, not "freeze everything in place".
"""
from __future__ import annotations

import logging
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

from .. import bot_state


log = logging.getLogger("dashboard.tennis_bot")

BOT_KEY = "tennis"

# Cache of upstream's pure-function entrypoints. Populated on first
# tick by ``_load_upstream`` so the import cost is paid lazily, after
# the HTTP server is already serving — keeps the dashboard's startup
# fast even though the tennis bundle pulls in pandas + sklearn.
_upstream: dict[str, Callable[..., Any]] | None = None

# Snapshot of the previous tick's per-ticker yes-ask prices. Mirrors
# the module-global in ``scripts/run_live_monitor.py`` (line 55) so
# the overreaction rule has somewhere to read prev_yes from.
_prev_market_by_ticker: dict[str, dict] = {}


def _load_upstream(repo_path: Path) -> dict[str, Callable[..., Any]]:
    """Inject the tennis-forecast repo onto sys.path and import the
    pure functions we need. Returns a dict of callables — no module
    objects leak out so callers can't accidentally reach into upstream
    state.
    """
    if not repo_path.exists():
        raise RuntimeError(
            f"tennis bot disabled: tennis-forecast repo not found at "
            f"{repo_path}. Either clone it there or set "
            f"tennis_trader.repo_path in dashboard.yaml."
        )
    src_parent = str(repo_path)
    if src_parent not in sys.path:
        # Match what scripts/run_live_monitor.py:25 does — prepend so
        # upstream's `src.X.Y` imports resolve before any colliding
        # package in site-packages.
        sys.path.insert(0, src_parent)

    # Imports are deferred to here so the dashboard can start without
    # pandas/sklearn loaded — the bot only pulls them in when the
    # daemon thread actually runs its first tick.
    from src.data import kalshi_markets  # type: ignore  # noqa: E402
    from src.dashboard.export_watchlist import (  # type: ignore  # noqa: E402
        build_watchlist_records,
        export as export_watchlist,
    )
    from src.trading.simulator import tick as simulator_tick  # type: ignore  # noqa: E402

    return {
        "fetch_tennis_markets": kalshi_markets.fetch_tennis_markets,
        "collapse_to_matches": kalshi_markets.collapse_to_matches,
        "write_live_state": kalshi_markets.write_live_state,
        "build_watchlist_records": build_watchlist_records,
        "export_watchlist": export_watchlist,
        "simulator_tick": simulator_tick,
    }


def _require_kalshi_creds() -> None:
    """Same check upstream does at startup. Without creds the SDK can't
    fetch live markets and we'd just log empty ticks forever — better
    to log a single startup error and let the operator notice.
    """
    import os
    missing = [k for k in ("KALSHI_API_KEY_ID", "KALSHI_PRIVATE_KEY_PATH")
                if not os.environ.get(k, "").strip()]
    if missing:
        raise RuntimeError(
            f"tennis bot needs Kalshi creds in env: {', '.join(missing)} "
            f"missing. The dashboard's systemd unit loads these from "
            f"/root/gas-prices/.env; check that file exists."
        )


def _one_tick(upstream: dict[str, Callable[..., Any]]) -> None:
    """Single iteration. Mirrors ``scripts/run_live_monitor.py:_one_tick``
    but routes the simulator's open-new step through the bot_state
    toggle: when tennis is paused we pass an empty watchlist so
    existing positions still get marked + settled but no new exposure
    is opened.
    """
    global _prev_market_by_ticker

    raw_markets = upstream["fetch_tennis_markets"]()
    new_prev = {m.get("ticker"): m for m in raw_markets if m.get("ticker")}
    records = upstream["collapse_to_matches"](
        raw_markets, prev_markets_by_ticker=_prev_market_by_ticker,
    )
    _prev_market_by_ticker = new_prev
    upstream["write_live_state"](records)

    rows = upstream["build_watchlist_records"]()
    upstream["export_watchlist"](records=rows)

    # Bot-state gate: if paused, hand the simulator an empty watchlist
    # so its open-new-positions step finds no candidates. The mark and
    # settle steps still run against ``records`` so any open position
    # closes out normally when its match finishes.
    enabled = bot_state.is_bot_enabled(BOT_KEY)
    rows_for_sim = rows if enabled else []
    state = upstream["simulator_tick"](rows_for_sim, records)

    log.info(
        "tennis tick — %d kalshi markets / %d matches / %d watchlist rows "
        "/ %d open / %d closed (P&L %+.3f, ROI %s)%s",
        len(raw_markets), len(records), len(rows),
        state["stats"].get("open_count", 0),
        state["stats"].get("total_closed", 0),
        state["stats"].get("total_realized_pnl", 0.0),
        ("—" if state["stats"].get("roi") is None
         else f"{state['stats']['roi'] * 100:+.1f}%"),
        ("" if enabled else " [PAUSED — no new positions]"),
    )


def start_daemon(tennis_trader_cfg: dict,
                  interval_seconds: int | None = None) -> threading.Thread:
    """Spawn the tennis-trading background thread. Daemon = True so
    Ctrl-C / SIGTERM on the dashboard tears it down cleanly. No-op
    (returns a dead thread) when tennis_trader.enabled is false.

    Expected config shape (under ``tennis_trader:`` in dashboard.yaml):

        tennis_trader:
          enabled: true
          repo_path: /root/tennis-forecast
          interval_seconds: 60
    """
    enabled = bool(tennis_trader_cfg.get("enabled"))
    repo_path = Path(tennis_trader_cfg.get("repo_path",
                                            "/root/tennis-forecast"))
    interval = int(interval_seconds
                    or tennis_trader_cfg.get("interval_seconds", 60))

    def _loop() -> None:
        log.info("tennis_bot starting (interval=%ds, repo=%s, bot_key=%s)",
                  interval, repo_path, BOT_KEY)
        try:
            _require_kalshi_creds()
        except RuntimeError as exc:
            log.error("%s", exc)
            return  # thread exits — operator will see the log line
        try:
            upstream = _load_upstream(repo_path)
        except Exception:  # noqa: BLE001
            log.exception("tennis_bot failed to import upstream package")
            return
        log.info("tennis_bot upstream loaded; entering tick loop")
        while True:
            try:
                _one_tick(upstream)
            except Exception:  # noqa: BLE001
                log.exception("tennis_bot tick failed")
            time.sleep(interval)

    t = threading.Thread(target=_loop, daemon=True, name="tennis-bot")
    if enabled:
        t.start()
    else:
        log.info("tennis_bot disabled in config (tennis_trader.enabled=false)")
    return t
