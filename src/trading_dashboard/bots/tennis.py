"""Tennis bot — runs the Baseline Break (ATP / WTA match) loop inside
the dashboard process.

Shape B (hand-rolled tick): the upstream entry point in tennis-forecast
is a script with an inline ``while True``, so we reproduce the
per-tick body here and spawn our own thread around it.

Bot toggle behaviour
--------------------
Each iteration ALWAYS fetches markets and updates the watchlist so
the dashboard UI stays fresh. When ``bot_state.is_bot_enabled("tennis")``
returns False we hand the simulator an empty watchlist, so its
mark-to-market and settlement steps still run on existing positions
but no new ones open. (Tennis positions are short-lived — minutes to
hours — so freezing the open-new step but keeping settle-existing
running is the right balance. Macro bots like unemployment use the
coarser gate_bot_tick helper instead.)
"""
from __future__ import annotations

import logging
import time
from typing import Any, Callable

from . import _base
from .. import bot_state


log = logging.getLogger("dashboard.tennis-bot")

BOT_KEY = "tennis"

# Per-process cache of the previous tick's per-ticker yes-ask prices.
# Mirrors the global in scripts/run_live_monitor.py line 55; needed
# for the overreaction rule to see how the market moved between ticks.
_prev_market_by_ticker: dict[str, dict] = {}


def _load_upstream(repo_path: str) -> dict[str, Callable[..., Any]]:
    """Import tennis-forecast's pure functions. The repo isn't
    pip-installable so we inject its root onto sys.path (it puts its
    own ``src/`` on the path via ``scripts/run_live_monitor.py:25``;
    we mirror that by passing ``subdir=None`` and letting upstream's
    ``from src.X import Y`` imports resolve from the repo root).
    """
    _base.inject_sys_path(repo_path, subdir=None)

    # Deferred imports — paid once on first tick, after the HTTP
    # server is already serving requests. Keeps dashboard startup
    # fast even though this bundle pulls in pandas + sklearn.
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


def _one_tick(upstream: dict[str, Callable[..., Any]]) -> None:
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

    # Fine-grained gate: pass empty rows when paused so the simulator
    # still settles + marks existing positions but doesn't open new
    # ones. See module docstring.
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


def start_daemon(cfg: dict) -> Any:
    """Spawn the tennis-trading background thread. No-op when
    ``enabled`` is false. Config shape::

        tennis_trader:
          enabled: true
          repo_path: /root/tennis-forecast
          interval_seconds: 60
    """
    enabled = bool(cfg.get("enabled"))
    repo_path = cfg.get("repo_path", "/root/tennis-forecast")
    interval = int(cfg.get("interval_seconds", 60))

    def _run() -> None:
        log.info("tennis-bot starting (interval=%ds, repo=%s)",
                  interval, repo_path)
        _base.require_kalshi_creds()
        upstream = _load_upstream(repo_path)
        log.info("tennis-bot upstream loaded; entering tick loop")
        while True:
            try:
                _one_tick(upstream)
            except Exception:  # noqa: BLE001
                log.exception("tennis-bot tick failed")
            time.sleep(interval)

    return _base.spawn_daemon("tennis-bot", _run, enabled=enabled)
