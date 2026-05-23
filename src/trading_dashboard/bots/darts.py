"""Darts bot — runs the in-process loop for PDC darts match markets.
Shape B (hand-rolled tick), structurally identical to tennis and
table-tennis.

The "background discovery thread" noted in the earlier inventory is
inside the upstream's ``kalshi_markets`` module, not the live-monitor
script — it's a TTL cache that refreshes when stale, not a separate
thread. Nothing extra to spawn here.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Callable

from . import _base
from .. import bot_state


log = logging.getLogger("dashboard.darts-bot")

BOT_KEY = "darts"

_prev_market_by_ticker: dict[str, dict] = {}


def _load_upstream(repo_path: str) -> dict[str, Callable[..., Any]]:
    """Load darts upstream under a unique alias to avoid the
    ``src/`` namespace collision with tennis and table-tennis."""
    import importlib
    _base.load_upstream_as_alias(repo_path, "darts_src", subdir="src")

    kalshi_markets = importlib.import_module("darts_src.data.kalshi_markets")
    export_watchlist_mod = importlib.import_module(
        "darts_src.dashboard.export_watchlist",
    )
    simulator_mod = importlib.import_module("darts_src.trading.simulator")

    return {
        "fetch_markets": kalshi_markets.fetch_darts_markets,
        "collapse_to_matches": kalshi_markets.collapse_to_matches,
        "write_live_state": kalshi_markets.write_live_state,
        "build_watchlist_records": export_watchlist_mod.build_watchlist_records,
        "export_watchlist": export_watchlist_mod.export,
        "simulator_tick": simulator_mod.tick,
    }


def _one_tick(upstream: dict[str, Callable[..., Any]]) -> None:
    global _prev_market_by_ticker

    raw_markets = upstream["fetch_markets"]()
    new_prev = {m.get("ticker"): m for m in raw_markets if m.get("ticker")}
    records = upstream["collapse_to_matches"](
        raw_markets, prev_markets_by_ticker=_prev_market_by_ticker,
    )
    _prev_market_by_ticker = new_prev
    upstream["write_live_state"](records)

    rows = upstream["build_watchlist_records"]()
    upstream["export_watchlist"](records=rows)

    enabled = bot_state.is_bot_enabled(BOT_KEY)
    rows_for_sim = rows if enabled else []
    state = upstream["simulator_tick"](rows_for_sim, records)

    log.info(
        "darts tick — %d markets / %d matches / %d rows / "
        "%d open / %d closed (P&L %+.3f)%s",
        len(raw_markets), len(records), len(rows),
        state["stats"].get("open_count", 0),
        state["stats"].get("total_closed", 0),
        state["stats"].get("total_realized_pnl", 0.0),
        ("" if enabled else " [PAUSED]"),
    )


def start_daemon(cfg: dict) -> Any:
    """Spawn the darts daemon. Config::

        darts_trader:
          enabled: true
          repo_path: /root/darts-forecast
          interval_seconds: 60
    """
    enabled = bool(cfg.get("enabled"))
    repo_path = cfg.get("repo_path", "/root/darts-forecast")
    interval = int(cfg.get("interval_seconds", 60))

    def _run() -> None:
        log.info("darts-bot starting (interval=%ds, repo=%s)",
                  interval, repo_path)
        _base.require_kalshi_creds()
        upstream = _load_upstream(repo_path)
        log.info("darts-bot upstream loaded; entering tick loop")
        while True:
            try:
                _one_tick(upstream)
            except Exception:  # noqa: BLE001
                log.exception("darts-bot tick failed")
            time.sleep(interval)

    return _base.spawn_daemon("darts-bot", _run, enabled=enabled)
