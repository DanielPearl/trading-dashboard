"""World Cup bot — in-process sim-trading loop for KXWCGAME match
markets. Shape B (hand-rolled tick), structurally identical to the
table-tennis bot.

Each KXWCGAME event ("Portugal vs Spain: Regulation Time Moneyline")
carries three binary outcome markets (team A / team B / TIE). The
upstream exporter emits one watchlist row per outcome market, so the
binary buy-YES/buy-NO simulator logic applies unchanged — the same
buy gates, profit-lock exit, and settlement flow as tennis.

The model is pre-match only; the exporter's ``prematch`` gate blocks
opens once kickoff (the market's occurrence_datetime) is inside the
buffer, while existing positions keep settling in-play.

Bot toggle: same pattern as tennis — pass empty rows to the simulator
when paused, so existing positions still settle but no new ones open.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Callable

from . import _base
from .. import bot_state


log = logging.getLogger("dashboard.world-cup-bot")

BOT_KEY = "world-cup"

_prev_market_by_ticker: dict[str, dict] = {}


def _load_upstream(repo_path: str) -> dict[str, Callable[..., Any]]:
    """Load the world-cup upstream under a unique alias to avoid the
    ``src/`` namespace collision with the other sport bots."""
    import importlib
    _base.load_upstream_as_alias(repo_path, "world_cup_src", subdir="src")

    kalshi_markets = importlib.import_module(
        "world_cup_src.data.kalshi_markets",
    )
    export_watchlist_mod = importlib.import_module(
        "world_cup_src.dashboard.export_watchlist",
    )
    simulator_mod = importlib.import_module(
        "world_cup_src.trading.simulator",
    )

    return {
        "fetch_markets": kalshi_markets.fetch_world_cup_markets,
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

    rows = upstream["build_watchlist_records"](records)
    upstream["export_watchlist"](rows)

    enabled = bot_state.is_bot_enabled(BOT_KEY)
    rows_for_sim = rows if enabled else []
    state = upstream["simulator_tick"](rows_for_sim, records)

    log.info(
        "world-cup tick — %d markets / %d matches / %d rows / "
        "%d open / %d closed (P&L %+.3f)%s",
        len(raw_markets), len(records), len(rows),
        state["stats"].get("open_count", 0),
        state["stats"].get("total_closed", 0),
        state["stats"].get("total_realized_pnl", 0.0),
        ("" if enabled else " [PAUSED]"),
    )


def start_daemon(cfg: dict) -> Any:
    """Spawn the world-cup daemon. Config::

        world_cup_trader:
          enabled: true
          repo_path: /root/world-cup
          interval_seconds: 300
    """
    enabled = bool(cfg.get("enabled"))
    repo_path = cfg.get("repo_path", "/root/world-cup")
    interval = int(cfg.get("interval_seconds", 300))

    def _run() -> None:
        log.info("world-cup-bot starting (interval=%ds, repo=%s)",
                 interval, repo_path)
        _base.require_kalshi_creds()
        upstream = _load_upstream(repo_path)
        log.info("world-cup-bot upstream loaded; entering tick loop")
        while True:
            try:
                _one_tick(upstream)
            except Exception:  # noqa: BLE001
                log.exception("world-cup-bot tick failed")
            time.sleep(interval)

    return _base.spawn_daemon("world-cup-bot", _run, enabled=enabled)
