"""Table-tennis bot — runs the in-process loop for table-tennis match
markets (TT). Shape B (hand-rolled tick), structurally identical to
the tennis bot.

Upstream's package layout mirrors tennis-forecast: imports resolve
from the repo root via the ``_REPO_ROOT = parents[2]`` trick in
``src/utils/config.py``, so we just inject the repo root onto
sys.path and the upstream's own path resolution does the rest. No
``resolve_cfg_paths`` call needed — the bot's config loader is
already CWD-independent.

Bot toggle: same pattern as tennis — pass empty rows to the
simulator when paused, so existing positions still settle but no
new ones open.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Callable

from . import _base
from .. import bot_state


log = logging.getLogger("dashboard.table-tennis-bot")

BOT_KEY = "table-tennis"

_prev_market_by_ticker: dict[str, dict] = {}


def _load_upstream(repo_path: str) -> dict[str, Callable[..., Any]]:
    _base.inject_sys_path(repo_path, subdir=None)

    from src.data import kalshi_markets  # type: ignore  # noqa: E402
    from src.dashboard.export_watchlist import (  # type: ignore  # noqa: E402
        build_watchlist_records,
        export as export_watchlist,
    )
    from src.trading.simulator import tick as simulator_tick  # type: ignore  # noqa: E402

    return {
        "fetch_markets": kalshi_markets.fetch_table_tennis_markets,
        "collapse_to_matches": kalshi_markets.collapse_to_matches,
        "write_live_state": kalshi_markets.write_live_state,
        "build_watchlist_records": build_watchlist_records,
        "export_watchlist": export_watchlist,
        "simulator_tick": simulator_tick,
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
        "table-tennis tick — %d markets / %d matches / %d rows / "
        "%d open / %d closed (P&L %+.3f)%s",
        len(raw_markets), len(records), len(rows),
        state["stats"].get("open_count", 0),
        state["stats"].get("total_closed", 0),
        state["stats"].get("total_realized_pnl", 0.0),
        ("" if enabled else " [PAUSED]"),
    )


def start_daemon(cfg: dict) -> Any:
    """Spawn the table-tennis daemon. Config::

        table_tennis_trader:
          enabled: true
          repo_path: /root/table-tennis-forecast
          interval_seconds: 60
    """
    enabled = bool(cfg.get("enabled"))
    repo_path = cfg.get("repo_path", "/root/table-tennis-forecast")
    interval = int(cfg.get("interval_seconds", 60))

    def _run() -> None:
        log.info("table-tennis-bot starting (interval=%ds, repo=%s)",
                  interval, repo_path)
        _base.require_kalshi_creds()
        upstream = _load_upstream(repo_path)
        log.info("table-tennis-bot upstream loaded; entering tick loop")
        while True:
            try:
                _one_tick(upstream)
            except Exception:  # noqa: BLE001
                log.exception("table-tennis-bot tick failed")
            time.sleep(interval)

    return _base.spawn_daemon("table-tennis-bot", _run, enabled=enabled)
