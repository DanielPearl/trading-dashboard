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


def _one_tick(upstream: dict[str, Callable[..., Any]],
              live_executor: Any = None,
              paper_sim: bool = True) -> None:
    """One tick. Which book it trades depends on the hosting process:

      SIM dashboard process  (paper_sim=True, no executor)
        → paper simulator writes data/outputs/ for the sim site.
      LIVE dashboard process (paper_sim=False, executor set)
        → real-money executor writes data/outputs-live/ for the live
          site (dry_run inside the executor gates actual orders); the
          watchlist is mirrored there so the live page shows the same
          rows the executor evaluated.

    Each process fetches Kalshi and builds rows itself, so neither
    depends on the other being up. The Home-tab toggle read here is
    per-process (the live service points KALSHI_BOT_STATES_PATH at
    bot_states_live.json), so the LIVE page's world-cup toggle gates
    real orders and the SIM page's toggle gates paper trades.
    """
    global _prev_market_by_ticker

    raw_markets = upstream["fetch_markets"]()
    new_prev = {m.get("ticker"): m for m in raw_markets if m.get("ticker")}
    records = upstream["collapse_to_matches"](
        raw_markets, prev_markets_by_ticker=_prev_market_by_ticker,
    )
    _prev_market_by_ticker = new_prev

    rows = upstream["build_watchlist_records"](records)

    enabled = bot_state.is_bot_enabled(BOT_KEY)
    rows_for_trading = rows if enabled else []

    state = None
    live_label = ""
    if paper_sim:
        upstream["write_live_state"](records)
        upstream["export_watchlist"](rows)
        state = upstream["simulator_tick"](rows_for_trading, records)
    if live_executor is not None:
        # Arming requires the toggle to be EXPLICITLY on in this
        # process's bot-states file (bot_states_live.json) — a missing
        # entry means dry-run, never "default armed".
        entry = bot_state.get_all_states().get(BOT_KEY) or {}
        armed = entry.get("enabled") is True
        live_state = live_executor.tick(rows_for_trading, records,
                                        armed=armed)
        live_label = (
            " [LIVE — DRY-RUN]" if getattr(live_executor, "dry_run", True)
            else " [LIVE — REAL ORDERS]")
        live_label += (f" live_open={live_state['stats'].get('open_count', 0)}"
                       f" live_pnl={live_state['stats'].get('total_realized_pnl', 0.0):+.2f}")
        # Watchlist mirror for the live page (reads outputs-live/).
        try:
            import json as _json
            from datetime import datetime as _dt, timezone as _tz
            live_dir = live_executor.state_path.parent
            live_dir.mkdir(parents=True, exist_ok=True)
            tmp = live_dir / "watchlist.json.tmp"
            tmp.write_text(_json.dumps(
                {"generated_at": _dt.now(_tz.utc).isoformat(
                    timespec="seconds"), "rows": rows}, default=str))
            tmp.replace(live_dir / "watchlist.json")
        except Exception:  # noqa: BLE001 — mirror is best-effort
            log.exception("watchlist mirror to outputs-live failed")
        if state is None:
            state = live_state

    log.info(
        "world-cup tick — %d markets / %d matches / %d rows / "
        "%d open / %d closed (P&L %+.3f)%s%s",
        len(raw_markets), len(records), len(rows),
        state["stats"].get("open_count", 0),
        state["stats"].get("total_closed", 0),
        state["stats"].get("total_realized_pnl", 0.0),
        live_label,
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
    live_cfg = cfg.get("live")  # present → this process runs the
    #                              real-money executor
    # Paper sim runs unless a live block is present (the sim dashboard
    # config has no live block; the live dashboard config does).
    # Overridable with an explicit paper_sim: key.
    paper_sim = bool(cfg.get("paper_sim", live_cfg is None))

    def _run() -> None:
        from pathlib import Path
        log.info("world-cup-bot starting (interval=%ds, repo=%s, "
                 "paper_sim=%s, live=%s)", interval, repo_path,
                 paper_sim, "yes" if live_cfg else "no")
        _base.require_kalshi_creds()
        upstream = _load_upstream(repo_path)
        executor = None
        if live_cfg is not None:
            from . import world_cup_live_executor
            state_path = live_cfg.get(
                "sim_state_path",
                str(Path(repo_path) / "data" / "outputs-live"
                    / "sim_state.json"))
            executor = world_cup_live_executor.WorldCupLiveExecutor(
                cfg=live_cfg, state_path=state_path)
        log.info("world-cup-bot upstream loaded; entering tick loop")
        while True:
            try:
                _one_tick(upstream, live_executor=executor,
                          paper_sim=paper_sim)
            except Exception:  # noqa: BLE001
                log.exception("world-cup-bot tick failed")
            time.sleep(interval)

    return _base.spawn_daemon("world-cup-bot", _run, enabled=enabled)
