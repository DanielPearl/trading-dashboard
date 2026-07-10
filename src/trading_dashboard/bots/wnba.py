"""WNBA bot — in-process trading loop for KXWNBAGAME game markets.
Shape B (hand-rolled tick), structurally identical to the MLB bot.

2026-07 rearchitecture: previously Shape A — the upstream wnba_bot
package's Bot class (ESPN Elo model) ran its own loop into
sim.db and bots/_sport_adapter.py translated that into sport-shape
JSON. The probability source is now the sharp-book benchmark itself
(devigged Pinnacle WNBA moneyline via kalshi_sdk.pinnacle — guest feed
first, Odds-API cascade fallback), so the upstream repo grew the same
data/dashboard/trading modules the baseball repo uses and the legacy
Elo path is retired from trading. Rows without a benchmark line
are never buy-eligible, and out of season (no open KXWNBAGAME markets)
the bot idles cheaply.

Bot toggle: same pattern as tennis — pass empty rows to the simulator
when paused, so existing positions still settle but no new ones open.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Callable

from . import _base
from .. import bot_state


log = logging.getLogger("dashboard.wnba-bot")

BOT_KEY = "wnba"

_prev_market_by_ticker: dict[str, dict] = {}


def _load_upstream(repo_path: str) -> dict[str, Callable[..., Any]]:
    """Load the WNBA upstream under a unique alias to avoid the
    ``src/`` namespace collision with the other sport bots."""
    import importlib
    _base.load_upstream_as_alias(repo_path, "wnba_src", subdir="src")

    kalshi_markets = importlib.import_module(
        "wnba_src.data.kalshi_markets",
    )
    export_watchlist_mod = importlib.import_module(
        "wnba_src.dashboard.export_watchlist",
    )
    simulator_mod = importlib.import_module(
        "wnba_src.trading.simulator",
    )

    return {
        "fetch_markets": kalshi_markets.fetch_wnba_markets,
        "collapse_to_matches": kalshi_markets.collapse_to_matches,
        "write_live_state": kalshi_markets.write_live_state,
        "build_watchlist_records": export_watchlist_mod.build_watchlist_records,
        "export_watchlist": export_watchlist_mod.export,
        "simulator_tick": simulator_mod.tick,
    }


def _one_tick(upstream: dict[str, Callable[..., Any]],
              live_executor: Any = None,
              paper_sim: bool = True) -> None:
    """One tick. Same sim/live process split as the MLB bot: the SIM
    dashboard runs the paper simulator on data/outputs/, the LIVE
    dashboard runs the real-money executor on outputs-live/ (dry_run
    inside the executor gates actual orders) and mirrors the watchlist
    there so the live page shows what the executor saw."""
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
        "wnba tick — %d markets / %d games / %d rows / "
        "%d open / %d closed (P&L %+.3f)%s%s",
        len(raw_markets), len(records), len(rows),
        state["stats"].get("open_count", 0),
        state["stats"].get("total_closed", 0),
        state["stats"].get("total_realized_pnl", 0.0),
        live_label,
        ("" if enabled else " [PAUSED]"),
    )


def start_daemon(cfg: dict) -> Any:
    """Spawn the wnba daemon. Config::

        wnba_trader:
          enabled: true
          repo_path: /root/wnba
          interval_seconds: 120
          live:              # presence → this process runs the executor
            dry_run: true
            sim_state_path: /root/wnba/data/outputs-live/sim_state.json
    """
    enabled = bool(cfg.get("enabled"))
    repo_path = cfg.get("repo_path", "/root/wnba")
    interval = int(cfg.get("interval_seconds", 120))
    live_cfg = cfg.get("live")
    paper_sim = bool(cfg.get("paper_sim", live_cfg is None))

    def _run() -> None:
        from pathlib import Path
        log.info("wnba-bot starting (interval=%ds, repo=%s, "
                 "paper_sim=%s, live=%s)", interval, repo_path,
                 paper_sim, "yes" if live_cfg else "no")
        _base.require_kalshi_creds()
        upstream = _load_upstream(repo_path)
        executor = None
        if live_cfg is not None:
            from ._sport_live_executor import SportLiveExecutor
            state_path = live_cfg.get(
                "sim_state_path",
                str(Path(repo_path) / "data" / "outputs-live"
                    / "sim_state.json"))
            executor = SportLiveExecutor(
                cfg=live_cfg, state_path=state_path,
                bot_key=BOT_KEY, tournament="WNBA", surface="Basketball",
                win_verb="winning")
        log.info("wnba-bot upstream loaded; entering tick loop")
        while True:
            try:
                _one_tick(upstream, live_executor=executor,
                          paper_sim=paper_sim)
            except Exception:  # noqa: BLE001
                log.exception("wnba-bot tick failed")
            time.sleep(interval)

    return _base.spawn_daemon("wnba-bot", _run, enabled=enabled)
