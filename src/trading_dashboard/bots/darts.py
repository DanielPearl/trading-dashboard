"""Darts bot — runs the in-process loop for PDC darts match markets.
Shape B (hand-rolled tick), structurally identical to tennis and
table-tennis.

Since 2026-07 the probability source is the sharp-book benchmark
itself: the devigged Pinnacle darts moneyline from the guest API
(kalshi_sdk.pinnacle.pinnacle_guest_probs_by_pair — The Odds API has
no darts feed at all). The upstream exporter still builds the rows
(titles, scores, 180s, live state), then the shared benchmark pass
(bots/_benchmark_rows.py) overwrites probs / edge / EV / gates so
every trade is Pinnacle-vs-Kalshi price discrepancy, same as MLB.
Rows without a Pinnacle line are WATCH-only — between PDC events
(when Pinnacle quotes nothing) the bot goes correctly dormant.

The "background discovery thread" noted in the earlier inventory is
inside the upstream's ``kalshi_markets`` module, not the live-monitor
script — it's a TTL cache that refreshes when stale, not a separate
thread. Nothing extra to spawn here.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Callable

from . import _base, _benchmark_rows
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


def _benchmark_lookup() -> dict:
    try:
        from kalshi_sdk.pinnacle import pinnacle_guest_probs_by_pair
    except ImportError:
        return {}
    try:
        return pinnacle_guest_probs_by_pair("darts") or {}
    except Exception:  # noqa: BLE001
        log.exception("darts benchmark lookup failed")
        return {}


def _one_tick(upstream: dict[str, Callable[..., Any]],
              live_executor: Any = None,
              paper_sim: bool = True) -> None:
    """One tick. Same sim/live process split as the world-cup and MLB
    bots: the SIM dashboard runs the paper simulator on data/outputs/,
    the LIVE dashboard runs the real-money executor on outputs-live/
    (dry_run inside the executor gates actual orders) and mirrors the
    watchlist there so the live page shows what the executor saw."""
    global _prev_market_by_ticker

    raw_markets = upstream["fetch_markets"]()
    new_prev = {m.get("ticker"): m for m in raw_markets if m.get("ticker")}
    records = upstream["collapse_to_matches"](
        raw_markets, prev_markets_by_ticker=_prev_market_by_ticker,
    )
    _prev_market_by_ticker = new_prev
    upstream["write_live_state"](records)

    rows = upstream["build_watchlist_records"]()
    matched = _benchmark_rows.apply_benchmark(
        rows, records, _benchmark_lookup(), win_verb="winning")

    enabled = bot_state.is_bot_enabled(BOT_KEY)
    rows_for_trading = rows if enabled else []

    state = None
    live_label = ""
    if paper_sim:
        upstream["export_watchlist"](records=rows)
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
        "darts tick — %d markets / %d matches / %d rows (%d benchmarked) / "
        "%d open / %d closed (P&L %+.3f)%s%s",
        len(raw_markets), len(records), len(rows), matched,
        state["stats"].get("open_count", 0),
        state["stats"].get("total_closed", 0),
        state["stats"].get("total_realized_pnl", 0.0),
        live_label,
        ("" if enabled else " [PAUSED]"),
    )


def start_daemon(cfg: dict) -> Any:
    """Spawn the darts daemon. Config::

        darts_trader:
          enabled: true
          repo_path: /root/darts-forecast
          interval_seconds: 60
          live:              # presence → this process runs the executor
            dry_run: true
            sim_state_path: /root/darts-forecast/data/outputs-live/sim_state.json
    """
    enabled = bool(cfg.get("enabled"))
    repo_path = cfg.get("repo_path", "/root/darts-forecast")
    interval = int(cfg.get("interval_seconds", 60))
    live_cfg = cfg.get("live")
    paper_sim = bool(cfg.get("paper_sim", live_cfg is None))

    def _run() -> None:
        from pathlib import Path
        log.info("darts-bot starting (interval=%ds, repo=%s, "
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
                bot_key=BOT_KEY, tournament="PDC", surface="Indoor",
                win_verb="winning")
        log.info("darts-bot upstream loaded; entering tick loop")
        while True:
            try:
                _one_tick(upstream, live_executor=executor,
                          paper_sim=paper_sim)
            except Exception:  # noqa: BLE001
                log.exception("darts-bot tick failed")
            time.sleep(interval)

    return _base.spawn_daemon("darts-bot", _run, enabled=enabled)
