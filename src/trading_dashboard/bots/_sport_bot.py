"""Generic Shape-B sport bot — one parameterized tick loop.

mlb / nba / wnba / world_cup / darts / table_tennis used to be six
copies of the same ~185-line module differing only in a handful of
strings. Each per-sport module is now a :class:`SportSpec` plus
``start_daemon = make_start_daemon(SPEC)``; everything they shared
lives here once.

The loop per tick:

  1. fetch open Kalshi markets via the upstream repo's
     ``kalshi_markets`` module (loaded under a unique ``sys.modules``
     alias so the seven repos' ``src/`` packages don't collide),
  2. collapse to per-game records (previous tick's markets passed in
     for price-move columns),
  3. build tennis-shape watchlist rows via the upstream exporter —
     the mlb-family exporters take the fresh ``records``; the
     darts-family exporters read their own repo state and instead get
     a benchmark-rewrite pass (``_benchmark_rows.apply_benchmark``
     with the Pinnacle guest feed) over the built rows,
  4. trade them:
       SIM dashboard process  (paper_sim=True, no executor)
         → paper simulator writes data/outputs/ for the sim site.
       LIVE dashboard process (paper_sim=False, executor set)
         → real-money executor writes data/outputs-live/ (dry_run
           inside the executor gates actual orders) and the watchlist
           is mirrored there so the live page shows the same rows the
           executor evaluated.

Each process fetches Kalshi and builds rows itself, so neither
depends on the other being up. The Home-tab toggle read here is
per-process (the live service points KALSHI_BOT_STATES_PATH at
bot_states_live.json), so the LIVE page's toggle gates real orders
and the SIM page's toggle gates paper trades. Arming requires the
toggle to be EXPLICITLY on — a missing entry means dry-run, never
"default armed". When paused, rows still build (page stays live) but
the traders get an empty list, so existing positions keep settling
while no new ones open.
"""
from __future__ import annotations

import importlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from . import _base
from .. import bot_state


@dataclass(frozen=True)
class SportSpec:
    """Everything that differs between the Shape-B sport bots."""

    bot_key: str            # bot_state toggle key + executor bot_key
    name: str               # log prefix / daemon thread name
    alias: str              # sys.modules alias for the upstream src/
    fetch_fn: str           # kalshi_markets fetch-function name
    repo_default: str
    interval_default: int
    noun: str               # tick-log wording: "games" | "matches"
    # Constructor params for the shared SportLiveExecutor
    # (tournament / surface / win_verb / orphan_ticker_prefixes...).
    executor_kwargs: dict[str, Any] = field(default_factory=dict)
    # darts / table-tennis: rewrite rows onto the Pinnacle guest
    # benchmark. None → the exporter's own probs stand (mlb-family
    # exporters are already benchmark-driven upstream).
    benchmark_guest_sport: str | None = None
    benchmark_kwargs: dict[str, Any] = field(default_factory=dict)
    # mlb-family: build_watchlist_records(records) + live_state written
    # only in the sim process. darts-family (False): builder takes no
    # args and live_state is written every tick, before rows build.
    records_to_builder: bool = True


def _load_upstream(spec: SportSpec,
                   repo_path: str) -> dict[str, Callable[..., Any]]:
    _base.load_upstream_as_alias(repo_path, spec.alias, subdir="src")
    kalshi_markets = importlib.import_module(f"{spec.alias}.data.kalshi_markets")
    exporter = importlib.import_module(f"{spec.alias}.dashboard.export_watchlist")
    simulator = importlib.import_module(f"{spec.alias}.trading.simulator")
    return {
        "fetch_markets": getattr(kalshi_markets, spec.fetch_fn),
        "collapse_to_matches": kalshi_markets.collapse_to_matches,
        "write_live_state": kalshi_markets.write_live_state,
        "build_watchlist_records": exporter.build_watchlist_records,
        "export_watchlist": exporter.export,
        "simulator_tick": simulator.tick,
    }


def _benchmark_lookup(spec: SportSpec, log: logging.Logger) -> dict:
    try:
        from kalshi_sdk.pinnacle import pinnacle_guest_probs_by_pair
    except ImportError:
        return {}
    try:
        return pinnacle_guest_probs_by_pair(spec.benchmark_guest_sport) or {}
    except Exception:  # noqa: BLE001
        log.exception("%s benchmark lookup failed", spec.name)
        return {}


def _one_tick(spec: SportSpec, upstream: dict[str, Callable[..., Any]],
              log: logging.Logger, prev_holder: dict,
              live_executor: Any = None, paper_sim: bool = True) -> None:
    raw_markets = upstream["fetch_markets"]()
    new_prev = {m.get("ticker"): m for m in raw_markets if m.get("ticker")}
    records = upstream["collapse_to_matches"](
        raw_markets, prev_markets_by_ticker=prev_holder.get("prev") or {},
    )
    prev_holder["prev"] = new_prev

    matched = None
    if spec.records_to_builder:
        rows = upstream["build_watchlist_records"](records)
    else:
        from . import _benchmark_rows
        upstream["write_live_state"](records)
        rows = upstream["build_watchlist_records"]()
        matched = _benchmark_rows.apply_benchmark(
            rows, records, _benchmark_lookup(spec, log),
            **spec.benchmark_kwargs)

    enabled = bot_state.is_bot_enabled(spec.bot_key)
    rows_for_trading = rows if enabled else []

    state = None
    live_label = ""
    if paper_sim:
        if spec.records_to_builder:
            upstream["write_live_state"](records)
        upstream["export_watchlist"](rows)
        state = upstream["simulator_tick"](rows_for_trading, records)
    if live_executor is not None:
        entry = bot_state.get_all_states().get(spec.bot_key) or {}
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

    bench_bit = "" if matched is None else f"({matched} benchmarked) "
    log.info(
        "%s tick — %d markets / %d %s / %d rows %s/ "
        "%d open / %d closed (P&L %+.3f)%s%s",
        spec.name, len(raw_markets), len(records), spec.noun, len(rows),
        bench_bit,
        state["stats"].get("open_count", 0),
        state["stats"].get("total_closed", 0),
        state["stats"].get("total_realized_pnl", 0.0),
        live_label,
        ("" if enabled else " [PAUSED]"),
    )


def make_start_daemon(spec: SportSpec) -> Callable[[dict], Any]:
    """Build the module-level ``start_daemon(cfg)`` for one sport::

        <bot>_trader:
          enabled: true
          repo_path: <spec.repo_default>
          interval_seconds: <spec.interval_default>
          live:              # presence → this process runs the executor
            dry_run: true
            sim_state_path: .../outputs-live/sim_state.json
    """
    log = logging.getLogger(f"dashboard.{spec.name}-bot")

    def start_daemon(cfg: dict) -> Any:
        enabled = bool(cfg.get("enabled"))
        repo_path = cfg.get("repo_path", spec.repo_default)
        interval = int(cfg.get("interval_seconds", spec.interval_default))
        live_cfg = cfg.get("live")
        # Paper sim runs unless a live block is present (the sim
        # dashboard config has no live block; the live dashboard config
        # does). Overridable with an explicit paper_sim: key.
        paper_sim = bool(cfg.get("paper_sim", live_cfg is None))

        def _run() -> None:
            from pathlib import Path
            log.info("%s-bot starting (interval=%ds, repo=%s, "
                     "paper_sim=%s, live=%s)", spec.name, interval,
                     repo_path, paper_sim, "yes" if live_cfg else "no")
            _base.require_kalshi_creds()
            upstream = _load_upstream(spec, repo_path)
            executor = None
            if live_cfg is not None:
                from ._sport_live_executor import SportLiveExecutor
                state_path = live_cfg.get(
                    "sim_state_path",
                    str(Path(repo_path) / "data" / "outputs-live"
                        / "sim_state.json"))
                executor = SportLiveExecutor(
                    cfg=live_cfg, state_path=state_path,
                    bot_key=spec.bot_key, **spec.executor_kwargs)
            log.info("%s-bot upstream loaded; entering tick loop",
                     spec.name)
            prev_holder: dict = {}
            while True:
                try:
                    _one_tick(spec, upstream, log, prev_holder,
                              live_executor=executor, paper_sim=paper_sim)
                except Exception:  # noqa: BLE001
                    log.exception("%s-bot tick failed", spec.name)
                time.sleep(interval)

        return _base.spawn_daemon(f"{spec.name}-bot", _run, enabled=enabled)

    return start_daemon
