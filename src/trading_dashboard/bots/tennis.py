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
from datetime import datetime, timezone
from pathlib import Path
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
    """Load tennis-forecast's pure functions under a unique alias so
    its ``src/`` package doesn't collide with table-tennis or darts'
    same-named ``src/`` packages. See ``_base.load_upstream_as_alias``.

    Also registers ``src.models.*`` -> ``tennis_src.models.*`` pickle
    aliases so joblib can unpickle the prematch bundle's
    WeightedEnsemble class (recorded under its training-time module
    path ``src.models.train_prematch_model``).
    """
    import importlib
    import sys
    _base.load_upstream_as_alias(repo_path, "tennis_src", subdir="src")
    _base.register_pickle_aliases(
        "tennis_src",
        ("train_prematch_model", "predict", "calibration_layer",
         "live_adjustment_model", "predict_inmatch"),
    )

    # Bundles trained via `python -m src.models.train_prematch_model`
    # pickle WeightedEnsemble under module `__main__` (because the
    # trainer ran AS __main__). The dashboard's __main__ is dashboard.py,
    # so joblib's unpickle `getattr(__main__, "WeightedEnsemble")` would
    # AttributeError without this shim.
    trainer_mod = importlib.import_module(
        "tennis_src.models.train_prematch_model"
    )
    main_mod = sys.modules.get("__main__")
    if main_mod is not None and hasattr(trainer_mod, "WeightedEnsemble"):
        setattr(main_mod, "WeightedEnsemble", trainer_mod.WeightedEnsemble)

    kalshi_markets = importlib.import_module("tennis_src.data.kalshi_markets")
    api_tennis_live = importlib.import_module("tennis_src.data.api_tennis_live")
    export_watchlist_mod = importlib.import_module(
        "tennis_src.dashboard.export_watchlist",
    )
    simulator_mod = importlib.import_module("tennis_src.trading.simulator")

    return {
        "fetch_tennis_markets": kalshi_markets.fetch_tennis_markets,
        "collapse_to_matches": kalshi_markets.collapse_to_matches,
        "write_live_state": kalshi_markets.write_live_state,
        "fetch_livescore": api_tennis_live.fetch_livescore,
        "build_state_by_key": api_tennis_live.build_state_by_key,
        "build_watchlist_records": export_watchlist_mod.build_watchlist_records,
        "export_watchlist": export_watchlist_mod.export,
        "simulator_tick": simulator_mod.tick,
    }


def _one_tick(upstream: dict[str, Callable[..., Any]],
                live_executor: Any = None) -> None:
    """Single tennis tick. The first six steps are mode-agnostic
    (fetch markets, build watchlist, export JSON). Step seven —
    'what does the simulator do with these rows' — is the only
    place sim and live diverge:

      sim  → upstream's paper simulator. Writes positions to
             ``data/outputs/sim_state.json``; never touches Kalshi.
      live → ``tennis_live_executor.TennisLiveExecutor`` (passed
             in as ``live_executor``). Places real Kalshi orders
             when its dry_run flag is off; writes positions to
             ``data/outputs-live/sim_state.json`` so it stays
             isolated from the sim state file.

    Bot-state toggle is honoured in both modes: paused → no new
    positions, existing positions still mark + settle normally.
    """
    global _prev_market_by_ticker

    raw_markets = upstream["fetch_tennis_markets"]()
    new_prev = {m.get("ticker"): m for m in raw_markets if m.get("ticker")}
    # Live in-match state from api-tennis.com — gated on env var
    # presence inside fetch_livescore. Empty dict when the key isn't
    # set or the API is down, in which case collapse_to_matches
    # falls back to the historic zero-filled defaults.
    live_state_by_key = upstream["build_state_by_key"](
        upstream["fetch_livescore"]()
    )
    records = upstream["collapse_to_matches"](
        raw_markets,
        prev_markets_by_ticker=_prev_market_by_ticker,
        live_state_by_key=live_state_by_key,
    )
    _prev_market_by_ticker = new_prev
    upstream["write_live_state"](records)

    rows = upstream["build_watchlist_records"]()
    upstream["export_watchlist"](records=rows)

    enabled = bot_state.is_bot_enabled(BOT_KEY)
    rows_for_sim = rows if enabled else []

    if live_executor is not None:
        # Live mode — call the executor instead of the paper sim.
        # Inside, dry_run still gates whether REAL orders fire.
        state = live_executor.tick(rows_for_sim, records)
        mode_label = (
            " [LIVE — DRY-RUN]" if getattr(live_executor, "dry_run", True)
            else " [LIVE — REAL ORDERS]"
        )
        # Mirror the watchlist to outputs-live/ so the live dashboard
        # config (which points there) shows what the bot is actually
        # considering. ``export_watchlist`` writes to outputs/ by
        # default per tennis-forecast's config; the live dashboard's
        # bot config reads outputs-live/watchlist.json. Without this
        # mirror the live "Tradeable matches" panel reads empty even
        # while the bot is happily trading.
        try:
            import json as _json
            from pathlib import Path as _Path
            live_dir = _Path(live_executor.state_path).parent
            live_dir.mkdir(parents=True, exist_ok=True)
            wl_live = live_dir / "watchlist.json"
            generated_at = (
                datetime.now(timezone.utc).isoformat(timespec="seconds")
            )
            with wl_live.open("w", encoding="utf-8") as f:
                _json.dump(
                    {"generated_at": generated_at, "rows": rows},
                    f, separators=(",", ":"), default=str,
                )
        except Exception:  # noqa: BLE001 — mirror is best-effort
            log.exception("watchlist mirror to outputs-live failed")
    else:
        state = upstream["simulator_tick"](rows_for_sim, records)
        mode_label = ""

    log.info(
        "tennis tick — %d kalshi markets / %d matches / %d watchlist rows "
        "/ %d open / %d closed (P&L %+.3f, ROI %s)%s%s",
        len(raw_markets), len(records), len(rows),
        state["stats"].get("open_count", 0),
        state["stats"].get("total_closed", 0),
        state["stats"].get("total_realized_pnl", 0.0),
        ("—" if state["stats"].get("roi") is None
         else f"{state['stats']['roi'] * 100:+.1f}%"),
        mode_label,
        ("" if enabled else " [PAUSED — no new positions]"),
    )


def start_daemon(cfg: dict) -> Any:
    """Spawn the tennis-trading background thread. No-op when
    ``enabled`` is false.

    Config shape (sim)::

        tennis_trader:
          enabled: true
          repo_path: /root/tennis-forecast
          interval_seconds: 60

    Config shape (live — adds the ``live`` block; presence of this
    block flips the daemon from paper-simulation to the real-money
    executor)::

        tennis_trader:
          enabled: true
          repo_path: /root/tennis-forecast
          interval_seconds: 60
          live:
            dry_run: true            # safety: never set false in code
            sim_state_path: /root/tennis-forecast/data/outputs-live/sim_state.json
            max_open_positions: 5
            max_orders_per_day: 20
            contracts_per_order: 1
            min_edge_pp: 0.05
            max_entry_price_cents: 70
    """
    enabled = bool(cfg.get("enabled"))
    repo_path = cfg.get("repo_path", "/root/tennis-forecast")
    interval = int(cfg.get("interval_seconds", 60))
    live_cfg = cfg.get("live")  # None for sim, dict for live

    def _run() -> None:
        log.info("tennis-bot starting (interval=%ds, repo=%s, "
                  "mode=%s)",
                  interval, repo_path,
                  "LIVE" if live_cfg else "SIM")
        _base.require_kalshi_creds()
        upstream = _load_upstream(repo_path)

        # Instantiate the live executor once at startup so it can
        # carry its rolling daily counter + Kalshi client across
        # ticks. None in sim mode so _one_tick falls back to the
        # paper simulator.
        executor = None
        if live_cfg is not None:
            from . import tennis_live_executor
            state_path = live_cfg.get(
                "sim_state_path",
                str(Path(repo_path) / "data" / "outputs-live"
                    / "sim_state.json"),
            )
            executor = tennis_live_executor.TennisLiveExecutor(
                cfg=live_cfg, state_path=state_path,
            )

        log.info("tennis-bot upstream loaded; entering tick loop")
        while True:
            try:
                _one_tick(upstream, live_executor=executor)
            except Exception:  # noqa: BLE001
                log.exception("tennis-bot tick failed")
            time.sleep(interval)

    return _base.spawn_daemon("tennis-bot", _run, enabled=enabled)
