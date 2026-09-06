"""Weather bot — runs the forecast-driven daily-weather trader inside
the dashboard process.

Same shape as the billboard / reality-leaks daemons: upstream's
``export()`` does the whole discover / forecast / price / gate /
trade / write cycle in one call and returns
``(watchlist_json_path, sim_db_path)``.

Order size is fixed at ``buy_criteria.CONTRACTS_PER_ORDER`` (1
contract = $1 per bet) inside the upstream trader — there is no knob
here or in the bot's YAML.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Callable

from . import _base
from .. import bot_state

log = logging.getLogger("dashboard.weather-bot")

BOT_KEY = "weather"


def _load_upstream(repo_path: str):
    import importlib
    _base.load_upstream_as_alias(repo_path, "weather_src", subdir="src")
    export_mod = importlib.import_module(
        "weather_src.weather_bot.dashboard.export_watchlist",
    )
    config_mod = importlib.import_module("weather_src.weather_bot.config")
    return export_mod.export, config_mod


def start_daemon(cfg: dict) -> Any:
    """Spawn the weather daemon.

    SIM service::

        weather_trader:
          enabled: true
          repo_path: /root/weather-forecast
          interval_seconds: 900

    LIVE service — the ``live:`` block is what arms real orders::

        weather_trader:
          enabled: true
          repo_path: /root/weather-forecast
          interval_seconds: 900
          live:
            dry_run: false      # <- the only switch that sends orders

    Three gates must align before a real order goes out, matching the
    tennis / nba / wnba arming model:

      1. ``enabled: true`` here (the daemon thread runs at all)
      2. the weather toggle ON in the LIVE dashboard's Home tab
      3. ``live.dry_run: false``

    The absence of a ``live:`` block forces paper mode upstream — the
    sim service cannot place a real order even if the bot repo's own
    config.yaml says ``dry_run: false``. Position sizing is not
    configurable from YAML at all: the upstream trader takes it from
    ``kalshi_sdk.buy_criteria.CONTRACTS_PER_ORDER`` (1 contract = $1),
    as it does for every gate.
    """
    enabled = bool(cfg.get("enabled"))
    repo_path = cfg.get("repo_path", "/root/weather-forecast")
    interval = int(cfg.get("interval_seconds", 900))
    live_cfg = cfg.get("live") or None

    def _run() -> None:
        mode = "LIVE" if live_cfg else "SIM"
        log.info("weather-bot starting (%s, interval=%ds, repo=%s)",
                 mode, interval, repo_path)
        export, config_mod = _load_upstream(repo_path)
        bot_cfg = config_mod.live_config(config_mod.load_config(), live_cfg)
        if bot_cfg.run.dry_run:
            log.info("weather-bot PAPER mode — no orders will be placed "
                     "(ledger: %s)", bot_cfg.paths.sim_db)
        else:
            log.warning("weather-bot ARMED — REAL $1 orders will be placed "
                        "when the Home toggle is on (ledger: %s)",
                        bot_cfg.paths.sim_db)
        log.info("weather-bot upstream loaded; entering tick loop")
        while True:
            try:
                if bot_state.is_bot_enabled(BOT_KEY):
                    json_path, _db_path = export(bot_cfg)
                    log.info("weather tick (%s) — wrote %s", mode, json_path)
                else:
                    log.info("weather tick skipped — paused on dashboard")
            except Exception:  # noqa: BLE001
                log.exception("weather-bot tick failed")
            time.sleep(max(60, interval))

    return _base.spawn_daemon("weather-bot", _run, enabled=enabled)
