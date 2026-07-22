"""Hormuz Forecast bot — runs the peak-traffic macro loop inside the
dashboard process.

Shape A (same as unemployment-claims / gas / cpi): upstream's
``hormuz_forecast_bot.main.Bot`` implements ``while True: tick(); sleep()``.
We instantiate it once, monkey-patch ``tick`` to honour the Home-tab
toggle, and hand control to ``Bot.run()``.

SIM ONLY. The upstream bot's ``execution.dry_run`` is true and it only
ever writes sim.db — it never calls the live order path — so this daemon
stays in the SIM dashboard service permanently (no live: block).
"""
from __future__ import annotations

import logging
from typing import Any

from . import _base


log = logging.getLogger("dashboard.hormuz-bot")

BOT_KEY = "hormuz"


def start_daemon(cfg: dict) -> Any:
    """Spawn the Hormuz Forecast background thread. Config::

        hormuz_trader:
          enabled: true
          repo_path: /root/port-forecast
          config_path: /root/port-forecast/config/config.yaml
    """
    enabled = bool(cfg.get("enabled"))
    repo_path = cfg.get("repo_path", "/root/port-forecast")
    config_path = cfg.get("config_path",
                          "/root/port-forecast/config/config.yaml")

    def _run() -> None:
        log.info("hormuz-bot starting (repo=%s)", repo_path)
        _base.require_kalshi_creds()
        _base.inject_sys_path(repo_path, subdir="src")

        # Deferred imports — pulls in pandas/sklearn/scipy. Paid lazily
        # so a dashboard with this bot disabled doesn't carry the cost.
        from hormuz_forecast_bot.config import load_config  # type: ignore  # noqa: E402
        from hormuz_forecast_bot.main import Bot  # type: ignore  # noqa: E402

        upstream_cfg = load_config(config_path)
        # Upstream YAML uses CWD-relative paths; resolve them against the
        # repo so the bot reads its model and writes sim.db / model_card
        # where the dashboard's bot registry expects them.
        _base.resolve_cfg_paths(
            upstream_cfg, repo_path,
            "env.log_path",
            "model.artifact_path",
            "execution.sim_db_path",
            "execution.decisions_log_path",
        )
        bot = Bot(upstream_cfg)
        _base.gate_bot_tick(bot, BOT_KEY, log)
        log.info("hormuz-bot upstream loaded; handing off to Bot.run()")
        bot.run()

    return _base.spawn_daemon("hormuz-bot", _run, enabled=enabled)
