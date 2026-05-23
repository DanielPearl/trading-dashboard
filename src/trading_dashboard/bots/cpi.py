"""CPI bot — runs the Consumer Price Index (Core CPI MoM) macro loop
inside the dashboard process.

Shape A — structurally identical to unemployment-claims. Upstream's
``cpi_bot.main.Bot`` owns the run() loop and spawns its own
lightweight orderbook subthread; we instantiate the bot, gate
``tick`` on the Home-tab toggle, and let ``Bot.run()`` drive.

CPI positions are held until the BLS monthly release resolves the
market, so the coarse "pause = skip whole tick" gate is fine.
"""
from __future__ import annotations

import logging
from typing import Any

from . import _base


log = logging.getLogger("dashboard.cpi-bot")

BOT_KEY = "cpi"


def start_daemon(cfg: dict) -> Any:
    """Spawn the CPI background thread. Config::

        cpi_trader:
          enabled: true
          repo_path: /root/cpi
          config_path: /root/cpi/config/config.yaml
    """
    enabled = bool(cfg.get("enabled"))
    repo_path = cfg.get("repo_path", "/root/cpi")
    config_path = cfg.get("config_path", "/root/cpi/config/config.yaml")

    def _run() -> None:
        log.info("cpi-bot starting (repo=%s)", repo_path)
        _base.require_kalshi_creds()
        _base.inject_sys_path(repo_path, subdir="src")

        from cpi_bot.config import load_config  # type: ignore  # noqa: E402
        from cpi_bot.main import Bot  # type: ignore  # noqa: E402

        upstream_cfg = load_config(config_path)
        # Same CWD-relative path quirk as unemployment-claims — see
        # bots/unemployment_claims.py for the rationale.
        _base.resolve_cfg_paths(
            upstream_cfg, repo_path,
            "env.log_path",
            "model.artifact_path",
            "execution.sim_db_path",
            "execution.decisions_log_path",
        )
        bot = Bot(upstream_cfg)
        _base.gate_bot_tick(bot, BOT_KEY, log)
        log.info("cpi-bot upstream loaded; handing off to Bot.run()")
        bot.run()

    return _base.spawn_daemon("cpi-bot", _run, enabled=enabled)
