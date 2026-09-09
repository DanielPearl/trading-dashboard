"""GDP bot — the cpi_bot package run with ``run.reference: gdp``
(Atlanta Fed GDPNow → KXGDP quarterly ladders).

Same Shape A as bots/cpi.py: upstream's ``cpi_bot.main.Bot`` owns the
run() loop; we instantiate it with the PCE config, gate ``tick`` on
the Home-tab toggle, and let ``Bot.run()`` drive. One package, two
bots (user 2026-09-08: "a new bot for the pce using the same
mechanics as everything else").
"""
from __future__ import annotations

import logging
from typing import Any

from . import _base


log = logging.getLogger("dashboard.gdp-bot")

BOT_KEY = "gdp"


def start_daemon(cfg: dict) -> Any:
    """Spawn the PCE background thread. Config::

        gdp_trader:
          enabled: true
          repo_path: /root/cpi
          config_path: /root/cpi/config/config-gdp.yaml
    """
    enabled = bool(cfg.get("enabled"))
    repo_path = cfg.get("repo_path", "/root/cpi")
    config_path = cfg.get("config_path",
                          "/root/cpi/config/config-gdp.yaml")

    def _run() -> None:
        log.info("gdp-bot starting (repo=%s)", repo_path)
        _base.require_kalshi_creds()
        _base.inject_sys_path(repo_path, subdir="src")

        from cpi_bot.config import load_config  # type: ignore  # noqa: E402
        from cpi_bot.main import Bot  # type: ignore  # noqa: E402

        upstream_cfg = load_config(config_path)
        _base.resolve_cfg_paths(
            upstream_cfg, repo_path,
            "env.log_path",
            "execution.sim_db_path",
            "execution.decisions_log_path",
        )
        bot = Bot(upstream_cfg)
        _base.gate_bot_tick(bot, BOT_KEY, log)
        log.info("gdp-bot upstream loaded; handing off to Bot.run()")
        bot.run()

    return _base.spawn_daemon("gdp-bot", _run, enabled=enabled)
