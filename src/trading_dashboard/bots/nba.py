"""NBA bot — runs the NBA game-outcome forecast loop inside the
dashboard process.

Shape A — structurally identical to unemployment-claims and cpi.
Upstream's ``nba_bot.main.Bot`` owns the run() loop; we gate tick().
"""
from __future__ import annotations

import logging
from typing import Any

from . import _base


log = logging.getLogger("dashboard.nba-bot")

BOT_KEY = "nba"


def start_daemon(cfg: dict) -> Any:
    """Spawn the NBA background thread. Config::

        nba_trader:
          enabled: true
          repo_path: /root/nba
          config_path: /root/nba/config/config.yaml
    """
    enabled = bool(cfg.get("enabled"))
    repo_path = cfg.get("repo_path", "/root/nba")
    config_path = cfg.get("config_path", "/root/nba/config/config.yaml")

    def _run() -> None:
        log.info("nba-bot starting (repo=%s)", repo_path)
        _base.require_kalshi_creds()
        _base.inject_sys_path(repo_path, subdir="src")

        from nba_bot.config import load_config  # type: ignore  # noqa: E402
        from nba_bot.main import Bot  # type: ignore  # noqa: E402

        upstream_cfg = load_config(config_path)
        # Same CWD-relative path quirk as the other macro bots.
        _base.resolve_cfg_paths(
            upstream_cfg, repo_path,
            "env.log_path",
            "model.artifact_path",
            "execution.sim_db_path",
            "execution.decisions_log_path",
        )
        bot = Bot(upstream_cfg)
        _base.gate_bot_tick(bot, BOT_KEY, log)
        log.info("nba-bot upstream loaded; handing off to Bot.run()")
        bot.run()

    return _base.spawn_daemon("nba-bot", _run, enabled=enabled)
