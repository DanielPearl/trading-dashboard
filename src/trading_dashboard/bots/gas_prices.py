"""Gas-prices bot — runs the Retail Gas Prices (AAA national avg) loop
inside the dashboard process.

Shape A with one wrinkle — upstream's ``kalshi_gas_bot.main.Bot.run()``
spawns a second daemon thread for a 30s lightweight orderbook tick
(market hours only) alongside the main ``while True: tick()`` loop.
We don't need to do anything special: ``Bot.run()`` handles the
subthread spawn itself; we just hand it the bot and let it run.

The light-tick subthread fires ``_lightweight_tick()``, not
``tick()``, so it isn't gated by our ``gate_bot_tick`` patch. That's
fine — the light tick only updates orderbook state for already-open
positions; it doesn't open new ones, which is what the toggle is
about. If we ever need to gate it too, override ``_lightweight_tick``
the same way.
"""
from __future__ import annotations

import logging
from typing import Any

from . import _base


log = logging.getLogger("dashboard.gas-bot")

BOT_KEY = "gas-prices"


def start_daemon(cfg: dict) -> Any:
    """Spawn the gas-prices background thread. Config::

        gas_trader:
          enabled: true
          repo_path: /root/gas-prices
          config_path: /root/gas-prices/config/config.yaml
    """
    enabled = bool(cfg.get("enabled"))
    repo_path = cfg.get("repo_path", "/root/gas-prices")
    config_path = cfg.get("config_path", "/root/gas-prices/config/config.yaml")

    def _run() -> None:
        log.info("gas-bot starting (repo=%s)", repo_path)
        _base.require_kalshi_creds()
        _base.inject_sys_path(repo_path, subdir="src")

        from kalshi_gas_bot.config import load_config  # type: ignore  # noqa: E402
        from kalshi_gas_bot.main import Bot  # type: ignore  # noqa: E402

        upstream_cfg = load_config(config_path)
        # Note: gas-prices keeps artifacts under ./artifacts/, not
        # ./data/, so the resolve set differs slightly from the
        # other macro bots.
        _base.resolve_cfg_paths(
            upstream_cfg, repo_path,
            "env.log_path",
            "model.artifact_path",
            "execution.sim_db_path",
            "execution.decisions_log_path",
        )
        bot = Bot(upstream_cfg)
        _base.gate_bot_tick(bot, BOT_KEY, log)
        log.info("gas-bot upstream loaded; handing off to Bot.run()")
        bot.run()  # Bot.run() spawns its own _light_loop subthread.

    return _base.spawn_daemon("gas-bot", _run, enabled=enabled)
