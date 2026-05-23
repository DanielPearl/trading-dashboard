"""Unemployment-claims bot — runs the jobless-claims macro loop inside
the dashboard process.

Shape A (existing Bot class with its own run() loop): upstream's
``unemployment_claims_bot.main.Bot`` already implements
``while True: tick(); sleep()`` and spawns its own lightweight
orderbook subthread. We instantiate the bot once, monkey-patch
``tick`` to honour the Home-tab toggle, and hand control to
``Bot.run()``.

When the bot is paused via the toggle, ``tick`` short-circuits — no
mark-to-market, no settlement, no new positions. That's a coarser
gate than the tennis bot uses, but it's the right call here:
unemployment positions are held for weeks until the BLS release
resolves the market, so missing a few mark-to-market ticks during
a pause has no economic impact (the market settles whether we're
polling or not).
"""
from __future__ import annotations

import logging
from typing import Any

from . import _base


log = logging.getLogger("dashboard.unemployment-bot")

BOT_KEY = "unemployment-claims"


def start_daemon(cfg: dict) -> Any:
    """Spawn the unemployment-claims background thread. Config::

        unemployment_trader:
          enabled: true
          repo_path: /root/unemployment-claims
          config_path: /root/unemployment-claims/config/config.yaml
    """
    enabled = bool(cfg.get("enabled"))
    repo_path = cfg.get("repo_path", "/root/unemployment-claims")
    config_path = cfg.get("config_path",
                            "/root/unemployment-claims/config/config.yaml")

    def _run() -> None:
        log.info("unemployment-bot starting (repo=%s)", repo_path)
        _base.require_kalshi_creds()
        _base.inject_sys_path(repo_path, subdir="src")

        # Deferred imports — pulls in pandas, sklearn, the joblib
        # model. Paid lazily so a dashboard with this bot disabled
        # doesn't carry the cost.
        from unemployment_claims_bot.config import load_config  # type: ignore  # noqa: E402
        from unemployment_claims_bot.main import Bot  # type: ignore  # noqa: E402

        upstream_cfg = load_config(config_path)
        bot = Bot(upstream_cfg)
        _base.gate_bot_tick(bot, BOT_KEY, log)
        log.info("unemployment-bot upstream loaded; handing off to Bot.run()")
        bot.run()  # while True inside upstream — spawns its own light_loop

    return _base.spawn_daemon("unemployment-bot", _run, enabled=enabled)
