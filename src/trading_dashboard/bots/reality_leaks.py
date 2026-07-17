"""Reality Leaks bot — runs the leak-driven reality-TV paper trader
inside the SIM dashboard process.

Same trivially-simple shape as the billboard daemon: upstream's
``export()`` does the whole discover / poll-leaks / match / paper-
trade / write cycle in one call. PAPER ONLY — the upstream trader
writes sim.db and never places orders, so this daemon stays in the
sim service permanently (no live: block exists anywhere).

Kalshi access is public-data-only (no auth), so unlike the other
bots this daemon does NOT require Kalshi creds to start. Reddit
leak polling upgrades to OAuth automatically when REDDIT_CLIENT_ID /
REDDIT_CLIENT_SECRET are in the environment (they're in
/root/gas-prices/.env for the survivor bot already).
"""
from __future__ import annotations

import logging
import time
from typing import Any, Callable

from . import _base
from .. import bot_state


log = logging.getLogger("dashboard.reality-leaks-bot")

BOT_KEY = "reality-leaks"


def _load_upstream(repo_path: str) -> Callable[[], Any]:
    import importlib
    _base.load_upstream_as_alias(repo_path, "reality_leaks_src",
                                  subdir="src")
    export_mod = importlib.import_module(
        "reality_leaks_src.reality_leaks.dashboard.export_watchlist",
    )
    return export_mod.export


def start_daemon(cfg: dict) -> Any:
    """Spawn the reality-leaks daemon. Config::

        reality_leaks_trader:
          enabled: true
          repo_path: /root/reality-leaks
          interval_seconds: 300
    """
    enabled = bool(cfg.get("enabled"))
    repo_path = cfg.get("repo_path", "/root/reality-leaks")
    interval = int(cfg.get("interval_seconds", 300))

    def _run() -> None:
        log.info("reality-leaks-bot starting (interval=%ds, repo=%s)",
                  interval, repo_path)
        export = _load_upstream(repo_path)
        log.info("reality-leaks-bot upstream loaded; entering tick loop")
        while True:
            try:
                if bot_state.is_bot_enabled(BOT_KEY):
                    json_path, db_path = export()
                    log.info("reality-leaks tick — wrote %s", json_path)
                else:
                    log.info("reality-leaks tick skipped — paused on "
                             "dashboard")
            except Exception:  # noqa: BLE001
                log.exception("reality-leaks-bot tick failed")
            time.sleep(max(15, interval))

    return _base.spawn_daemon("reality-leaks-bot", _run, enabled=enabled)
