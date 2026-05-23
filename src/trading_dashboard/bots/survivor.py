"""Survivor-elimination bot — runs the in-process loop for
Will-X-Be-Eliminated (KXSURVIVOR) markets.

Shape B (hand-rolled tick) but simpler than tennis: the upstream
exposes a single ``export()`` function that fetches + scores + writes
the watchlist in one call. No separate market fetch / simulator
step — the simulator runs inside ``export()``.

No bot_state.is_bot_enabled gate because the upstream's export()
doesn't split market-fetch from position-open. To gate trading
without losing the watchlist refresh, we'd need to fork the upstream
to expose a "read-only" mode. For now the bot toggle on the Home
tab is a no-op for survivor; flip the YAML's enabled: false to fully
stop the daemon if needed.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Callable

from . import _base


log = logging.getLogger("dashboard.survivor-bot")

BOT_KEY = "survivor"


def _load_upstream(repo_path: str) -> Callable[[], Any]:
    _base.inject_sys_path(repo_path, subdir=None)
    from src.survivor.dashboard.export_watchlist import export  # type: ignore  # noqa: E402
    return export


def start_daemon(cfg: dict) -> Any:
    """Spawn the survivor daemon. Config::

        survivor_trader:
          enabled: true
          repo_path: /root/survivor-elimination
          interval_seconds: 300
    """
    enabled = bool(cfg.get("enabled"))
    repo_path = cfg.get("repo_path", "/root/survivor-elimination")
    interval = int(cfg.get("interval_seconds", 300))

    def _run() -> None:
        log.info("survivor-bot starting (interval=%ds, repo=%s)",
                  interval, repo_path)
        # Survivor's upstream doesn't gate on Kalshi creds at startup
        # (its data sources include the survivor wiki); we don't
        # require them either, but if they're set the markets fetch
        # gets richer.
        export = _load_upstream(repo_path)
        log.info("survivor-bot upstream loaded; entering tick loop")
        while True:
            try:
                _, json_path = export()
                log.info("survivor tick — wrote %s", json_path)
            except Exception:  # noqa: BLE001
                log.exception("survivor-bot tick failed")
            time.sleep(interval)

    return _base.spawn_daemon("survivor-bot", _run, enabled=enabled)
