"""Billboard bot — runs the Billboard Hot 100 #1 (KXTOPSONG)
forecast loop inside the dashboard process.

Shape B but trivially simple — upstream's ``export()`` does the whole
fetch / predict / write / trade cycle in a single call, returning
``(csv_path, json_path)``. The bot module just wraps it in a sleep
loop, mirroring the upstream's ``scripts/run_live_monitor.py``.

The model.joblib is a large artifact; upstream's
``predict_candidates()`` lazy-loads it on first call via
``@lru_cache``. So importing this module costs nothing for memory
— the spike only happens on the first tick that actually finds
Billboard markets open.

LIVE trading armed 2026-07-16 per user sign-off: upstream's
``trading.live.dry_run: false`` places real marketable-limit IOC
orders (1 contract per pick, unified validator suite) and records
positions in live.db. This daemon now runs in the LIVE dashboard
service; the sim service's block is disabled. The Home-tab toggle
(bot_states_live.json) gates whether the tick runs at all.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Callable

from . import _base
from .. import bot_state


log = logging.getLogger("dashboard.billboard-bot")

BOT_KEY = "billboard"


def _load_upstream(repo_path: str) -> Callable[[], Any]:
    """Load billboard upstream under a unique alias. Package layout
    is ``src/billboard/...`` (deeper-nested), same shape as
    survivor.
    """
    import importlib
    _base.load_upstream_as_alias(repo_path, "billboard_src", subdir="src")
    export_mod = importlib.import_module(
        "billboard_src.billboard.dashboard.export_watchlist",
    )
    return export_mod.export


def start_daemon(cfg: dict) -> Any:
    """Spawn the billboard daemon. Config::

        billboard_trader:
          enabled: true
          repo_path: /root/billboard-charts
          interval_seconds: 300
    """
    enabled = bool(cfg.get("enabled"))
    repo_path = cfg.get("repo_path", "/root/billboard-charts")
    interval = int(cfg.get("interval_seconds", 300))

    def _run() -> None:
        log.info("billboard-bot starting (interval=%ds, repo=%s)",
                  interval, repo_path)
        _base.require_kalshi_creds()
        export = _load_upstream(repo_path)
        log.info("billboard-bot upstream loaded; entering tick loop "
                  "(81MB model lazy-loads on first market hit)")
        while True:
            try:
                if bot_state.is_bot_enabled(BOT_KEY):
                    csv_path, json_path = export()
                    log.info("billboard tick — wrote %s", json_path)
                else:
                    log.info("billboard tick skipped — paused on dashboard")
            except Exception:  # noqa: BLE001
                log.exception("billboard-bot tick failed")
            time.sleep(max(15, interval))

    return _base.spawn_daemon("billboard-bot", _run, enabled=enabled)
