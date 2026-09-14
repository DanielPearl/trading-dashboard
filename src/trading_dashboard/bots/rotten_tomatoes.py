"""Rotten Tomatoes bot thread — Beta-Binomial Tomatometer model.

Shape: simplest in the fleet. The upstream repo is PAPER-ONLY (no
order code exists there), so this wrapper just drives its tick() on
an interval against the mode's own sqlite and lets the standard
pane / analytics / history plumbing read the results. The Home
toggle isn't consulted for ticking because there is nothing to arm —
pricing the pane IS the bot's whole job until the calibration
record earns a live-executor conversation.
"""
from __future__ import annotations

import logging
import threading
import time

from . import _base

BOT_KEY = "rotten-tomatoes"
REPO_DEFAULT = "/root/rotten-tomatoes"
INTERVAL_S = 900  # matches the RT fetch TTL — faster would re-read cache


def start_daemon(cfg: dict) -> threading.Thread | None:
    log = logging.getLogger("dashboard.rotten-tomatoes")
    repo = cfg.get("repo_path") or REPO_DEFAULT
    db_path = cfg.get("db_path")
    if not db_path:
        log.warning("rotten-tomatoes: no db_path configured — not starting")
        return None

    def _loop() -> None:
        try:
            _base.load_upstream_as_alias(repo, "rt_bot_src", subdir="src")
            import importlib
            main = importlib.import_module("rt_bot_src.rt_bot.main")
        except Exception:  # noqa: BLE001
            log.exception("rotten-tomatoes upstream load failed")
            return
        log.info("rotten-tomatoes started (paper-only, interval=%ds, "
                 "db=%s)", INTERVAL_S, db_path)
        while True:
            try:
                c = main.tick(db_path, dry_run=True)
                log.info("rotten-tomatoes tick — %d series / %d markets "
                         "(%d priced, %d gap) / +%d paper",
                         c["series"], c["markets"], c["priced"],
                         c["gap"], c["paper_entries"])
            except Exception:  # noqa: BLE001
                log.exception("rotten-tomatoes tick failed")
            time.sleep(int(cfg.get("interval_seconds") or INTERVAL_S))

    t = threading.Thread(target=_loop, daemon=True,
                         name="rotten-tomatoes-bot")
    t.start()
    return t
