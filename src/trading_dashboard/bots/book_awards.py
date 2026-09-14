"""Book Awards bot thread — National Book Award winner model.

Drives the upstream repo's tick() on an interval against the mode's
own sqlite; the standard pane / analytics / history plumbing reads
the results. The pane prices every listed KXBOOKAWARDS* market with
a live model % whether or not anything is armed.

ARMING (three gates, all required before a single live order):
  1. this service's config carries ``live.dry_run: false``
     (only dashboard-live.yaml does — the sim service has no live
     block, so it can never trade)
  2. the Book Awards toggle is ON in the Home tab
     (``bot_state.is_bot_enabled`` — defaults OFF for new bots)
  3. the upstream tick's own shared gates pass
     (``macro_entry_gate``, 25pp uncalibrated ceiling)
The toggle is re-read every tick, so flipping it OFF pauses live
entries at the next tick without a restart; paper pricing continues
either way.
"""
from __future__ import annotations

import logging
import threading
import time

from . import _base
from .. import bot_state

BOT_KEY = "book-awards"
REPO_DEFAULT = "/root/book-award-forecast"
# The NBF page + wiki features carry a 6h refresh TTL upstream;
# ticking hourly keeps Kalshi quotes fresh without re-scraping.
INTERVAL_S = 3600


def start_daemon(cfg: dict) -> threading.Thread | None:
    log = logging.getLogger("dashboard.book-awards")
    repo = cfg.get("repo_path") or REPO_DEFAULT
    db_path = cfg.get("db_path")
    if not db_path:
        log.warning("book-awards: no db_path configured — not starting")
        return None
    live_armed_cfg = (cfg.get("live") or {}).get("dry_run") is False

    def _loop() -> None:
        try:
            _base.load_upstream_as_alias(repo, "book_bot_src",
                                         subdir="src")
            import importlib
            main = importlib.import_module("book_bot_src.book_bot.main")
        except Exception:  # noqa: BLE001
            log.exception("book-awards upstream load failed")
            return
        log.info("book-awards started (interval=%ds, db=%s, "
                 "live-capable=%s)", INTERVAL_S, db_path,
                 live_armed_cfg)
        while True:
            try:
                # Toggle re-read every tick — the ONLY runtime switch
                # between paper and live once the live config arms.
                dry = not (live_armed_cfg
                           and bot_state.is_bot_enabled(BOT_KEY))
                c = main.tick(db_path, dry_run=dry)
                log.info("book-awards tick (%s) — %d series / %d "
                         "markets (%d priced, %d gap) / +%d paper "
                         "+%d live",
                         "paper" if dry else "LIVE",
                         c["series"], c["markets"], c["priced"],
                         c["gap"], c["paper_entries"],
                         c.get("live_entries", 0))
            except Exception:  # noqa: BLE001
                log.exception("book-awards tick failed")
            time.sleep(int(cfg.get("interval_seconds") or INTERVAL_S))

    t = threading.Thread(target=_loop, daemon=True,
                         name="book-awards-bot")
    t.start()
    return t
