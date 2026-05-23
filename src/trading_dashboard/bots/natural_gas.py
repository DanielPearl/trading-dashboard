"""Natural-gas bot — in-process scheduler that runs the once-per-cycle
NatGas pipeline at the same three UTC times its old systemd timer
fired (06:00, 16:00, 20:30).

This is the only Tier-3 bot from the inventory: the upstream isn't a
``while True`` loop — it's a ``main()`` that does the full daily
fetch / score / write in one go and exits. The old droplet topology
was a oneshot service plus a systemd ``.timer`` unit. Inside the
dashboard process we replace the timer with a scheduler thread that
sleeps until the next firing time, runs main(), repeats.

Upstream-package quirk
----------------------
Unlike tennis / darts / etc., the natgas upstream uses bare
``from src.config import ...`` imports inside its files (not relative
imports), AND a top-level ``scripts/`` package. So we can't use the
``load_upstream_as_alias`` helper — the alias would still leave
those bare ``src.X`` imports resolving via sys.path. Instead we
sys.path-inject the repo root and let ``src`` / ``scripts`` resolve
to the natgas versions globally.

This works because none of the other folded-in bots use bare
``src.X`` imports anymore (they all went through the alias loader
in the live-monitor batch). If a future bot needs bare ``src``
imports too, both can't share — the second would shadow the first.
"""
from __future__ import annotations

import logging
import sys
import threading
import time
from datetime import datetime, time as dtime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from . import _base
from .. import bot_state


log = logging.getLogger("dashboard.natgas-bot")

BOT_KEY = "natural-gas"

# UTC times the upstream timer fired. Matches
# deploy/kalshi-natgas-bot.timer:
#   OnCalendar=*-*-* 06:00:00 UTC   (~1am ET)
#   OnCalendar=*-*-* 16:00:00 UTC   (~12pm ET)
#   OnCalendar=*-*-* 20:30:00 UTC   (~4:30pm ET — before market close)
_FIRING_TIMES_UTC: tuple[dtime, ...] = (
    dtime(6, 0, tzinfo=timezone.utc),
    dtime(16, 0, tzinfo=timezone.utc),
    dtime(20, 30, tzinfo=timezone.utc),
)


def _next_firing(now: datetime) -> datetime:
    """Return the next datetime in UTC at which a tick should fire,
    given the current UTC clock. Walks today's firing times first;
    if all have passed, rolls to tomorrow's first firing.
    """
    today = now.date()
    candidates = [
        datetime.combine(today, t, tzinfo=timezone.utc)
        for t in _FIRING_TIMES_UTC
    ]
    for c in candidates:
        if c > now:
            return c
    # All of today's firings are past; first firing tomorrow.
    tomorrow = today + timedelta(days=1)
    return datetime.combine(tomorrow, _FIRING_TIMES_UTC[0],
                             tzinfo=timezone.utc)


def _load_upstream(repo_path: str) -> Callable[[], int]:
    """Import natgas's main() with sys.path injection. See module
    docstring for why this can't use the alias loader.
    """
    root = Path(repo_path)
    if not root.exists():
        raise RuntimeError(
            f"natgas bot disabled: repo not found at {root}.",
        )
    p = str(root)
    if p not in sys.path:
        sys.path.insert(0, p)

    # Imports succeed iff natgas's repo_path is on sys.path and no
    # other bot has globally claimed the bare ``src`` / ``scripts``
    # namespaces. See module docstring.
    from scripts.run_daily import main  # type: ignore  # noqa: E402
    return main


def start_daemon(cfg: dict) -> Any:
    """Spawn the natgas scheduler thread. The thread is always alive
    (it's mostly sleeping); it just no-ops when paused via the
    Home-tab toggle, the same as the macro bots' coarse-grained gate.

    Config::

        natgas_trader:
          enabled: true
          repo_path: /root/peak-load   # legacy droplet path
    """
    enabled = bool(cfg.get("enabled"))
    repo_path = cfg.get("repo_path", "/root/peak-load")

    def _run() -> None:
        log.info("natgas-bot starting (repo=%s, firings_utc=%s)",
                  repo_path,
                  ", ".join(t.strftime("%H:%M") for t in _FIRING_TIMES_UTC))
        _base.require_kalshi_creds()
        natgas_main = _load_upstream(repo_path)
        log.info("natgas-bot upstream loaded; entering scheduler loop")

        while True:
            now = datetime.now(timezone.utc)
            target = _next_firing(now)
            wait_s = (target - now).total_seconds()
            log.info("natgas-bot sleeping %.0fs until next firing at %s",
                      wait_s, target.isoformat())
            time.sleep(max(1.0, wait_s))

            if not bot_state.is_bot_enabled(BOT_KEY):
                log.info("natgas-bot firing skipped — paused on dashboard")
                continue
            try:
                rc = natgas_main()
                log.info("natgas-bot run complete (rc=%s)", rc)
            except Exception:  # noqa: BLE001
                log.exception("natgas-bot run failed")

    return _base.spawn_daemon("natgas-bot", _run, enabled=enabled)
