"""Shared helpers for in-process bot daemons.

Each bot module under this package follows one of two shapes:

  A) "Existing Bot class with its own run() loop" — the bot exposes a
     ``Bot`` class whose ``run()`` already does ``while True: tick();
     sleep()`` and optionally spawns its own subthreads (light ticks,
     etc.). We instantiate it, monkey-patch ``tick`` so it checks
     ``bot_state.is_bot_enabled`` before doing work, and let ``run()``
     drive the loop. Used by: unemployment-claims, cpi, nba, gas-prices.

  B) "Hand-rolled tick loop" — the upstream entry point is a script
     with an inline ``while True``; nothing reusable to call. We
     reproduce the per-tick body in our own ``_one_tick`` and spawn
     our own thread around it. Used by: tennis, survivor, table-tennis,
     darts, billboard, natural-gas (with a scheduler twist).

This module factors out the parts that are identical across both:

  * ``inject_sys_path`` — make an upstream repo's ``src/`` (or root)
    importable without pip-installing the repo. Mirrors what every
    bot's own ``run.py`` does at startup.
  * ``require_kalshi_creds`` — raise a clean error if the SDK env
    vars are missing, so the bot logs once at startup rather than
    looping forever on auth failures.
  * ``spawn_daemon`` — the daemon thread wrapper with try/except so
    a single tick crash doesn't kill the bot, and ``enabled=False``
    yields a no-op (no thread started, no upstream imported).
"""
from __future__ import annotations

import logging
import os
import sys
import threading
from pathlib import Path
from typing import Callable


def inject_sys_path(repo_path: str | Path, subdir: str | None = "src") -> Path:
    """Insert ``<repo_path>/<subdir>`` (or just ``<repo_path>``) at the
    head of ``sys.path`` so the upstream bot's package becomes
    importable. Returns the resolved path for logging.

    Raises if the path doesn't exist — better to fail fast at daemon
    startup than to log a stream of ImportError tracebacks on every
    tick.
    """
    root = Path(repo_path)
    target = root / subdir if subdir else root
    if not target.exists():
        raise RuntimeError(
            f"bot disabled: expected upstream package at {target}. "
            f"Either clone the bot's repo there or fix the repo_path "
            f"in dashboard.yaml.",
        )
    p = str(target)
    if p not in sys.path:
        # Prepend so upstream's ``src.X.Y`` / ``foo_bot.bar`` imports
        # resolve before any same-named module in site-packages.
        sys.path.insert(0, p)
    return target


def require_kalshi_creds() -> None:
    """Sanity check: the SDK reads ``KALSHI_API_KEY_ID`` and
    ``KALSHI_PRIVATE_KEY_PATH`` from the environment. Raise if either
    is missing so the operator notices on startup instead of silently
    burning ticks.
    """
    missing = [k for k in ("KALSHI_API_KEY_ID", "KALSHI_PRIVATE_KEY_PATH")
                if not os.environ.get(k, "").strip()]
    if missing:
        raise RuntimeError(
            f"bot needs Kalshi creds in env: {', '.join(missing)} "
            f"missing. The dashboard's systemd unit loads these from "
            f"/root/gas-prices/.env; check that file exists.",
        )


def spawn_daemon(name: str,
                  target: Callable[[], None],
                  enabled: bool = True) -> threading.Thread:
    """Run ``target`` once in a daemon thread named ``name``. If
    ``target`` is itself a ``while True`` loop (the common case),
    this is the bot's main loop; otherwise it's a one-shot setup
    routine that returns after spawning whatever it needs to.

    When ``enabled`` is False we don't call ``target`` at all — no
    thread, no upstream imports — so a dashboard with bots disabled
    in config has zero cost beyond the config read.

    Exceptions inside ``target`` are caught and logged via
    ``dashboard.<name>`` so a broken bot doesn't kill the dashboard
    process.
    """
    log = logging.getLogger(f"dashboard.{name}")

    def _wrapped() -> None:
        try:
            target()
        except Exception:  # noqa: BLE001
            # ``target`` is normally a long-lived loop; if it returns
            # at all, it's because of an unrecoverable startup error
            # (bad creds, missing repo). Log and let the thread exit;
            # operator will see the error.
            log.exception("%s daemon failed", name)

    t = threading.Thread(target=_wrapped, daemon=True, name=name)
    if enabled:
        t.start()
    else:
        log.info("%s disabled in config", name)
    return t


def resolve_cfg_paths(cfg: object, repo_path: str | Path,
                       *attr_paths: str) -> object:
    """Make relative path fields on an upstream config object absolute
    by prefixing them with ``repo_path``.

    Each upstream bot's config YAML uses CWD-relative paths
    (``data/model.pkl``, ``data/sim.db``, etc.) because their original
    systemd units set ``WorkingDirectory=/root/<bot>``. Inside the
    dashboard process the CWD is ``/root/trading-dashboard``, so
    those reads find a different (empty) ``data/`` dir and the bots
    silently train fresh models / write to orphaned DB files. This
    helper walks the config and resolves the listed dotted-name path
    attributes to absolute paths.

    ``attr_paths`` are dotted strings like ``"execution.sim_db_path"``
    so a bot module can declare its config schema in one line:

        resolve_cfg_paths(cfg, repo_path,
            "env.log_path",
            "model.artifact_path",
            "execution.sim_db_path",
            "execution.decisions_log_path",
        )

    No-op for fields that are already absolute, so it's safe to call
    multiple times.
    """
    root = Path(repo_path)
    for attr_path in attr_paths:
        parts = attr_path.split(".")
        obj = cfg
        for p in parts[:-1]:
            obj = getattr(obj, p)
        current = getattr(obj, parts[-1], None)
        if current and not Path(current).is_absolute():
            setattr(obj, parts[-1], str(root / current))
    return cfg


def gate_bot_tick(bot: object, bot_key: str, log: logging.Logger) -> None:
    """Wrap ``bot.tick`` so it short-circuits when the Home-tab toggle
    is off. Used by shape-A bots that hand their main loop over to an
    upstream ``Bot.run()`` — we can't reach into the loop to gate it,
    but we can replace the method it calls.

    When paused, the bot still goes through its normal sleep cycle
    (so the next ``bot_state`` check happens on schedule). Existing
    positions are NOT marked-to-market or settled during the pause —
    accepting that trade-off for now because macro-bot positions
    settle on data release whether we're polling or not. If a bot
    needs finer-grained gating (settle existing but don't open new),
    skip this helper and gate inside the bot module.
    """
    # Late import to avoid a circular dependency at module import time
    # (this module is imported from bots/__init__.py during dashboard
    # startup, before the package's bot_state is settled).
    from .. import bot_state

    _orig_tick = bot.tick  # type: ignore[attr-defined]

    def _gated_tick(*args, **kwargs):
        if not bot_state.is_bot_enabled(bot_key):
            log.info("%s tick skipped — bot paused on dashboard", bot_key)
            return None
        return _orig_tick(*args, **kwargs)

    bot.tick = _gated_tick  # type: ignore[attr-defined]
