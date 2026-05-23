"""NBA bot — runs the NBA game-outcome forecast loop inside the
dashboard process.

Shape A (upstream Bot class with its own run() loop) with a sport-
adapter hook: after each tick(), we read NBA's sim.db and emit a
sport-format watchlist.json + sim_state.json so NBA renders under
the unified sport page layout alongside tennis, table-tennis, and
darts.

NBA games are head-to-head ("Will Team A win?" YES/NO contracts),
which fits the sport schema's per-match shape naturally. The
adapter handles the side-pair → match collapse and the field
translation. See bots/_sport_adapter.py for the full mapping.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Optional

from . import _base
from . import _sport_adapter


log = logging.getLogger("dashboard.nba-bot")

BOT_KEY = "nba"

# NBA ticker pattern. Matches what the upstream nba_bot's
# market_scanner.py uses. Example:
#   KXNBAGAME-26MAY25NYKCLE-CLE
#                ↑YY↑MMM↑DD↑away↑home  ↑team_being_asked
_TICKER_RE = re.compile(
    r"^KXNBAGAME-\d{2}[A-Z]{3}\d{2}(?P<away>[A-Z]{3})(?P<home>[A-Z]{3})"
    r"-(?P<team>[A-Z]{3})$",
)


def _parse_nba_ticker(ticker: str) -> Optional[dict]:
    """Decode an NBA market ticker into the parsed-ticker shape the
    sport adapter expects. Returns None for malformed tickers so
    the adapter can skip them gracefully.

    ``base_id`` is everything up to (but not including) the final
    ``-TEAM`` suffix, so the two sides of a game ("-NYK" and "-CLE")
    collapse into one match row.
    """
    m = _TICKER_RE.match(ticker or "")
    if not m:
        return None
    away = m.group("away")
    home = m.group("home")
    team = m.group("team")
    if team not in (away, home):
        return None
    # base_id strips the "-<team>" suffix so both sides hash to the
    # same group.
    base_id = ticker[: ticker.rfind("-")]
    return {
        "base_id": base_id,
        "away": away,
        "home": home,
        "team_being_asked": team,
    }


def _sync_sport_json(repo_path: str, db_path: str,
                       watchlist_out: str, sim_state_out: str) -> None:
    """Translate NBA's sim.db into sport-shape JSON for the
    dashboard's sport renderer. Called after each Bot.tick().
    """
    _sport_adapter.translate(
        db_path=db_path,
        watchlist_out=watchlist_out,
        sim_state_out=sim_state_out,
        ticker_parser=_parse_nba_ticker,
        tournament_label="NBA",
        surface_label="",
    )


def start_daemon(cfg: dict) -> Any:
    """Spawn the NBA background thread. Config::

        nba_trader:
          enabled: true
          repo_path: /root/nba
          config_path: /root/nba/config/config.yaml
          # Optional — defaults match upstream paths if omitted.
          db_path: /root/nba/data/sim.db
          watchlist_json_path: /root/nba/data/outputs/watchlist.json
          sim_state_path: /root/nba/data/outputs/sim_state.json
    """
    from .. import bot_state

    enabled = bool(cfg.get("enabled"))
    repo_path = cfg.get("repo_path", "/root/nba")
    config_path = cfg.get("config_path", "/root/nba/config/config.yaml")
    db_path = cfg.get("db_path",
                       str(Path(repo_path) / "data" / "sim.db"))
    watchlist_out = cfg.get("watchlist_json_path",
                              str(Path(repo_path) / "data" / "outputs"
                                  / "watchlist.json"))
    sim_state_out = cfg.get("sim_state_path",
                              str(Path(repo_path) / "data" / "outputs"
                                  / "sim_state.json"))

    def _run() -> None:
        log.info("nba-bot starting (repo=%s)", repo_path)
        _base.require_kalshi_creds()
        _base.inject_sys_path(repo_path, subdir="src")

        from nba_bot.config import load_config  # type: ignore  # noqa: E402
        from nba_bot.main import Bot  # type: ignore  # noqa: E402

        upstream_cfg = load_config(config_path)
        _base.resolve_cfg_paths(
            upstream_cfg, repo_path,
            "env.log_path",
            "model.artifact_path",
            "execution.sim_db_path",
            "execution.decisions_log_path",
        )
        bot = Bot(upstream_cfg)

        # Wrap tick() to (1) honour the Home-tab toggle, (2) refresh
        # the sport-shape JSONs after each tick so NBA renders under
        # the unified sport page layout.
        _orig_tick = bot.tick

        def _gated_and_synced_tick(*args, **kwargs):
            if not bot_state.is_bot_enabled(BOT_KEY):
                log.info("nba tick skipped — bot paused on dashboard")
                return None
            result = _orig_tick(*args, **kwargs)
            try:
                _sync_sport_json(repo_path, db_path, watchlist_out,
                                  sim_state_out)
            except Exception:  # noqa: BLE001
                log.exception("nba sport-json sync failed (non-fatal)")
            return result

        bot.tick = _gated_and_synced_tick  # type: ignore[attr-defined]
        log.info("nba-bot upstream loaded; handing off to Bot.run() "
                  "(sport-json sync after each tick)")
        # Best-effort initial sync so the sport page renders something
        # even before the first tick fires (which can take a minute or
        # more for the macro NBA loop).
        try:
            _sync_sport_json(repo_path, db_path, watchlist_out,
                              sim_state_out)
        except Exception:  # noqa: BLE001
            log.exception("nba initial sport-json sync failed (non-fatal)")
        bot.run()

    return _base.spawn_daemon("nba-bot", _run, enabled=enabled)
