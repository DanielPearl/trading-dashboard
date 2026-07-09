"""WNBA bot — runs the WNBA game-outcome forecast loop inside the
dashboard process.

A sibling of bots/nba.py. Shape A (upstream Bot class with its own
run() loop) with a sport-adapter hook: after each tick(), we read
WNBA's sim.db and emit a sport-format watchlist.json + sim_state.json
so WNBA renders under the unified sport page layout alongside NBA,
tennis, table-tennis, and darts.

Unlike the NBA bot, the WNBA bot sources all of its data from ESPN's
public JSON endpoints (no stats.nba.com, no API key), so the
rate-limit spam filter / negative-fetch cache that nba.py installs are
unnecessary here.

WNBA team codes are variable length (GS, NY, LV, LA = 2 chars; CONN =
4; the rest = 3), so the ticker parser splits the concatenated
``{AWAY}{HOME}`` body against the known WNBA team set, using the
``-TEAM`` suffix to disambiguate.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Optional

from . import _base
from . import _sport_adapter

log = logging.getLogger("dashboard.wnba-bot")

BOT_KEY = "wnba"

# Current Kalshi WNBA team universe — used to split concatenated event
# bodies like "PHXGS" -> away=PHX, home=GS. Kept in sync with
# wnba_bot.data_sources.WNBA_TEAM_CODES.
_WNBA_TEAM_CODES = {
    "ATL", "CHI", "CONN", "DAL", "GS", "IND", "LV", "LA",
    "MIN", "NY", "PHX", "POR", "SEA", "TOR", "WSH",
}
# Alias → canonical map. Mostly ESPN spellings, plus Kalshi's own PDX
# ticker code for the Portland Fire (canonical set uses ESPN's POR —
# without the alias every LV@PDX-style ticker fails to parse and the
# game never reaches the watchlist).
_ESPN_TO_KALSHI = {"CON": "CONN", "WAS": "WSH", "LVA": "LV", "GSV": "GS",
                    "PDX": "POR"}


def _norm(code: str) -> str:
    code = (code or "").strip().upper()
    return _ESPN_TO_KALSHI.get(code, code)


# KXWNBAGAME-YYMMMDD{AWAY}{HOME}-{TEAM}, e.g. KXWNBAGAME-26JUN09PHXGS-GS
_TICKER_RE = re.compile(
    r"^KXWNBAGAME-\d{2}[A-Z]{3}\d{2}(?P<body>[A-Z]{4,8})-(?P<team>[A-Z]{2,4})$",
)


def _split_body(body: str, team: str) -> Optional[tuple]:
    """Split ``{AWAY}{HOME}`` into (away, home) using the -TEAM hint.
    Both halves are normalized so alias codes (Kalshi's PDX for POR)
    can't leave the two sides of one game with mismatched pairs."""
    body, team = body.upper(), _norm(team)
    if body.startswith(team) and body[len(team):]:
        return team, _norm(body[len(team):])
    if body.endswith(team) and body[:len(body) - len(team)]:
        return _norm(body[:len(body) - len(team)]), team
    # Fall back to known-team-set search.
    for i in range(2, len(body) - 1):
        a, h = _norm(body[:i]), _norm(body[i:])
        if a in _WNBA_TEAM_CODES and h in _WNBA_TEAM_CODES:
            return a, h
    return None


def _parse_wnba_ticker(ticker: str) -> Optional[dict]:
    """Decode a WNBA market ticker into the parsed-ticker shape the sport
    adapter expects. Returns None for malformed tickers so the adapter
    can skip them gracefully.

    ``base_id`` is everything up to (but not including) the final
    ``-TEAM`` suffix, so the two sides of a game collapse into one match.
    """
    m = _TICKER_RE.match(ticker or "")
    if not m:
        return None
    team = _norm(m.group("team"))
    split = _split_body(m.group("body"), team)
    if split is None:
        return None
    away, home = split
    if team not in (away, home):
        return None
    base_id = ticker[: ticker.rfind("-")]
    return {
        "base_id": base_id,
        "away": away,
        "home": home,
        "team_being_asked": team,
    }


def _sync_sport_json(repo_path: str, db_path: str,
                     watchlist_out: str, sim_state_out: str,
                     metrics_out: str | None = None) -> None:
    """Translate WNBA's sim.db into sport-shape JSON for the dashboard's
    sport renderer. Called after each Bot.tick(). When ``metrics_out`` is
    provided, also synthesizes a metrics.json from the latest
    ``model_snapshots`` row so the sport Models tab renders WNBA's
    accuracy / brier / log-loss / ROC AUC card the same way it does for
    the other sport bots.
    """
    _sport_adapter.translate(
        db_path=db_path,
        watchlist_out=watchlist_out,
        sim_state_out=sim_state_out,
        ticker_parser=_parse_wnba_ticker,
        tournament_label="WNBA",
        surface_label="",
        metrics_out=metrics_out,
    )


def _mirror_watchlist_live(watchlist_out: str, executor: Any) -> None:
    """Copy the just-written sport-shape watchlist.json into the live
    executor's output directory. Mirrors tennis.py's live mirror step
    so the LIVE dashboard's "Tradeable matches" panel shows the same
    rows the executor is considering.
    """
    import json as _json
    from pathlib import Path as _Path
    try:
        wl_src = _Path(watchlist_out)
        if not wl_src.exists():
            return
        with wl_src.open("r", encoding="utf-8") as f:
            payload = _json.load(f)
        live_dir = _Path(executor.state_path).parent
        live_dir.mkdir(parents=True, exist_ok=True)
        wl_live = live_dir / "watchlist.json"
        with wl_live.open("w", encoding="utf-8") as f:
            _json.dump(payload, f, separators=(",", ":"), default=str)
    except Exception:  # noqa: BLE001
        log.exception("wnba watchlist mirror to outputs-live failed")


def _run_live_executor(watchlist_out: str, executor: Any) -> None:
    """After a bot tick, load the freshly-written watchlist.json and
    hand its rows to the live executor. Mirrors the tennis pattern of
    running the executor synchronously right after each sport-json
    sync so sim and live evaluate the same rows.
    """
    from .. import bot_state
    import json as _json
    try:
        with open(watchlist_out, "r", encoding="utf-8") as f:
            payload = _json.load(f)
        rows = payload.get("rows") or []
    except (OSError, ValueError):
        log.exception("wnba watchlist read failed; skipping live tick")
        return
    # ``armed`` == the WNBA Home-tab toggle on the LIVE dashboard.
    # bot_state.is_bot_enabled is the SIM toggle; live has its own
    # store. Fall back to False if the live-store helper is absent.
    try:
        armed = bool(bot_state.is_bot_enabled(BOT_KEY, mode="live"))
    except TypeError:  # older bot_state without ``mode`` kwarg
        armed = False
    try:
        # The unified SportLiveExecutor signature takes both watchlist
        # rows and per-event records. WNBA's flat-row watchlist doesn't
        # ship record-level snapshots (per-market status / result), so
        # we pass an empty list — the executor falls back to direct
        # Kalshi lookups per open position.
        executor.tick(rows, [], armed=armed)
    except Exception:  # noqa: BLE001
        log.exception("wnba-live-executor tick failed")


def start_daemon(cfg: dict) -> Any:
    """Spawn the WNBA background thread. Config::

        wnba_trader:
          enabled: true
          repo_path: /root/wnba
          config_path: /root/wnba/config/config.yaml
          # Optional — defaults match upstream paths if omitted.
          db_path: /root/wnba/data/sim.db
          watchlist_json_path: /root/wnba/data/outputs/watchlist.json
          sim_state_path: /root/wnba/data/outputs/sim_state.json

    Live-trading config (adds a ``live:`` block). Presence of the block
    flips this daemon from sim-only to sim + live-executor after each
    tick::

        wnba_trader:
          ...
          live:
            dry_run: true
            sim_state_path: /root/wnba/data/outputs-live/sim_state.json
            max_open_positions: 4
            max_orders_per_day: 6
            contracts_per_order: 1
            min_edge_pp: 0.08
            max_entry_price_cents: 70
            min_entry_price_cents: 15
            price_deviation_cents: 3
            profit_lock_yes_bid_cents: 95
            require_pinnacle: true
    """
    from .. import bot_state

    enabled = bool(cfg.get("enabled"))
    repo_path = cfg.get("repo_path", "/root/wnba")
    config_path = cfg.get("config_path", "/root/wnba/config/config.yaml")
    db_path = cfg.get("db_path",
                      str(Path(repo_path) / "data" / "sim.db"))
    watchlist_out = cfg.get("watchlist_json_path",
                            str(Path(repo_path) / "data" / "outputs"
                                / "watchlist.json"))
    sim_state_out = cfg.get("sim_state_path",
                            str(Path(repo_path) / "data" / "outputs"
                                / "sim_state.json"))
    metrics_out = cfg.get("metrics_path",
                          str(Path(repo_path) / "data" / "processed"
                              / "artifacts" / "metrics.json"))
    live_cfg = cfg.get("live")  # None for sim-only

    def _run() -> None:
        log.info("wnba-bot starting (repo=%s, mode=%s)", repo_path,
                 "LIVE" if live_cfg else "SIM")
        _base.require_kalshi_creds()
        _base.inject_sys_path(repo_path, subdir="src")

        from wnba_bot.config import load_config  # type: ignore  # noqa: E402
        from wnba_bot.main import Bot  # type: ignore  # noqa: E402

        upstream_cfg = load_config(config_path)
        _base.resolve_cfg_paths(
            upstream_cfg, repo_path,
            "env.log_path",
            "model.artifact_path",
            "execution.sim_db_path",
            "execution.decisions_log_path",
        )
        bot = Bot(upstream_cfg)

        executor = None
        if live_cfg is not None:
            from . import wnba_live_executor  # noqa: E402
            state_path = live_cfg.get(
                "sim_state_path",
                str(Path(repo_path) / "data" / "outputs-live"
                    / "sim_state.json"),
            )
            executor = wnba_live_executor.WNBALiveExecutor(
                cfg=live_cfg, state_path=state_path,
            )

        _orig_tick = bot.tick

        def _gated_and_synced_tick(*args, **kwargs):
            if not bot_state.is_bot_enabled(BOT_KEY):
                log.info("wnba tick skipped — bot paused on dashboard")
                return None
            result = _orig_tick(*args, **kwargs)
            try:
                _sync_sport_json(repo_path, db_path, watchlist_out,
                                 sim_state_out, metrics_out)
            except Exception:  # noqa: BLE001
                log.exception("wnba sport-json sync failed (non-fatal)")
            # After the sport-adapter has written the fresh watchlist,
            # let the live executor consume it (dry-run by default).
            if executor is not None:
                _mirror_watchlist_live(watchlist_out, executor)
                _run_live_executor(watchlist_out, executor)
            return result

        bot.tick = _gated_and_synced_tick  # type: ignore[attr-defined]
        log.info("wnba-bot upstream loaded; handing off to Bot.run() "
                 "(sport-json sync%s after each tick)",
                 " + live executor" if executor else "")
        try:
            _sync_sport_json(repo_path, db_path, watchlist_out, sim_state_out)
        except Exception:  # noqa: BLE001
            log.exception("wnba initial sport-json sync failed (non-fatal)")
        bot.run()

    return _base.spawn_daemon("wnba-bot", _run, enabled=enabled)
