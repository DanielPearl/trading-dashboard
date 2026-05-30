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
import time
from pathlib import Path
from typing import Any, Optional

from . import _base
from . import _sport_adapter


class _NbaApiFetchSpamFilter(logging.Filter):
    """Drop the upstream "game-log fetch for YYYY-YY failed" + the
    "skipping season … due to fetch error" follow-up.

    Digital Ocean cloud IPs are rate-limited (HTTP 403 or empty body)
    by stats.nba.com, so nba_api fetches reliably fail on the
    droplet. Upstream's nba_bot.data_sources retries 3× with backoff
    per season before giving up, then logs a "skipping season" line
    — once per season per tick, so for two open seasons that's
    8 warning lines per tick × ~12 ticks/hour during market hours
    ≈ 100/hour. The bot degrades gracefully (predictions still
    work from the existing cached training data), but the warning
    storm masks any real failure in the journal.

    Drop both message families at the logger level. The retries
    themselves are also negative-cached below (see
    ``_install_negative_fetch_cache``) so we additionally cut the
    network volume, not just the log volume.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        if "game-log fetch for" in msg and "failed (attempt" in msg:
            return False
        if "skipping season" in msg and "nba_api fetch failed" in msg:
            return False
        return True


def _patch_simple_imputers_for_sklearn_18(root: Any) -> int:
    """Walk a model graph and alias each SimpleImputer's old
    ``_fill_dtype`` attribute to the new ``_fit_dtype`` (or vice
    versa) so a pickle from sklearn 1.6 keeps working under 1.8.

    Returns the number of imputers patched (useful for logging /
    tests). Walks common sklearn container attrs without depending
    on the specific pipeline shape — safe to call on any object.
    """
    try:
        from sklearn.impute import SimpleImputer
    except ImportError:
        return 0
    seen: set[int] = set()
    patched = 0

    def _walk(obj: Any, depth: int = 0) -> None:
        nonlocal patched
        if obj is None or depth > 8:
            return
        oid = id(obj)
        if oid in seen:
            return
        seen.add(oid)
        if isinstance(obj, SimpleImputer):
            if hasattr(obj, "_fill_dtype") and not hasattr(obj, "_fit_dtype"):
                obj._fit_dtype = obj._fill_dtype  # type: ignore[attr-defined]
                patched += 1
            elif hasattr(obj, "_fit_dtype") and not hasattr(obj, "_fill_dtype"):
                obj._fill_dtype = obj._fit_dtype  # type: ignore[attr-defined]
                patched += 1
            return
        # Walk common sklearn-y container attrs. Includes a couple
        # of NBA-bot-specific names (``classifier``, ``meta_classifier``,
        # ``_model``) that wrap the actual Pipeline/estimator.
        for attr in (
            "steps", "named_steps", "estimator", "estimators_",
            "base_estimator", "calibrated_classifiers_",
            "final_estimator_", "transformer", "transformers_",
            "classifier", "meta_classifier", "_model", "pipeline",
            "model",
        ):
            v = getattr(obj, attr, None)
            if v is None:
                continue
            if isinstance(v, dict):
                for sub in v.values():
                    _walk(sub, depth + 1)
            elif isinstance(v, (list, tuple)):
                for sub in v:
                    if isinstance(sub, tuple) and len(sub) >= 2:
                        sub = sub[1]
                    _walk(sub, depth + 1)
            else:
                _walk(v, depth + 1)

    _walk(root)
    if patched:
        log.info("patched %d SimpleImputer instance(s) for sklearn-1.8 compat",
                 patched)
    return patched


def _install_negative_fetch_cache(repo_path: str) -> None:
    """Patch upstream's ``fetch_season_game_logs`` to short-circuit
    when a recent fetch attempt failed.

    The upstream function checks a positive cache (the season's CSV
    on disk) but has no negative cache, so a failing season gets
    re-attempted on every tick — three full HTTP round-trips with
    exponential backoff, all guaranteed to fail again on the
    droplet's blocked IP. We monkey-patch the function to remember
    the last failure timestamp per season and refuse to retry for
    24h. That drops the API volume from ~100 calls/hour to one
    per season per day.
    """
    import importlib
    try:
        data_sources = importlib.import_module(
            "nba_bot.data_sources",
        )
    except Exception:  # noqa: BLE001
        # Upstream not yet importable in this thread — caller is
        # responsible for ordering; just bail.
        return

    orig = getattr(data_sources, "fetch_season_game_logs", None)
    if orig is None or getattr(orig, "_negative_cached", False):
        return  # nothing to patch, or already patched

    failed_at: dict[str, float] = {}
    COOLDOWN_S = 24 * 3600

    def _wrapped(season: str, *args, **kwargs):
        last = failed_at.get(season)
        if last is not None and (time.time() - last) < COOLDOWN_S:
            raise RuntimeError(
                f"nba_api fetch for {season} suppressed by negative "
                f"cache (last failure was "
                f"{int((time.time() - last) / 60)} min ago; cooldown "
                f"24h to avoid hammering stats.nba.com).",
            )
        try:
            return orig(season, *args, **kwargs)
        except Exception:
            failed_at[season] = time.time()
            raise

    _wrapped._negative_cached = True  # type: ignore[attr-defined]
    data_sources.fetch_season_game_logs = _wrapped  # type: ignore[attr-defined]


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
                       watchlist_out: str, sim_state_out: str,
                       metrics_out: str | None = None) -> None:
    """Translate NBA's sim.db into sport-shape JSON for the
    dashboard's sport renderer. Called after each Bot.tick().

    When ``metrics_out`` is provided, also writes a synthesized
    ``metrics.json`` derived from the latest ``model_snapshots``
    row so the sport Models tab can render NBA's accuracy /
    brier / log-loss / ROC AUC card the same way it does for
    tennis. Without this, NBA's Models tab is "unavailable"
    because tennis-style metrics.json doesn't exist for it.
    """
    _sport_adapter.translate(
        db_path=db_path,
        watchlist_out=watchlist_out,
        sim_state_out=sim_state_out,
        ticker_parser=_parse_nba_ticker,
        tournament_label="NBA",
        surface_label="",
        metrics_out=metrics_out,
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
    # Where the synthesized metrics.json lands. Defaults to the
    # canonical training-artifacts directory under each bot — the
    # same place tennis / darts / table-tennis write their real
    # metrics.json. Both the sim and live dashboard configs point
    # at this same file (training artifacts are shared between
    # modes), so writing once benefits both pages.
    metrics_out = cfg.get("metrics_path",
                            str(Path(repo_path) / "data" / "processed"
                                / "artifacts" / "metrics.json"))

    def _run() -> None:
        log.info("nba-bot starting (repo=%s)", repo_path)
        _base.require_kalshi_creds()
        _base.inject_sys_path(repo_path, subdir="src")

        from nba_bot.config import load_config  # type: ignore  # noqa: E402
        from nba_bot.main import Bot  # type: ignore  # noqa: E402

        # Quiet the recurring stats.nba.com rate-limit warnings and
        # negative-cache the fetches that produce them. See the
        # filter / cache docstrings above for the why. Both are
        # opt-in here so nba.py's daemon owns the noise reduction
        # without modifying upstream code.
        logging.getLogger("nba_bot.data_sources").addFilter(
            _NbaApiFetchSpamFilter(),
        )
        _install_negative_fetch_cache(repo_path)

        upstream_cfg = load_config(config_path)
        _base.resolve_cfg_paths(
            upstream_cfg, repo_path,
            "env.log_path",
            "model.artifact_path",
            "execution.sim_db_path",
            "execution.decisions_log_path",
        )
        bot = Bot(upstream_cfg)

        # sklearn ≥1.7 renamed ``SimpleImputer._fill_dtype`` to
        # ``_fit_dtype``. The NBA bundle was pickled under sklearn
        # 1.6.1; predict_proba in 1.8 now raises ``AttributeError:
        # 'SimpleImputer' object has no attribute '_fill_dtype'``
        # on the first tick — observed 460× / day in the sim
        # diagnostic. Walk the loaded bot graph and alias the new
        # attribute name back to the old so the cached pickle keeps
        # working until the NBA pipeline is retrained under 1.8.
        _patch_simple_imputers_for_sklearn_18(bot)

        # Wrap tick() to (1) honour the Home-tab toggle, (2) refresh
        # the sport-shape JSONs after each tick so NBA renders under
        # the unified sport page layout.
        _orig_tick = bot.tick

        def _gated_and_synced_tick(*args, **kwargs):
            if not bot_state.is_bot_enabled(BOT_KEY):
                log.info("nba tick skipped — bot paused on dashboard")
                return None
            try:
                result = _orig_tick(*args, **kwargs)
            except RuntimeError as exc:
                # The negative-cache that throttles stats.nba.com calls
                # means NBA can't fetch fresh game-log data when the
                # weekly retrain timer fires. Upstream's _train_and_save
                # then raises "game-log panel is empty — cannot train"
                # because all seasons came back empty. The bot's main
                # model still works (it uses the previously-trained
                # one); only the periodic retrain attempt fails. Log
                # it once at WARNING (so the diagnosis collector
                # doesn't flag it as a real bug) and let the next
                # tick continue normally.
                msg = str(exc)
                if "game-log panel is empty" in msg:
                    log.warning(
                        "nba retrain skipped — no fresh game-log data "
                        "(stats.nba.com is rate-limited from this "
                        "droplet's IP, and no cached panel exists for "
                        "the active seasons). Predictions still use "
                        "the existing trained model; only the periodic "
                        "retrain is paused."
                    )
                    return None
                raise
            try:
                _sync_sport_json(repo_path, db_path, watchlist_out,
                                  sim_state_out, metrics_out)
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
