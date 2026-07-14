"""HTTP server — request handler, serve loop, CLI entry point."""
from __future__ import annotations

import argparse
import json
import threading
import time
from datetime import datetime
from datetime import timezone
from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any
from typing import List
from .data import (
    filter_history_by_period,
    _compute_active_bets_totals,
    _live_kalshi_held_tickers,
    _merge_kalshi_with_local,
    _tennis_like_snapshot,
    build_snapshot,
    fetch_active_bets_with_marks,
    fetch_bet_history,
    fetch_global_summary,
    fetch_latest_model,
    fetch_latest_open_position,
    fetch_ticker_yes_prob_history,
    fetch_underlying_history,
    fetch_watchlist,
    pick_recent_market_view_ticker,
    resolve_bot_thresholds,
)
from .page import render_page
from .panels import PERIOD_OPTIONS, _period_days, default_filter_bot

import logging
log = logging.getLogger("dashboard")


# --------------------------------------------------------------------------- #
# Server
# --------------------------------------------------------------------------- #

# ── Cross-bot rollup cache ───────────────────────────────────────────
# The Summary / Home / History data (global summary cards, cross-bot
# active bets, cross-bot paper history, per-bot model cards) is
# identical regardless of which bot the URL selects, yet it used to be
# recomputed on EVERY page request — 13 bots × (bet-history scan +
# watchlist scan + summary aggregate) per click. Cached for a few
# seconds keyed by (mode, period); callers get shallow copies so the
# renderers can annotate rows without corrupting the shared cache
# (the HTTP server is threaded).
_ROLLUP_TTL_S = 10.0
_ROLLUP_CACHE: dict = {}
_ROLLUP_LOCK = threading.Lock()


def _cross_bot_rollup(bots: List[dict], *, period_days: int | None,
                       period_key: str, mode: str):
    key = (mode, period_key)
    now = time.time()
    with _ROLLUP_LOCK:
        hit = _ROLLUP_CACHE.get(key)
    if hit is not None and now - hit[0] < _ROLLUP_TTL_S:
        gs, gab, gh, bm = hit[1]
    else:
        gs, gab, gh, bm = _compute_cross_bot_rollup(
            bots, period_days=period_days, mode=mode)
        with _ROLLUP_LOCK:
            _ROLLUP_CACHE[key] = (now, (gs, gab, gh, bm))
    return (dict(gs), [dict(r) for r in gab],
            [dict(r) for r in gh], [dict(r) for r in bm])


def _compute_cross_bot_rollup(bots: List[dict], *, period_days: int | None,
                               mode: str):
    # Global cross-bot fetches (these power the Summary section
    # which is identical regardless of which bot is selected).
    global_summary = fetch_global_summary(bots,
                                           period_days=period_days)
    # On LIVE, cross-reference every bot's sim_state active
    # bets against Kalshi's real portfolio so the Home tab
    # Active bets card + table don't drift when the bot
    # executor's reconciliation lags. ``None`` = skip filter
    # (sim dashboard, or Kalshi fetch failed / no creds).
    _live_held_tickers = (_live_kalshi_held_tickers()
                           if mode == "live" else None)

    def _keep_on_kalshi(ab: dict) -> bool:
        """Filter passthrough — sim dashboard always keeps
        rows; live dashboard keeps only rows whose ticker is
        in the live-held set. Sport rows may carry the
        parent EVENT ticker while the portfolio lists the
        per-side MARKET ticker (event + "-XXX"), so accept
        a prefix match too."""
        if _live_held_tickers is None:
            return True
        t = str(ab.get("ticker") or "")
        if t in _live_held_tickers:
            return True
        return any(h.startswith(t + "-")
                   for h in _live_held_tickers)

    global_active_bets: List[dict] = []
    global_history: List[dict] = []
    # Per-bot models for the Performance tab — one card-row
    # per bot showing accuracy / precision / recall / F1.
    # Include bots whose sim.db doesn't exist yet (e.g. a
    # newly-registered bot before its first run) so they
    # show up in the grid with a "no snapshot yet"
    # placeholder rather than vanishing entirely. The
    # fetch_* helpers below all tolerate a missing DB.
    bot_models: List[dict] = []
    for b in bots:
        # Tennis bot doesn't have a sim.db, but it does have a
        # metrics.json / coefficients.json. Synthesize a model
        # dict for the card grid so the tennis bot shows up
        # alongside the Kalshi bots on the home page.
        if b.get("dashboard_type") in ("sport", "survivor", "billboard"):
            # Tennis, survivor, and billboard share the
            # sim_state.json shape — the survivor and
            # billboard adapters delegate
            # closed_positions_for_rollup to the tennis
            # adapter under the hood. The
            # model_summary_for_card signature is the
            # same across all three.
            from . import tennis as _tennis
            from . import survivor as _survivor
            from . import billboard as _billboard
            if b.get("dashboard_type") == "survivor":
                adapter = _survivor
            elif b.get("dashboard_type") == "billboard":
                adapter = _billboard
            else:
                adapter = _tennis
            if b.get("key") == "world-cup":
                # No metrics.json — the card summary comes
                # from the offline bake-off report, plus
                # the sim trader's win/loss ledger.
                from . import world_cup as _world_cup
                m = _world_cup.model_summary_for_card(
                    b.get("model_report_path"),
                    b.get("sim_state_path"))
            else:
                m = adapter.model_summary_for_card(
                    b.get("metrics_path"),
                    b.get("sim_state_path"),
                )
            bot_models.append({
                "bot": b,
                "model": m,
                "rules_text": "",
                "strike_count": 0,
                "strike_lo": None, "strike_hi": None,
            })
            # Pull open paper bets into the cross-bot
            # active-bets table. On LIVE, drop any row
            # whose ticker isn't in the actual Kalshi
            # portfolio (see ``_keep_on_kalshi``).
            if b.get("dashboard_type") == "sport":
                for ab in _tennis.active_bets_for_rollup(
                    b.get("sim_state_path"),
                    watchlist_path=b.get("watchlist_json_path"),
                ):
                    if not _keep_on_kalshi(ab):
                        continue
                    ab["_bot_name"] = b["name"]
                    ab["_bot_key"] = b["key"]
                    ab["_dashboard_type"] = b.get("dashboard_type") or "standard"
                    ab["_display"] = b.get("display") or {}
                    try:
                        from . import in_game as _ig
                        _pred = _ig.predict(b, ab)
                        if _pred is not None:
                            ab["_in_game"] = {
                                "live_prob_yes": _pred.live_prob_yes,
                                "confidence": _pred.confidence,
                                "action": _pred.recommended_action,
                                "reason": _pred.reason,
                            }
                    except Exception:  # noqa: BLE001
                        log.exception("in_game.predict in enrich failed")
                    global_active_bets.append(ab)
                # Real Kalshi positions in this bot's series
                # that the executor didn't open (manual /
                # external orders) — same union as the
                # per-bot watchlist page, so the home-page
                # Active bets shows every current contract.
                if mode == "live":
                    from .kalshi_client import get_open_positions
                    _kpos, _kerr = get_open_positions()
                    _prefixes = (b.get("series_prefixes")
                                 or ([b.get("series_ticker")]
                                     if b.get("series_ticker")
                                     else []))
                    if _kpos and _prefixes:
                        _covered = {
                            str(ab.get("ticker") or "")
                            for ab in global_active_bets
                            if ab.get("_bot_key") == b["key"]
                        }
                        _wl_payload = _tennis.load_watchlist(
                            b.get("watchlist_json_path"))
                        for ab in (
                            _tennis.kalshi_positions_to_active_bets(
                                _kpos, _wl_payload, _prefixes,
                                exclude_tickers=_covered,
                            )):
                            ab["_bot_name"] = b["name"]
                            ab["_bot_key"] = b["key"]
                            ab["_dashboard_type"] = "sport"
                            ab["_display"] = b.get("display") or {}
                            global_active_bets.append(ab)
            elif b.get("dashboard_type") == "billboard":
                # Billboard writes a real sim.db — same
                # readers as the standard bots below.
                for ab in fetch_active_bets_with_marks(b["db_path"]):
                    if not _keep_on_kalshi(ab):
                        continue
                    ab["_bot_name"] = b["name"]
                    ab["_bot_key"] = b["key"]
                    ab["_dashboard_type"] = "billboard"
                    ab["_display"] = b.get("display") or {}
                    global_active_bets.append(ab)
            # Closed paper bets into the cross-bot history
            # so hedge exits + natural settles surface on
            # the History tab. Same row shape the standard
            # ``fetch_bet_history`` produces. Billboard
            # closed bets live in its sim.db (standard
            # schema); the legacy
            # ``_billboard.closed_positions_for_rollup``
            # stub always returned [].
            # 2026-07-09: per-bot cap raised from 50 to
            # 10_000 so bots with hundreds of settled
            # paper closes (tennis at 174, WNBA at 99, NBA
            # at 28, etc.) fully contribute to the cross-
            # bot History tab. The period filter below
            # still trims by the user-selected window
            # (30d / 90d / all-time); this cap is just the
            # "don't OOM on a runaway ledger" safety.
            # LIVE history is REAL FILLS ONLY (user
            # 2026-07-10: "the history page should only
            # show real bets that were made"). That
            # supersedes the same-day paper-side fallbacks
            # that backfilled By-bot from paper ledgers
            # when a bot had no live closes — a bot that
            # never traded real money now simply shows
            # empty on the LIVE ledger, and the sport
            # rollup drops dry-run evaluations too.
            if b.get("dashboard_type") == "billboard":
                closed_iter = fetch_bet_history(
                    b["db_path"], limit=10_000)
            elif b.get("dashboard_type") == "sport":
                closed_iter = list(
                    adapter.closed_positions_for_rollup(
                        b.get("sim_state_path"), limit=10_000,
                        real_only=(mode == "live"),
                    ))
            else:
                closed_iter = list(
                    adapter.closed_positions_for_rollup(
                        b.get("sim_state_path"), limit=10_000,
                    ))
            for h in closed_iter:
                h["_bot_name"] = b["name"]
                h["_bot_key"] = b["key"]
                h["_dashboard_type"] = b.get("dashboard_type") or "standard"
                h["_display"] = b.get("display") or {}
                global_history.append(h)
            continue
        if b.get("dashboard_type") and b["dashboard_type"] != "standard":
            continue
        for ab in fetch_active_bets_with_marks(b["db_path"]):
            if not _keep_on_kalshi(ab):
                continue
            ab["_bot_name"] = b["name"]
            ab["_bot_key"] = b["key"]
            ab["_dashboard_type"] = b.get("dashboard_type") or "standard"
            # Attach the bot's display config so the
            # question column can be formatted in the bot's
            # native units (K claims vs $ vs ...).
            ab["_display"] = b.get("display") or {}
            # In-game model advisory — attached for the
            # active-bets table renderer. None for non-
            # sport bots; harmless to read.
            try:
                from . import in_game as _ig
                _pred = _ig.predict(b, ab)
                if _pred is not None:
                    ab["_in_game"] = {
                        "live_prob_yes": _pred.live_prob_yes,
                        "confidence": _pred.confidence,
                        "action": _pred.recommended_action,
                        "reason": _pred.reason,
                    }
            except Exception:  # noqa: BLE001
                log.exception("in_game.predict in enrich failed")
            global_active_bets.append(ab)
        # See the per-bot cap comment in the sport-family
        # branch above — same reasoning for the standard
        # sim.db bots.
        #
        # No paper-side fallback here: LIVE history is real
        # fills only (user 2026-07-10), so macro bots that
        # have never traded live show empty on the LIVE
        # ledger — their paper history stays on the sim
        # dashboard, which reads sim.db directly.
        _closed_rows = fetch_bet_history(
            b["db_path"], limit=10_000)
        for h in _closed_rows:
            h["_bot_name"] = b["name"]
            h["_bot_key"] = b["key"]
            h["_dashboard_type"] = b.get("dashboard_type") or "standard"
            h["_display"] = b.get("display") or {}
            global_history.append(h)
        m = fetch_latest_model(b["db_path"])
        # Pull contract rules from the bot's watchlist —
        # any one populated row will do (the rules_primary
        # text is the same template across the whole series).
        rules_text = ""
        strike_count = 0
        strike_lo = strike_hi = None
        bot_wl = fetch_watchlist(b["db_path"])
        for wv in bot_wl:
            if not rules_text:
                rt = (wv.get("rules_primary") or "").strip()
                if rt:
                    rules_text = rt
            sl = wv.get("strike_low")
            if sl is not None:
                strike_count += 1
                try:
                    slf = float(sl)
                    strike_lo = slf if strike_lo is None else min(strike_lo, slf)
                    strike_hi = slf if strike_hi is None else max(strike_hi, slf)
                except (TypeError, ValueError):
                    pass
        bot_models.append({
            "bot": b,
            "model": m,
            "rules_text": rules_text,
            "strike_count": strike_count,
            "strike_lo": strike_lo,
            "strike_hi": strike_hi,
        })
    global_active_bets.sort(key=lambda x: x.get("opened_at", ""), reverse=True)
    global_history.sort(key=lambda x: x.get("exited_at", ""), reverse=True)
    # Override the Summary's active-bets headline fields
    # with values computed straight from the global active
    # bets list (post-hide-settled, same per-row math as
    # the table renderer). Guarantees the Money spent /
    # Potential gain / Active bots / Active contracts
    # cards equal the column totals of the table just
    # below them.
    global_summary.update(_compute_active_bets_totals(global_active_bets))
    # Period-filter the history so the History tab agrees
    # with the rest of the period-aware UI. None → keep all.
    global_history = filter_history_by_period(global_history, period_days)
    return global_summary, global_active_bets, global_history, bot_models


class Handler(BaseHTTPRequestHandler):
    """Multi-bot HTTP handler.

    Each entry in ``bots`` is a dict with: key, name, db_path,
    decisions_path, available. The URL ``?bot=<key>`` selects which one
    to render; absent / unknown falls back to the first entry.
    """
    bots: List[dict] = []
    risk_caps: dict = {}
    edge_cfg: dict = {}
    validator_cfg: dict = {}
    hedge_cfg: dict = {}
    # "sim" or "live". Set by serve() at process start; read by the
    # render_page() call below so the header text, body data-mode
    # attribute, and meta warning all reflect which dashboard the
    # user is on.
    mode: str = "sim"
    # File paths to each live executor's sim_state.json. The Home
    # tab on the live dashboard concatenates open_positions from
    # all of these into the "Open Real-Money Positions" table.
    # Empty on the sim side.
    live_state_paths: List[str] = []

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        log.info("%s - %s", self.address_string(), format % args)

    def handle_one_request(self) -> None:  # noqa: D401
        """Wrap the base implementation to swallow benign network drops.

        Scanner traffic (CONNECT probes, /geoserver/web/ scans, etc.) and
        browser tabs that close mid-response routinely raise
        ConnectionResetError / BrokenPipeError from ``self.wfile.write``.
        The default BaseServer.handle_error path then dumps a full
        socketserver → http.server → do_GET traceback to stderr — which
        floods the journal with red herrings that look like real bugs to
        anyone reading the logs (or to the diagnosis collector). None of
        these are actionable: the client is gone before we noticed. We
        drop one informational line instead and move on.
        """
        try:
            super().handle_one_request()
        except (ConnectionResetError, BrokenPipeError,
                ConnectionAbortedError) as e:
            self.log_message("client disconnected mid-response (%s)",
                             type(e).__name__)

    def _resolve_bot(self, query: str) -> dict:
        from urllib.parse import parse_qs
        qs = parse_qs(query)
        requested = qs.get("bot", [None])[0]
        if requested:
            for b in self.bots:
                if b["key"] == requested:
                    return b
        # No ?bot= in the URL: default to the topmost bot in the
        # filter dropdown (user 2026-07-13), so opening Contracts →
        # Watchlist shows the same bot the picker shows first. Falls
        # back to the first registered bot if nothing is whitelisted.
        return default_filter_bot(self.bots) or self.bots[0]

    def do_GET(self) -> None:
        from urllib.parse import urlparse, parse_qs
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            try:
                bot = self._resolve_bot(parsed.query)
                # Period filter for the summary cards: ?period=day|week|month|year|all
                qs_top = parse_qs(parsed.query)
                period_key = qs_top.get("period", ["all"])[0]
                if period_key not in {k for k, _, _ in PERIOD_OPTIONS}:
                    period_key = "all"
                period_days = _period_days(period_key)
                # Active tab for the per-bot pane: ?tab=watchlist|model|activebet|rules
                tab_key = qs_top.get("tab", ["home"])[0]
                # `performance` was merged into `home`; legacy URLs
                # silently redirect to home so deep links keep working.
                if tab_key == "performance":
                    tab_key = "home"
                if tab_key not in {"home", "contracts", "watchlist",
                                    "models", "training", "history",
                                    "seasons"}:
                    tab_key = "home"
                # Models tab supports a pregame / ingame view toggle on
                # sport bots. Defaults to pregame; ignored for non-sport
                # bots (the Models panel only renders the pregame view
                # for them anyway).
                model_view = qs_top.get("model_view", ["pregame"])[0]
                if model_view not in {"pregame", "ingame"}:
                    model_view = "pregame"

                # Survivor-elimination uses the same JSON-source pattern
                # as the tennis bot (watchlist.json + metrics.json +
                # coefficients.json), but with a Survivor-shaped
                # per-contestant table. Dispatch early — the standard
                # render path expects a sim.db.
                if bot.get("dashboard_type") == "survivor":
                    from . import survivor as _survivor
                    survivor_tab = "models" if tab_key == "models" else "watchlist"
                    body = _survivor.render_page(
                        metrics_path=bot.get("metrics_path"),
                        coefficients_path=bot.get("coefficients_path"),
                        watchlist_path=bot.get("watchlist_json_path"),
                        sim_state_path=bot.get("sim_state_path"),
                        available_bots=self.bots,
                        current_bot_key=bot["key"],
                        tab_key=survivor_tab,
                    )
                    payload = body.encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                    return

                # Tennis-shape bots (tennis / table-tennis / darts) used
                # to dispatch into their own ``tennis.render_page`` here.
                # Phase 2a routes them through the standard render path
                # so every page is generated by ``render_page`` with the
                # same chrome and tab structure — only the data source
                # adapter differs. The branch below builds the
                # render_page args from watchlist.json + sim_state.json
                # and falls through to the cross-bot rollup + final
                # render at the bottom of this method.

                db_path = bot.get("db_path") or ""

                # Tennis-shape bots write JSON, not sim.db. Adapt their
                # watchlist + open positions into the standard row
                # schema so the shared ``render_page`` consumes them
                # exactly the way it consumes Kalshi event-bot data.
                if bot.get("dashboard_type") == "sport":
                    from . import tennis as _tennis
                    payload_wl = _tennis.load_watchlist(
                        bot.get("watchlist_json_path"))
                    watchlist = _tennis.build_standard_watchlist_rows(payload_wl)
                    bot_active_bets = _tennis.active_bets_for_rollup(
                        bot.get("sim_state_path"),
                        watchlist_path=bot.get("watchlist_json_path"),
                    )
                    # On the LIVE dashboard, "Active bets" must reflect
                    # what's actually in the Kalshi portfolio — the
                    # bot's local sim_state can drift when the executor's
                    # reconciliation lags (a position closed on Kalshi
                    # but sim_state still shows it open, or vice versa,
                    # is the exact failure the user hit on 2026-07-09
                    # with Vaccari vs Baybars). Cross-reference the sim
                    # rows against Kalshi's live /portfolio/positions:
                    # drop any row whose ticker isn't currently held on
                    # Kalshi. If the Kalshi fetch fails (no creds /
                    # transient error), leave the sim rows alone rather
                    # than blanking the section — same graceful-
                    # degradation stance as the balance helper. The SIM
                    # dashboard continues to render every sim_state open
                    # position since that IS the paper-trading
                    # portfolio.
                    if self.mode == "live":
                        from .kalshi_client import get_open_positions
                        kalshi_pos, _kalshi_err = get_open_positions()
                        if kalshi_pos is not None:
                            held_tickers = {p.get("ticker")
                                             for p in kalshi_pos
                                             if p.get("ticker")}

                            def _held(ab_ticker: str) -> bool:
                                # sim_state rows carry either the real
                                # market ticker or the parent EVENT
                                # ticker (tennis-era rollup) — a held
                                # market ticker always startswith its
                                # event ticker + "-".
                                if ab_ticker in held_tickers:
                                    return True
                                return any(h.startswith(ab_ticker + "-")
                                           for h in held_tickers)

                            bot_active_bets = [
                                ab for ab in bot_active_bets
                                if _held(str(ab.get("ticker") or ""))
                            ]
                            # UNION with real Kalshi positions in this
                            # bot's series that the executor didn't
                            # open (manual buys, external orders) —
                            # per user 2026-07-09: "Active bets should
                            # show the bets that were bought", e.g. a
                            # hand-bought KXNBASUMMERGAME position must
                            # appear on the NBA page. Rendered through
                            # the same tennis-column row shape.
                            prefixes = (bot.get("series_prefixes")
                                        or ([bot.get("series_ticker")]
                                            if bot.get("series_ticker")
                                            else []))
                            covered = {str(ab.get("ticker") or "")
                                       for ab in bot_active_bets}
                            bot_active_bets += (
                                _tennis.kalshi_positions_to_active_bets(
                                    kalshi_pos, payload_wl, prefixes,
                                    exclude_tickers=covered,
                                ))
                    for ab in bot_active_bets:
                        ab.setdefault("_display", bot.get("display") or {})
                    # Sport bots have no per-bot "latest open position"
                    # singleton concept — the rollup is the source of
                    # truth.
                    latest_active = (bot_active_bets[0]
                                      if bot_active_bets else None)
                    model = None
                elif bot.get("dashboard_type") == "billboard":
                    # Billboard mirrors the tennis pattern: watchlist
                    # rows come from watchlist.json (synthesised into
                    # the standard schema by the billboard adapter),
                    # active bets / latest_active are always empty
                    # (the bot is advisory-only), and model is None so
                    # the standard _render_current_prediction returns
                    # early. Everything else flows through the shared
                    # render_page so the page is visually identical to
                    # retail-gas-prices.
                    from . import billboard as _billboard
                    payload_wl = _billboard.load_watchlist(
                        bot.get("watchlist_json_path"))
                    watchlist = _billboard.build_standard_watchlist_rows(
                        payload_wl)
                    # Live trader writes a standard sim.db; share the
                    # readers used by gas/claims/CPI so active bets +
                    # latest open render the same way.
                    bot_active_bets = fetch_active_bets_with_marks(db_path)
                    for ab in bot_active_bets:
                        ab.setdefault("_display", bot.get("display") or {})
                    latest_active = fetch_latest_open_position(db_path)
                    model = None
                else:
                    # Bot-scoped fetches for standard sim.db bots.
                    model = fetch_latest_model(db_path)
                    latest_active = fetch_latest_open_position(db_path)
                    watchlist = fetch_watchlist(db_path)
                    bot_active_bets = fetch_active_bets_with_marks(db_path)
                    for ab in bot_active_bets:
                        ab.setdefault("_display", bot.get("display") or {})
                # Open positions — fetched here (instead of just before
                # render) so we can pass their tickers into the Kalshi
                # fetch and force their parent events into the watchlist
                # ladder, even if they're on a different event than the
                # most-imminent one.
                open_position_tickers = {
                    ab.get("ticker") for ab in bot_active_bets
                    if ab.get("ticker")
                }
                # Will fall back to Kalshi markets below if `watchlist`
                # comes up empty (bot service not writing market_views,
                # or the bot is currently between events). Done after
                # the Kalshi fetch since both share the cache.
                # Local snapshots — kept around as the secondary source
                # for the hero current-value (used as a final fallback
                # if Kalshi creds are missing).  Tennis-shape bots have
                # no underlying time series.
                if db_path and Path(db_path).exists():
                    underlying_history = fetch_underlying_history(
                        db_path, hours=7 * 24, max_points=5000,
                    )
                else:
                    underlying_history = []
                # Chart source: Kalshi's implied-underlying forecast,
                # derived from the strike ladder. Same series Kalshi
                # itself plots on every market page — for each
                # candle timestamp, find the strike where YES=50% and
                # interpolate. Per-bot resolution comes from
                # display.chart_period_minutes (gas bots → daily;
                # jobless → 1-min so every recorded change shows up).
                kalshi_history: List[dict] = []
                atm_market: dict | None = None
                kalshi_markets: List[dict] = []
                contract_open_ts: float | None = None
                contract_close_ts: float | None = None
                event_title: str | None = None
                series_ticker = bot.get("series_ticker")
                chart_period = int(((bot.get("display") or {}).get(
                    "chart_period_minutes")) or 60)
                # Tennis-shape bots don't have an underlying price
                # series — the watchlist is per-match. Skip the Kalshi
                # candlestick fetch entirely so the hero renders an
                # empty chart frame rather than 500ing.
                if (series_ticker
                        and bot.get("dashboard_type") not in ("sport",
                                                              "billboard")):
                    from . import kalshi_client
                    # Sport series like KXNBAGAME have many concurrent
                    # open events (one per game on the slate). Narrowing
                    # to the most-imminent event hides the rest of the
                    # slate from the watchlist; flag those bots so the
                    # client returns every market with a future close.
                    all_open_events = bot.get("key") in {
                        "nba", "wnba", "tennis", "table-tennis", "darts",
                        "world-cup", "mlb",
                    }
                    try:
                        (kalshi_history, atm_market, kalshi_markets,
                         contract_open_ts, contract_close_ts,
                         event_title) = (
                            kalshi_client.fetch_underlying_history(
                                series_ticker,
                                period_minutes=chart_period,
                                extra_tickers=open_position_tickers,
                                all_open_events=all_open_events,
                            )
                        )
                    except Exception:  # noqa: BLE001
                        log.exception("kalshi candlestick fetch failed")
                        kalshi_history, atm_market = [], None
                        kalshi_markets, contract_open_ts = [], None
                        contract_close_ts = None
                        event_title = None
                # Probability series for the watchlist hero chart —
                # picks the most-relevant ticker (active bet → ATM →
                # first watchlist row → most-recently-updated ticker
                # in market_views) and pulls its YES-prob history from
                # market_views. The chart pins y to 0-100¢ since any
                # binary contract's value is bounded by the ticker's
                # 0..100 price range. When the first pick has no
                # recent data (e.g. mid-event-rollover, Kalshi's ATM
                # not yet scored locally) we fall back to whichever
                # ticker was most recently written so the chart still
                # plots something useful.
                prob_history: List[dict] = []
                if (db_path
                        and bot.get("dashboard_type") not in ("sport",
                                                              "survivor",
                                                              "billboard")):
                    candidates: List[str] = []
                    if latest_active and latest_active.get("ticker"):
                        candidates.append(latest_active["ticker"])
                    if atm_market and atm_market.get("ticker"):
                        candidates.append(atm_market["ticker"])
                    if watchlist:
                        t = (watchlist[0] or {}).get("ticker")
                        if t:
                            candidates.append(t)
                    fallback = pick_recent_market_view_ticker(db_path)
                    if fallback:
                        candidates.append(fallback)
                    seen: set = set()
                    for t in candidates:
                        if not t or t in seen:
                            continue
                        seen.add(t)
                        prob_history = fetch_ticker_yes_prob_history(
                            db_path, t, hours=7 * 24)
                        if prob_history:
                            break

                # Chart shows only the current event's data. The local
                # model_snapshots merge was retired with the 5-day view.

                # Hybrid watchlist: Kalshi spine + merged local data.
                # Kalshi gives us the canonical, always-up-to-date strike
                # ladder for the currently-open event. Local market_views
                # adds the bot's model probabilities, EV, verdict, etc.
                # — but only for markets the bot has actually scored.
                # If the bot is between events, the local rows are stale
                # (different event's tickers); the Kalshi spine ensures
                # the table still reflects today's market.
                if kalshi_markets:
                    watchlist = _merge_kalshi_with_local(
                        kalshi_markets, watchlist,
                    )

                # Cross-bot rollup (Summary cards, Home active bets,
                # cross-bot history, model cards) — cached, see
                # _cross_bot_rollup above.
                (global_summary, global_active_bets,
                 global_history, bot_models) = _cross_bot_rollup(
                    self.bots, period_days=period_days,
                    period_key=period_key, mode=self.mode)

                # Bot-scoped closed positions — used in Section 5 underneath
                # the active-bet table per request. Tennis-shape bots
                # have no sim.db; pull their closed paper-bet rollup
                # from the tennis adapter instead.
                if bot.get("dashboard_type") == "sport":
                    from . import tennis as _tennis
                    bot_closed_positions = _tennis.closed_positions_for_rollup(
                        bot.get("sim_state_path"), limit=100,
                        real_only=(self.mode == "live"),
                    )
                else:
                    bot_closed_positions = fetch_bet_history(db_path, limit=100)

                # Open positions were fetched above so their tickers
                # could be merged into the Kalshi watchlist scope.

                # Resolve per-bot thresholds. When the bot has written
                # ``data/effective_config.json`` at startup we render
                # the gates it actually applies (which can differ from
                # the dashboard YAML's display defaults per-bot); when
                # absent we fall through to the dashboard YAML and the
                # modal surfaces "showing dashboard defaults" so the
                # user knows the panel might not match reality.
                (bot_edge_cfg, bot_validator_cfg, bot_risk_caps,
                 bot_hedge_cfg, bot_extra_cfg,
                 threshold_source) = resolve_bot_thresholds(
                    bot,
                    fallback_edge=self.edge_cfg,
                    fallback_validators=self.validator_cfg,
                    fallback_risk=self.risk_caps,
                    fallback_hedge=self.hedge_cfg,
                )

                body = render_page(
                    model=model,
                    global_summary=global_summary,
                    global_active_bets=global_active_bets,
                    global_history=global_history,
                    latest_active=latest_active,
                    bot_active_bets=bot_active_bets,
                    bot_closed_positions=bot_closed_positions,
                    watchlist=watchlist,
                    underlying_history=underlying_history,
                    display=bot.get("display") or {},
                    kalshi_history=kalshi_history,
                    prob_history=prob_history,
                    atm_market=atm_market,
                    contract_open_ts=contract_open_ts,
                    contract_close_ts=contract_close_ts,
                    event_title=event_title,
                    risk_caps=bot_risk_caps,
                    edge_cfg=bot_edge_cfg,
                    validator_cfg=bot_validator_cfg,
                    hedge_cfg=bot_hedge_cfg,
                    extra_cfg=bot_extra_cfg,
                    threshold_source=threshold_source,
                    available_bots=self.bots,
                    current_bot=bot["key"],
                    period_key=period_key,
                    tab_key=tab_key,
                    bot_models=bot_models,
                    model_view=model_view,
                    mode=self.mode,
                    live_state_paths=self.live_state_paths,
                    query_string=parsed.query,
                )
            except Exception:  # noqa: BLE001
                log.exception("dashboard render failed")
                body = "<h1>500</h1><p>Dashboard error — check the journal.</p>"
                self.send_response(500)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                payload = body.encode("utf-8")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            payload = body.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        elif parsed.path == "/api/snapshot":
            # JSON payload that the page's JS polls every few seconds
            # to patch live cells in place. Same data the full HTML
            # render uses; the wire format is a flat dict keyed by
            # cell id so the JS can do straightforward DOM lookups.
            try:
                bot = self._resolve_bot(parsed.query)
                qs_snap = parse_qs(parsed.query)
                snap_period = qs_snap.get("period", ["all"])[0]
                if snap_period not in {k for k, _, _ in PERIOD_OPTIONS}:
                    snap_period = "all"
                snap_period_days = _period_days(snap_period)
                if bot.get("dashboard_type") == "sport":
                    # Tennis bots now render through the standard
                    # ``render_page`` — feed the JS poller a real
                    # snapshot built from the JSON watchlist + sim_state
                    # so live cells (Kalshi % / EV / verdict / etc.)
                    # patch in place the same way they do for sim.db
                    # bots.
                    from . import tennis as _tennis
                    payload_wl_snap = _tennis.load_watchlist(
                        bot.get("watchlist_json_path"))
                    snap_rows = _tennis.build_standard_watchlist_rows(
                        payload_wl_snap)
                    snap_actives = _tennis.active_bets_for_rollup(
                        bot.get("sim_state_path"),
                        watchlist_path=bot.get("watchlist_json_path"),
                    )
                    payload_dict = _tennis_like_snapshot(
                        snap_rows, snap_actives, self.bots,
                        edge_cfg=self.edge_cfg,
                        period_days=snap_period_days,
                        mode=self.mode,
                    )
                elif bot.get("dashboard_type") == "survivor":
                    # Survivor page also uses page reloads; the live
                    # monitor rewrites watchlist.json every few minutes.
                    payload_dict = {"bot": bot["key"], "type": "survivor"}
                elif bot.get("dashboard_type") == "billboard":
                    # Billboard's per-bot watchlist renders fully
                    # server-side, but the shared Home tab still
                    # polls /api/snapshot for live cross-bot summary
                    # cards. Returning {bot, type} alone makes the JS
                    # poller patch every Home card to 0 (because
                    # ``snap.summary || {}`` evaluates to {} on this
                    # payload), so the Billboard Home page diverges
                    # from every other bot's Home page after 5s. Feed
                    # the same cross-bot summary the tennis snapshot
                    # builds — watchlist/active_bets stay empty since
                    # there's nothing per-bot to live-patch.
                    payload_dict = _tennis_like_snapshot(
                        [], [], self.bots,
                        edge_cfg=self.edge_cfg,
                        period_days=snap_period_days,
                        mode=self.mode,
                    )
                    payload_dict["bot"] = bot["key"]
                    payload_dict["type"] = "billboard"
                else:
                    db_path = bot["db_path"]
                    payload_dict = build_snapshot(db_path, self.bots,
                                                   self.edge_cfg,
                                                   period_days=snap_period_days,
                                                   mode=self.mode)
            except Exception:  # noqa: BLE001
                log.exception("snapshot endpoint failed")
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"error":"render failed"}')
                return
            payload = json.dumps(payload_dict, default=str).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        elif parsed.path == "/api/diagnosis/latest":
            # Daily droplet diagnosis report. The scheduled
            # daily-droplet-diagnosis agent (or the fallback cron in
            # scripts/diagnosis_collector.py) writes to one of two
            # directories depending on which dashboard wrote it:
            #
            #   sim  → diagnosis/latest.json
            #   live → diagnosis-live/latest.json
            #
            # That separation keeps the sim collector from claiming the
            # live dashboard is "failing" because it's only auditing the
            # sim service (and vice versa). Both are relative to the
            # dashboard's working directory (/root/trading-dashboard on
            # the droplet). When no report exists yet (fresh install or
            # schedule hasn't fired), return a sentinel payload the UI
            # knows how to render as a "no diagnosis yet" empty state.
            diag_subdir = ("diagnosis-live"
                            if self.mode == "live"
                            else "diagnosis")
            diag_path = Path(diag_subdir) / "latest.json"
            try:
                if diag_path.exists():
                    payload = diag_path.read_bytes()
                else:
                    payload = json.dumps(
                        {"status": "no_report_yet"}
                    ).encode("utf-8")
            except OSError as e:
                log.exception("failed to read diagnosis/latest.json")
                payload = json.dumps({
                    "status": "failed",
                    "error": f"could not read latest.json: {e}",
                }).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        elif self.path == "/healthz":
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"ok\n")
        else:
            self.send_error(404)

    def do_POST(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler API)
        from urllib.parse import urlparse, parse_qs
        parsed = urlparse(self.path)
        if parsed.path == "/api/bot/toggle":
            # Flip the bot's enabled flag and return the new state.
            # The page's toggle JS reads the response to update the
            # card without a reload. Unknown bot keys still write a
            # state entry (idempotent) so future bot deploys can opt
            # in to honouring the toggle without a server restart.
            from . import bot_state
            qs = parse_qs(parsed.query)
            bot_key = (qs.get("bot", [""])[0] or "").strip()
            if not bot_key:
                self.send_error(400, "missing ?bot=")
                return
            try:
                entry = bot_state.toggle_bot(bot_key)
            except Exception:  # noqa: BLE001
                log.exception("bot toggle failed")
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"error":"toggle failed"}')
                return
            payload = json.dumps({
                "bot": bot_key,
                "enabled": entry["enabled"],
                "updated_at": entry["updated_at"],
            }).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        self.send_error(404)


def serve(host: str, port: int, bots: List[dict], risk_caps: dict,
          edge_cfg: dict, validator_cfg: dict, hedge_cfg: dict,
          tennis_trader_cfg: dict | None = None,
          unemployment_trader_cfg: dict | None = None,
          cpi_trader_cfg: dict | None = None,
          nba_trader_cfg: dict | None = None,
          wnba_trader_cfg: dict | None = None,
          gas_trader_cfg: dict | None = None,
          survivor_trader_cfg: dict | None = None,
          table_tennis_trader_cfg: dict | None = None,
          darts_trader_cfg: dict | None = None,
          natgas_trader_cfg: dict | None = None,
          billboard_trader_cfg: dict | None = None,
          world_cup_trader_cfg: dict | None = None,
          mlb_trader_cfg: dict | None = None,
          mode: str = "sim",
          live_state_paths: List[str] | None = None) -> None:
    Handler.bots = bots
    Handler.risk_caps = risk_caps
    Handler.edge_cfg = edge_cfg
    Handler.validator_cfg = validator_cfg
    Handler.hedge_cfg = hedge_cfg
    Handler.mode = mode
    # One-time read-path index on each bot's market_views table.
    # fetch_watchlist's latest-row-per-ticker CTE otherwise full-scans
    # the table on every call — CPI's is 390k rows / 400MB, ~250ms per
    # request before this. IF NOT EXISTS makes restarts a no-op; a
    # 5s busy timeout yields to a bot mid-write instead of failing
    # the startup.
    import sqlite3
    from contextlib import closing
    from .data import _conn
    for _b in bots:
        _db = _b.get("db_path") or ""
        if not _db or not Path(_db).exists():
            continue
        try:
            with closing(_conn(_db)) as _c:
                _c.execute("PRAGMA busy_timeout=5000")
                _c.execute(
                    "CREATE INDEX IF NOT EXISTS idx_mv_ticker_id "
                    "ON market_views(ticker, id)")
                _c.commit()
        except (sqlite3.OperationalError, sqlite3.DatabaseError) as e:
            log.info("market_views index skipped for %s (%s)",
                     _b.get("key"), e)
    Handler.live_state_paths = list(live_state_paths or [])
    # Auto-hedge daemon. Reads each sim.db bot's positions table on a
    # 30s interval and closes any position whose unrealized P&L per
    # contract has crossed the configured profit-lock or stop-loss
    # thresholds. No-op when hedge.enabled is false in config.
    from . import hedge_monitor
    hedge_monitor.start_daemon(bots, hedge_cfg)
    # Auto-pause daemon. Six-hourly walk of the bot list — bots with
    # three consecutive 30-day windows of negative realized P&L get
    # their on/off toggle flipped to OFF, with the action recorded
    # in data/regime_notifications.jsonl for the Home-tab panel.
    from . import regime_monitor
    regime_monitor.start_daemon(bots)
    # (Tennis odds snapshotter removed 2026-07-08 alongside the whole
    # in-game adjustment layer; the pre-match model is now the only
    # forecast we run so there's no velocity / volatility / divergence
    # consumer for the JSONL that daemon used to write.)
    # In-process tennis trader. Replaces the standalone
    # baseline-break-monitor.service that previously ran the 60s
    # poll loop. The thread imports tennis-forecast's pure functions
    # (predict, signals, simulator) via sys.path injection — see
    # bots/tennis.py docstring. No-op when tennis_trader.enabled
    # is false (the default until the operator opts in).
    if tennis_trader_cfg:
        from .bots import tennis as tennis_bot
        tennis_bot.start_daemon(tennis_trader_cfg)
    # Unemployment-claims trader. Shape A — upstream's Bot class owns
    # the run() loop; we just gate tick() on the Home-tab toggle.
    # See bots/unemployment_claims.py for the gating rationale.
    if unemployment_trader_cfg:
        from .bots import unemployment_claims as unemployment_bot
        unemployment_bot.start_daemon(unemployment_trader_cfg)
    # CPI trader. Shape A — same structure as unemployment-claims.
    if cpi_trader_cfg:
        from .bots import cpi as cpi_bot
        cpi_bot.start_daemon(cpi_trader_cfg)
    # NBA trader. Shape A — game-outcome forecast.
    if nba_trader_cfg:
        from .bots import nba as nba_bot
        nba_bot.start_daemon(nba_trader_cfg)
    # WNBA trader. Shape A — game-outcome forecast (ESPN data, no key).
    if wnba_trader_cfg:
        from .bots import wnba as wnba_bot
        wnba_bot.start_daemon(wnba_trader_cfg)
    # Gas-prices trader. Shape A with an internal light-tick
    # subthread spawned by Bot.run() (see bots/gas_prices.py).
    if gas_trader_cfg:
        from .bots import gas_prices as gas_bot
        gas_bot.start_daemon(gas_trader_cfg)
    # Live-monitor bots (Shape B — hand-rolled tick loops mirroring
    # tennis). Each one fetches Kalshi markets, runs the model, and
    # ticks its paper-trading simulator on a configurable interval.
    if survivor_trader_cfg:
        from .bots import survivor as survivor_bot
        survivor_bot.start_daemon(survivor_trader_cfg)
    if table_tennis_trader_cfg:
        from .bots import table_tennis as table_tennis_bot
        table_tennis_bot.start_daemon(table_tennis_trader_cfg)
    if darts_trader_cfg:
        from .bots import darts as darts_bot
        darts_bot.start_daemon(darts_trader_cfg)
    # World Cup trader — sim-only for now (no live executor). Trades
    # the three per-match outcome markets (team A / team B / TIE) with
    # the same binary buy/sell logic as the tennis-shape bots.
    if world_cup_trader_cfg:
        from .bots import world_cup as world_cup_bot
        world_cup_bot.start_daemon(world_cup_trader_cfg)
    # MLB trader — Shape B like world-cup. Paper sim in the SIM
    # process; MLBLiveExecutor in the LIVE process (its config carries
    # the ``live:`` block). Probability source is the devigged
    # Pinnacle/Betfair benchmark, so there's no model artifact.
    if mlb_trader_cfg:
        from .bots import mlb as mlb_bot
        mlb_bot.start_daemon(mlb_trader_cfg)
    # Natural-gas trader. The only Tier-3 bot — cron-scheduled
    # (3x/day UTC), so the in-process daemon is a scheduler that
    # sleeps between firings rather than a tight poll loop.
    if natgas_trader_cfg:
        from .bots import natural_gas as natgas_bot
        natgas_bot.start_daemon(natgas_trader_cfg)
    # Billboard trader — biggest model in the fleet (81MB joblib).
    # Lazy-loaded inside predict_top10_proba via @lru_cache, so this
    # daemon's startup memory footprint is small; the model spike
    # only lands on the first tick that finds open Billboard markets.
    if billboard_trader_cfg:
        from .bots import billboard as billboard_bot
        billboard_bot.start_daemon(billboard_trader_cfg)
    server = ThreadingHTTPServer((host, port), Handler)
    log.info("dashboard listening on http://%s:%d", host, port)
    log.info("registered bots: %s",
             ", ".join(f"{b['key']}{'' if b.get('available', True) else ' (no data)'}"
                       for b in bots))
    server.serve_forever()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Kalshi multi-bot trading dashboard")
    parser.add_argument("--host", default=None,
                        help="override host from config")
    parser.add_argument("--port", type=int, default=None,
                        help="override port from config")
    parser.add_argument("--config", default="config/dashboard.yaml")
    args = parser.parse_args(argv)

    from trading_dashboard.config import load_config  # noqa: E402
    from trading_dashboard.logging_setup import setup_logging  # noqa: E402

    cfg = load_config(args.config)
    setup_logging(None)

    risk_caps = {
        "max_open": cfg.risk.max_open_positions,
        "max_open_positions": cfg.risk.max_open_positions,
        "max_exposure": cfg.risk.max_total_exposure_cents,
        "max_total_exposure_cents": cfg.risk.max_total_exposure_cents,
        "max_bets_per_day": cfg.risk.max_bets_per_day,
        "bet_size_cents": cfg.risk.bet_size_cents,
        "cooldown_seconds": cfg.risk.cooldown_seconds_same_market,
        "cooldown_seconds_same_market": cfg.risk.cooldown_seconds_same_market,
    }
    edge_cfg = {
        "min_model_confidence": cfg.edge.min_model_confidence,
        "min_ev_per_contract": cfg.edge.min_ev_per_contract,
        "min_prob_edge_over_breakeven": cfg.edge.min_prob_edge_over_breakeven,
        "min_raw_model_edge": cfg.edge.min_raw_model_edge,
        "max_entry_price_cents": cfg.edge.max_entry_price_cents,
        "min_model_accuracy": cfg.edge.min_model_accuracy,
    }
    validator_cfg = {
        "max_spread_cents": cfg.validators.max_spread_cents,
        "min_book_depth_contracts": cfg.validators.min_book_depth_contracts,
        "min_minutes_to_close": cfg.validators.min_minutes_to_close,
        "max_minutes_to_close": cfg.validators.max_minutes_to_close,
        "prob_bounds_cents": cfg.validators.prob_bounds_cents,
        "min_volume": cfg.validators.min_volume,
        "min_open_interest": cfg.validators.min_open_interest,
        "min_depth_at_best_ask": cfg.validators.min_depth_at_best_ask,
        "basis_risk_strike_window_dollars":
            cfg.validators.basis_risk_strike_window_dollars,
        "basis_risk_max_hours_to_close":
            cfg.validators.basis_risk_max_hours_to_close,
    }
    hedge_cfg = {
        "enabled": cfg.hedge.enabled,
        "profit_lock_cents": cfg.hedge.profit_lock_cents,
        "stop_loss_cents": cfg.hedge.stop_loss_cents,
        "hedge_size_fraction": cfg.hedge.hedge_size_fraction,
    }

    # Live-mode bootstrap must happen BEFORE the bot registry's
    # availability check runs (a few lines down) — otherwise every
    # live macro bot gets ``available=False`` baked in at startup
    # because the placeholder live.db files aren't created until
    # later, and the dashboard's bot dropdown shows every bot as
    # "(no data)" forever (until the next restart re-evaluates).
    # Two bootstraps:
    #   1) bot_state.bootstrap_disabled() writes the per-bot toggle
    #      file with every bot OFF (the homepage toggle UI defaults
    #      to "enabled" for unlisted bots — wrong default for real
    #      money).
    #   2) live_bootstrap.bootstrap_live_data_files() pre-creates
    #      each live data file (empty SQLite mirroring the sim
    #      schema, empty-shell JSONs) so the standard renderer
    #      doesn't early-exit on missing db_path. The model card
    #      then populates from shared training artifacts; only the
    #      live-runtime sections show empty.
    if cfg.mode == "live":
        from . import bot_state
        bot_state.bootstrap_disabled([b.key for b in cfg.bots])
        from . import live_bootstrap
        # Convert the BotEntry dataclasses to the dict shape the
        # bootstrap walks (it reads db_path / watchlist_json_path /
        # sim_state_path — same keys the registry dict uses below).
        live_bootstrap.bootstrap_live_data_files([
            {"db_path": b.db_path,
             "watchlist_json_path": b.watchlist_json_path,
             "sim_state_path": b.sim_state_path}
            for b in cfg.bots
        ])

    # Bot registry comes from the dashboard YAML. Each entry's "available"
    # flag reflects whether the bot's sim.db exists on disk — selecting an
    # unavailable bot in the dropdown shows a friendly stub.
    bots: list[dict] = []
    for b in cfg.bots:
        if b.dashboard_type == "sport":
            # Sport bots are "available" if their watchlist JSON OR
            # their sim.db exists. Sport-shape bots (tennis,
            # table-tennis, darts) write watchlist.json directly on
            # every refresh; the NBA bot writes it via the
            # the upstream exporter after each tick, and on a fresh
            # boot the adapter hasn't run yet — so we accept db_path
            # as evidence the bot is wired up. The page renders an
            # empty-state placeholder until the adapter's first sync.
            available = bool(
                (b.watchlist_json_path
                 and Path(b.watchlist_json_path).exists())
                or (b.db_path and Path(b.db_path).exists())
            )
        elif b.dashboard_type == "survivor":
            # Available whenever the trained model artifact (metrics
            # file) exists. The bot card on the homepage and the bot
            # dropdown should stay visible whether or not there are
            # active "Will X be eliminated" markets — the watchlist
            # page itself surfaces the "no active elimination
            # contracts" empty state inside the standard chrome.
            available = bool(b.metrics_path
                             and Path(b.metrics_path).exists())
        elif b.dashboard_type == "billboard":
            # Same idiom as survivor — available whenever the trained
            # metrics file is on disk, even if no Billboard markets
            # are currently open. The watchlist page surfaces an
            # empty-state placeholder when rows is empty.
            available = bool(b.metrics_path
                             and Path(b.metrics_path).exists())
        else:
            available = Path(b.db_path).exists()
        bots.append({
            "key": b.key,
            "name": b.name,
            "db_path": b.db_path,
            "decisions_path": b.decisions_path,
            "dashboard_type": b.dashboard_type,
            "watchlist_json_path": b.watchlist_json_path,
            "metrics_path": b.metrics_path,
            "coefficients_path": b.coefficients_path,
            "sim_state_path": b.sim_state_path,
            "model_report_path": b.model_report_path,
            "training_data_path": b.training_data_path,
            "training_db_path": b.training_db_path,
            "series_ticker": b.series_ticker,
            "series_prefixes": list(b.series_prefixes or []),
            "seasons": [
                {"name": s.name, "start": s.start, "end": s.end}
                for s in (b.seasons or [])
            ],
            "display": {
                "underlying_label": b.display.underlying_label,
                "underlying_unit": b.display.underlying_unit,
                "underlying_decimals": b.display.underlying_decimals,
                "unit_position": b.display.unit_position,
                "divisor": b.display.divisor,
                "chart_period_minutes": b.display.chart_period_minutes,
                "prediction_period_label": b.display.prediction_period_label,
                "watchlist_title_use_event": b.display.watchlist_title_use_event,
                "question_format": b.display.question_format,
            },
            "available": available,
        })

    # Per-bot trader configs. Each section under the YAML's
    # ``<bot>_trader:`` key gets passed to the matching daemon in
    # serve(). Missing section → empty dict → daemon is a no-op.
    tennis_trader_cfg = cfg.raw.get("tennis_trader") or {}
    unemployment_trader_cfg = cfg.raw.get("unemployment_trader") or {}
    cpi_trader_cfg = cfg.raw.get("cpi_trader") or {}
    nba_trader_cfg = cfg.raw.get("nba_trader") or {}
    wnba_trader_cfg = cfg.raw.get("wnba_trader") or {}
    gas_trader_cfg = cfg.raw.get("gas_trader") or {}
    survivor_trader_cfg = cfg.raw.get("survivor_trader") or {}
    table_tennis_trader_cfg = cfg.raw.get("table_tennis_trader") or {}
    darts_trader_cfg = cfg.raw.get("darts_trader") or {}
    natgas_trader_cfg = cfg.raw.get("natgas_trader") or {}
    billboard_trader_cfg = cfg.raw.get("billboard_trader") or {}
    world_cup_trader_cfg = cfg.raw.get("world_cup_trader") or {}
    mlb_trader_cfg = cfg.raw.get("mlb_trader") or {}

    host = args.host or cfg.host
    port = args.port or cfg.port
    log.info("starting dashboard in %s mode on http://%s:%d",
             cfg.mode, host, port)

    # Walk every <bot>_trader.live.sim_state_path so the live home
    # tab knows where to read each executor's open positions from.
    # Only relevant in live mode (sim has no live executor / no
    # corresponding state file). Empty list on sim → home table
    # is skipped entirely.
    live_state_paths: list[str] = []
    if cfg.mode == "live":
        for trader_cfg in (
            tennis_trader_cfg, unemployment_trader_cfg, cpi_trader_cfg,
            nba_trader_cfg, wnba_trader_cfg, gas_trader_cfg,
            survivor_trader_cfg, table_tennis_trader_cfg, darts_trader_cfg,
            natgas_trader_cfg, billboard_trader_cfg, world_cup_trader_cfg,
            mlb_trader_cfg,
        ):
            live_block = (trader_cfg or {}).get("live") or {}
            path = live_block.get("sim_state_path")
            if path:
                live_state_paths.append(path)

    serve(host, port, bots, risk_caps, edge_cfg, validator_cfg, hedge_cfg,
          tennis_trader_cfg=tennis_trader_cfg,
          unemployment_trader_cfg=unemployment_trader_cfg,
          cpi_trader_cfg=cpi_trader_cfg,
          nba_trader_cfg=nba_trader_cfg,
          wnba_trader_cfg=wnba_trader_cfg,
          gas_trader_cfg=gas_trader_cfg,
          survivor_trader_cfg=survivor_trader_cfg,
          table_tennis_trader_cfg=table_tennis_trader_cfg,
          darts_trader_cfg=darts_trader_cfg,
          natgas_trader_cfg=natgas_trader_cfg,
          billboard_trader_cfg=billboard_trader_cfg,
          world_cup_trader_cfg=world_cup_trader_cfg,
          mlb_trader_cfg=mlb_trader_cfg,
          mode=cfg.mode,
          live_state_paths=live_state_paths)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
