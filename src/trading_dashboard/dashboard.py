"""Web dashboard for the gas-bot simulator.

A standalone process (separate systemd unit) that serves a single HTML
page on port 8080. The page reads from `data/sim.db` and `data/decisions.jsonl`
and shows the bot's state as if it were a real trading account.

What you see:
    - Current model snapshot (price, prediction, prob_up, quantile band)
    - Today's stats: bets count, exposure, realized P&L
    - Open positions: ticker, side, entry, current mark, unrealized P&L
    - Closed positions: entry, exit, realized P&L, win/loss
    - Recent decisions: every market the bot scored, including no-bets
      (so you can see what was considered AND why it was rejected)

Stdlib only — no Flask, no FastAPI. Auto-refreshes every 30 seconds.
Memory footprint is ~25 MB.
"""
from __future__ import annotations

import argparse
import html
import json
import logging
import math
import sqlite3
import sys
import time
from contextlib import closing
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, List, Tuple

log = logging.getLogger("dashboard")


# --------------------------------------------------------------------------- #
# Data access
# --------------------------------------------------------------------------- #

def _conn(db_path: str) -> sqlite3.Connection:
    c = sqlite3.connect(db_path)
    c.row_factory = sqlite3.Row
    return c


def fetch_latest_model(db_path: str) -> dict | None:
    if not Path(db_path).exists():
        return None
    try:
        with closing(_conn(db_path)) as c:
            row = c.execute(
                "SELECT * FROM model_snapshots ORDER BY id DESC LIMIT 1"
            ).fetchone()
            out = dict(row) if row else None
    except (sqlite3.OperationalError, sqlite3.DatabaseError):
        return None
    if out is None:
        return None
    # Augment with closed-bet feedback stats — separate try so a missing
    # training_pairs table on older bot DBs doesn't blank the model card.
    try:
        with closing(_conn(db_path)) as c:
            pairs = c.execute(
                "SELECT COUNT(*) AS n, "
                "       COALESCE(SUM(MIN(MAX(horizon_weeks, 0), 1)), 0) AS w "
                "FROM training_pairs"
            ).fetchone()
            if pairs:
                out["training_pairs_count"] = int(pairs["n"] or 0)
                out["training_pairs_total_weight"] = float(pairs["w"] or 0.0)
    except (sqlite3.OperationalError, sqlite3.DatabaseError):
        pass
    # Realized-bet win rate. Wins / (wins + losses) over all closed
    # positions for this bot. Distinct from the training-holdout
    # directional accuracy — this is the model's actual performance on
    # live Kalshi outcomes, the ultimate ground truth.
    try:
        with closing(_conn(db_path)) as c:
            wl = c.execute(
                "SELECT "
                "  SUM(CASE WHEN realized_pnl_cents > 0 THEN 1 ELSE 0 END) wins, "
                "  SUM(CASE WHEN realized_pnl_cents < 0 THEN 1 ELSE 0 END) losses "
                "FROM positions WHERE status = 'closed'"
            ).fetchone()
            if wl:
                out["actual_wins"] = int(wl["wins"] or 0)
                out["actual_losses"] = int(wl["losses"] or 0)
    except (sqlite3.OperationalError, sqlite3.DatabaseError):
        pass
    return out


def _safe_query(db_path: str, query: str, params: tuple = ()) -> List[dict]:
    """Single helper that all fetch_* functions use to query a bot's
    SQLite DB. Returns [] if the file is missing OR the table doesn't
    exist on this bot's schema. Bot DBs aren't required to have every
    table the gas-prices schema defines (e.g. natural-gas doesn't have
    `position_marks` since it's a daily-cadence bot, not continuous).

    Also tolerates files that aren't SQLite at all (DatabaseError) — this
    happens when a whale-type bot is in the registry; its db_path points
    at a JSONL, not a sim.db.
    """
    if not Path(db_path).exists():
        return []
    try:
        with closing(_conn(db_path)) as c:
            return [dict(r) for r in c.execute(query, params).fetchall()]
    except (sqlite3.OperationalError, sqlite3.DatabaseError):
        return []


def fetch_summary(db_path: str, period_days: int | None = None) -> dict:
    """Lifetime + recent stats used by the Summary section.

    ``period_days`` filters the period-scoped fields (period_bets_made,
    period_net_pnl_cents, period_wins, period_losses) to bets that
    opened (for bets_made) or closed (for P&L/wins/losses) within the
    last N days. None → lifetime.
    """
    empty = {
        "total_bets": 0, "open_count": 0, "exposure_cents": 0,
        "closed_count": 0, "realized_pnl_cents": 0,
        "wins_lifetime": 0, "losses_lifetime": 0,
        "avg_win_cents": 0, "avg_loss_cents": 0,
        "bets_today": 0, "this_week_pnl_cents": 0,
        "biggest_win_cents": 0, "biggest_loss_cents": 0,
        "period_bets_made": 0, "period_net_pnl_cents": 0,
        "period_wins": 0, "period_losses": 0,
        "period_money_spent_cents": 0,
        "period_money_gained_cents": 0,
        "potential_gain_cents": 0,
    }
    if not Path(db_path).exists():
        return empty
    try:
        with closing(_conn(db_path)) as c:
            total = c.execute(
                "SELECT COUNT(*) n FROM positions"
            ).fetchone()
            open_row = c.execute(
                "SELECT COUNT(*) n, COALESCE(SUM(entry_price_cents * contracts), 0) exp "
                "FROM positions WHERE status = 'open'"
            ).fetchone()
            closed_row = c.execute(
                "SELECT COUNT(*) n, COALESCE(SUM(realized_pnl_cents), 0) pnl, "
                "SUM(CASE WHEN realized_pnl_cents > 0 THEN 1 ELSE 0 END) wins, "
                "SUM(CASE WHEN realized_pnl_cents < 0 THEN 1 ELSE 0 END) losses "
                "FROM positions WHERE status = 'closed'"
            ).fetchone()
            avg_win = c.execute(
                "SELECT COALESCE(AVG(realized_pnl_cents), 0) v FROM positions "
                "WHERE status = 'closed' AND realized_pnl_cents > 0"
            ).fetchone()
            avg_loss = c.execute(
                "SELECT COALESCE(AVG(realized_pnl_cents), 0) v FROM positions "
                "WHERE status = 'closed' AND realized_pnl_cents < 0"
            ).fetchone()
            biggest_win = c.execute(
                "SELECT COALESCE(MAX(realized_pnl_cents), 0) v FROM positions WHERE status='closed'"
            ).fetchone()
            biggest_loss = c.execute(
                "SELECT COALESCE(MIN(realized_pnl_cents), 0) v FROM positions WHERE status='closed'"
            ).fetchone()
            this_week_pnl = c.execute(
                "SELECT COALESCE(SUM(realized_pnl_cents), 0) v FROM positions "
                "WHERE status = 'closed' "
                "  AND date(exited_at) >= date('now', '-6 days')"
            ).fetchone()
            this_week_wl = c.execute(
                "SELECT "
                "  SUM(CASE WHEN realized_pnl_cents > 0 THEN 1 ELSE 0 END) wins, "
                "  SUM(CASE WHEN realized_pnl_cents < 0 THEN 1 ELSE 0 END) losses "
                "FROM positions WHERE status = 'closed' "
                "  AND date(exited_at) >= date('now', '-6 days')"
            ).fetchone()
            bets_today = c.execute(
                "SELECT COUNT(*) n FROM trades "
                "WHERE kind = 'entry' AND substr(created_at, 1, 10) = date('now')"
            ).fetchone()
            # Potential gains from currently-open bets (always live —
            # never period-scoped, like the active-bets count). For a
            # YES bet at entry=65c × 10 contracts: potential payout if
            # the contract resolves on our side = (100−65) × 10 = 350c.
            potential = c.execute(
                "SELECT COALESCE(SUM((100 - entry_price_cents) * contracts), 0) v "
                "FROM positions WHERE status = 'open'"
            ).fetchone()
            # Period-filtered values: when period_days is None, scope
            # to lifetime; otherwise restrict to the rolling window.
            if period_days is None:
                period_bets_made = total["n"]
                period_net_pnl = closed_row["pnl"] or 0
                period_wins = closed_row["wins"] or 0
                period_losses = closed_row["losses"] or 0
                spent_row = c.execute(
                    "SELECT COALESCE(SUM(entry_price_cents * contracts), 0) v "
                    "FROM positions"
                ).fetchone()
                gained_row = c.execute(
                    "SELECT COALESCE("
                    "  SUM(entry_price_cents * contracts + realized_pnl_cents), 0"
                    ") v FROM positions WHERE status = 'closed'"
                ).fetchone()
                period_money_spent = spent_row["v"] or 0
                period_money_gained = gained_row["v"] or 0
            else:
                # bets_made = opened in the window (open or closed)
                period_window = f"-{int(period_days)} days"
                pmade = c.execute(
                    "SELECT COUNT(*) n FROM positions "
                    "WHERE date(opened_at) >= date('now', ?)",
                    (period_window,),
                ).fetchone()
                pclosed = c.execute(
                    "SELECT COALESCE(SUM(realized_pnl_cents), 0) pnl, "
                    "  SUM(CASE WHEN realized_pnl_cents > 0 THEN 1 ELSE 0 END) wins, "
                    "  SUM(CASE WHEN realized_pnl_cents < 0 THEN 1 ELSE 0 END) losses "
                    "FROM positions WHERE status = 'closed' "
                    "  AND date(exited_at) >= date('now', ?)",
                    (period_window,),
                ).fetchone()
                period_bets_made = pmade["n"]
                period_net_pnl = pclosed["pnl"] or 0
                period_wins = pclosed["wins"] or 0
                period_losses = pclosed["losses"] or 0
                # Money spent = cost basis of every position opened in
                # the period (open + closed). Bets still live count.
                spent_row = c.execute(
                    "SELECT COALESCE(SUM(entry_price_cents * contracts), 0) v "
                    "FROM positions WHERE date(opened_at) >= date('now', ?)",
                    (period_window,),
                ).fetchone()
                # Money gained = cash returned from positions closed
                # in the period (entry × contracts + realized_pnl ≥ 0).
                gained_row = c.execute(
                    "SELECT COALESCE("
                    "  SUM(entry_price_cents * contracts + realized_pnl_cents), 0"
                    ") v FROM positions WHERE status = 'closed' "
                    "  AND date(exited_at) >= date('now', ?)",
                    (period_window,),
                ).fetchone()
                period_money_spent = spent_row["v"] or 0
                period_money_gained = gained_row["v"] or 0
    except (sqlite3.OperationalError, sqlite3.DatabaseError):
        return empty
    return {
        "total_bets": int(total["n"] or 0),
        "open_count": int(open_row["n"] or 0),
        "exposure_cents": int(open_row["exp"] or 0),
        "closed_count": int(closed_row["n"] or 0),
        "realized_pnl_cents": int(closed_row["pnl"] or 0),
        "wins_lifetime": int(closed_row["wins"] or 0),
        "losses_lifetime": int(closed_row["losses"] or 0),
        "avg_win_cents": int(round(float(avg_win["v"] or 0))),
        "avg_loss_cents": int(round(float(avg_loss["v"] or 0))),
        "biggest_win_cents": int(biggest_win["v"] or 0),
        "biggest_loss_cents": int(biggest_loss["v"] or 0),
        "this_week_pnl_cents": int(this_week_pnl["v"] or 0),
        "this_week_wins": int(this_week_wl["wins"] or 0),
        "this_week_losses": int(this_week_wl["losses"] or 0),
        "bets_today": int(bets_today["n"] or 0),
        "period_bets_made": int(period_bets_made or 0),
        "period_net_pnl_cents": int(period_net_pnl or 0),
        "period_wins": int(period_wins or 0),
        "period_losses": int(period_losses or 0),
        "period_money_spent_cents": int(period_money_spent or 0),
        "period_money_gained_cents": int(period_money_gained or 0),
        "potential_gain_cents": int(potential["v"] or 0),
    }


def fetch_latest_open_position(db_path: str) -> dict | None:
    """The single most-recently opened position (with mark info).

    Tolerates DBs that don't have a positions/position_marks table at
    all — the natural-gas bot writes only model_snapshots + market_views
    since it produces signals, not simulated trades.
    """
    if not Path(db_path).exists():
        return None
    try:
        with closing(_conn(db_path)) as c:
            row = c.execute(
                "SELECT p.*, m.yes_ask_cents AS mark_yes_ask, m.no_ask_cents AS mark_no_ask, "
                "m.yes_bid_cents AS mark_yes_bid, m.mid_cents AS mark_mid, "
                "m.spread_cents AS mark_spread, m.updated_at AS mark_updated_at "
                "FROM positions p LEFT JOIN position_marks m ON p.id = m.position_id "
                "WHERE p.status = 'open' ORDER BY p.opened_at DESC LIMIT 1"
            ).fetchone()
    except (sqlite3.OperationalError, sqlite3.DatabaseError):
        return None
    return dict(row) if row else None


def fetch_active_bets_with_marks(db_path: str) -> List[dict]:
    """Open positions joined with their latest mark + latest mtc + the
    most recent market_views row's strike bounds (so the summary table
    can render the human Question for each row). The local schema
    uses strike_low / strike_high (not floor_strike / cap_strike — the
    Kalshi API names).

    For ``mark_yes_ask`` / ``mark_no_ask`` we COALESCE position_marks
    onto market_views so bots that don't write the position_marks
    table (e.g. natural-gas, which only emits model_snapshots +
    market_views) still show a live "Current" cell on the Home tab's
    active-bets table.
    """
    return _safe_query(
        db_path,
        "SELECT p.*, "
        "       COALESCE(m.yes_ask_cents, "
        "         (SELECT mv.yes_ask_cents FROM market_views mv "
        "            WHERE mv.ticker = p.ticker "
        "            ORDER BY mv.id DESC LIMIT 1)"
        "       ) AS mark_yes_ask, "
        "       COALESCE(m.no_ask_cents, "
        "         (SELECT mv.no_ask_cents FROM market_views mv "
        "            WHERE mv.ticker = p.ticker "
        "            ORDER BY mv.id DESC LIMIT 1)"
        "       ) AS mark_no_ask, "
        "       (SELECT mv.minutes_to_close FROM market_views mv "
        "          WHERE mv.ticker = p.ticker "
        "          ORDER BY mv.id DESC LIMIT 1) AS minutes_to_close, "
        "       (SELECT mv.strike_low FROM market_views mv "
        "          WHERE mv.ticker = p.ticker "
        "          ORDER BY mv.id DESC LIMIT 1) AS floor_strike, "
        "       (SELECT mv.strike_high FROM market_views mv "
        "          WHERE mv.ticker = p.ticker "
        "          ORDER BY mv.id DESC LIMIT 1) AS cap_strike "
        "FROM positions p LEFT JOIN position_marks m ON p.id = m.position_id "
        "WHERE p.status = 'open' ORDER BY p.opened_at DESC")


def fetch_bet_history(db_path: str, limit: int = 100) -> List[dict]:
    """Closed positions only — for the Bet History section.

    Tolerates schema drift across bots. ``gas_price_at_close`` only exists
    on the gas-prices simulator schema; for other bots (e.g. whale-watcher)
    we still want their closed bets to appear in the cross-bot Summary,
    just with an empty Gas-at-close cell. floor_strike + cap_strike are
    pulled via subqueries on market_views so the bet-history view can
    render the human Question text per row.
    """
    if not Path(db_path).exists():
        return []
    base_cols = ("p.id, p.ticker, p.side, p.entry_price_cents, p.exit_price_cents, "
                 "p.contracts, p.realized_pnl_cents, p.opened_at, p.exited_at")
    try:
        with closing(_conn(db_path)) as c:
            # Probe the schema once instead of try/except-ing the full query;
            # cheaper and clearer about *why* the fallback path runs.
            cols = {r["name"] for r in
                    c.execute("PRAGMA table_info(positions)").fetchall()}
            if not cols:
                # No positions table at all (e.g. natural-gas bot writes
                # only model_snapshots + market_views). Empty history.
                return []
            extras = [f"p.{c_}" for c_ in (
                "gas_price_at_close",
                "model_yes_prob_at_entry",
                "kalshi_yes_prob_at_entry",
                "break_even_probability",
                "expected_ev_at_entry",
                "error_type",
            ) if c_ in cols]
            select_cols = base_cols + (", " + ", ".join(extras) if extras else "")
            rows = c.execute(
                f"SELECT {select_cols}, "
                f"(SELECT mv.strike_low FROM market_views mv "
                f"   WHERE mv.ticker = p.ticker "
                f"   ORDER BY mv.id DESC LIMIT 1) AS floor_strike, "
                f"(SELECT mv.strike_high FROM market_views mv "
                f"   WHERE mv.ticker = p.ticker "
                f"   ORDER BY mv.id DESC LIMIT 1) AS cap_strike "
                f"FROM positions p WHERE p.status='closed' "
                f"ORDER BY p.exited_at DESC LIMIT ?", (limit,),
            ).fetchall()
    except (sqlite3.OperationalError, sqlite3.DatabaseError):
        return []
    return [dict(r) for r in rows]


def fetch_global_summary(bots: List[dict],
                          period_days: int | None = None) -> dict:
    """Cross-bot rollup for the Summary section's headline cards.

    ``active_bets`` is always the live across-bots count regardless of
    the period filter (per user spec — current state, not historical).
    The other rolled-up fields (``period_bets_made``,
    ``period_net_pnl_cents``, ``period_win_pct``) honor ``period_days``.
    """
    rollup = {
        "active_bets": 0,            # always current — never period-scoped
        "period_bets_made": 0,
        "period_net_pnl_cents": 0,
        "period_wins": 0,
        "period_losses": 0,
        "period_money_spent_cents": 0,
        "period_money_gained_cents": 0,
        "potential_gain_cents": 0,    # always current
        # Lifetime fields kept for callers that still want them.
        "total_bets": 0,
        "net_pnl_cents": 0,
        "wins": 0,
        "losses": 0,
        "best_bot_name": "—",
        "best_bot_pnl_cents": 0,
        "per_bot": [],
    }
    for b in bots:
        if not b.get("available"):
            continue
        # Whale-type bots don't write the standard positions schema —
        # their realized-P&L story is told on the whale page itself.
        # Skip them in the cross-bot rollup so the summary card row
        # stays focused on the recurrent-series bots.
        if b.get("dashboard_type") and b["dashboard_type"] != "standard":
            continue
        s = fetch_summary(b["db_path"], period_days=period_days)
        rollup["active_bets"] += s.get("open_count", 0)
        rollup["period_bets_made"] += s.get("period_bets_made", 0)
        rollup["period_net_pnl_cents"] += s.get("period_net_pnl_cents", 0)
        rollup["period_wins"] += s.get("period_wins", 0)
        rollup["period_losses"] += s.get("period_losses", 0)
        rollup["period_money_spent_cents"] += s.get("period_money_spent_cents", 0)
        rollup["period_money_gained_cents"] += s.get("period_money_gained_cents", 0)
        rollup["potential_gain_cents"] += s.get("potential_gain_cents", 0)
        rollup["total_bets"] += s.get("total_bets", 0)
        rollup["net_pnl_cents"] += s.get("realized_pnl_cents", 0)
        rollup["wins"] += s.get("wins_lifetime", 0)
        rollup["losses"] += s.get("losses_lifetime", 0)
        rollup["per_bot"].append((b["name"], s))
        better = (
            s.get("realized_pnl_cents", 0) > rollup["best_bot_pnl_cents"]
            or (s.get("realized_pnl_cents", 0) == rollup["best_bot_pnl_cents"]
                and s.get("total_bets", 0) > 0
                and rollup["best_bot_name"] == "—")
        )
        if better:
            rollup["best_bot_name"] = b["name"]
            rollup["best_bot_pnl_cents"] = s.get("realized_pnl_cents", 0)
    period_closed = rollup["period_wins"] + rollup["period_losses"]
    rollup["period_win_pct"] = (
        rollup["period_wins"] / period_closed if period_closed else 0.0
    )
    return rollup


def fetch_watchlist(db_path: str) -> List[dict]:
    """Latest market_view per ticker, filtered to markets with at least
    one of (Kalshi YES ask, Kalshi NO ask) populated AND still open.

    The user wants to see tickers when there is a probability to compare
    against — not when both sides are empty. Loose OR means "show if
    Kalshi has at least one quoted side".

    Markets that have closed are dropped: we never delete old
    market_views rows, so the latest-per-ticker cache contains entries
    for tickers from previous resolution weeks. We project the recorded
    minutes_to_close forward by however long ago the row was captured,
    and require the projection to still be positive.
    """
    if not Path(db_path).exists():
        return []
    try:
        with closing(_conn(db_path)) as c:
            rows = c.execute(
                "WITH latest AS ("
                "  SELECT ticker, MAX(id) AS id FROM market_views GROUP BY ticker"
                ") "
                "SELECT mv.* FROM market_views mv "
                "JOIN latest l ON mv.id = l.id "
                "WHERE mv.model_prob_yes IS NOT NULL "
                "  AND (mv.yes_ask_cents IS NOT NULL OR mv.no_ask_cents IS NOT NULL) "
                # Drop rows whose market has already closed in real time.
                # Project minutes_to_close forward by the elapsed time
                # since this row was recorded; positive => still open.
                # julianday() handles ISO8601 with timezone offsets.
                "  AND mv.minutes_to_close IS NOT NULL "
                "  AND mv.minutes_to_close - "
                "      (julianday('now') - julianday(mv.captured_at)) * 1440 > 0 "
                # Safety net: drop anything captured > 24 hours ago even
                # if its mtc projection is somehow still positive (e.g.
                # bot was offline; data is stale, not actionable).
                "  AND mv.captured_at >= datetime('now', '-1 day') "
                "ORDER BY "
                "  CASE mv.bot_verdict "
                "    WHEN 'BUY_YES' THEN 0 WHEN 'BUY_NO' THEN 0 "
                "    WHEN 'WATCH' THEN 1 ELSE 2 END, "
                "  mv.minutes_to_close ASC"
            ).fetchall()
        return [dict(r) for r in rows]
    except (sqlite3.OperationalError, sqlite3.DatabaseError):
        return []


def _watchlist_from_kalshi(markets: List[dict]) -> List[dict]:
    """Translate Kalshi /markets payloads into the same row shape that
    the local-DB watchlist uses, so the render path is uniform.

    Bot-specific fields (`model_prob_yes`, `bot_verdict`, `edge_yes`,
    `rejection_reason`, …) are left None — those only exist when the
    bot's running and writing market_views. Market-state fields
    (strikes, prices, volume, OI, mtc) come straight from Kalshi.

    Used when the bot service isn't running locally, so the user can
    still see the strike ladder for whichever event in the series is
    currently open. Auto-refreshes via the 60s Kalshi cache.
    """
    out: List[dict] = []
    now = datetime.now(timezone.utc)
    for m in markets:
        ticker = m.get("ticker", "")
        if not ticker:
            continue
        # Strike: floor_strike/cap_strike are dollars-as-floats. Some
        # markets are "between" range; we mirror gas-prices' convention
        # of using strike_low for "above $X".
        floor = m.get("floor_strike")
        cap = m.get("cap_strike")
        try:
            strike_low = float(floor) if floor is not None else None
        except (TypeError, ValueError):
            strike_low = None
        try:
            strike_high = float(cap) if cap is not None else None
        except (TypeError, ValueError):
            strike_high = None
        # Direction inference from strike_type when present.
        direction = "above"
        if strike_low is not None and strike_high is not None:
            direction = "between"
        # Prices come as decimal-dollar strings ("0.7600"). The local
        # schema uses int cents.
        def _cents(v: Any) -> int | None:
            if v is None or v == "":
                return None
            try:
                return int(round(float(v) * 100))
            except (TypeError, ValueError):
                return None
        ya = _cents(m.get("yes_ask_dollars"))
        na = _cents(m.get("no_ask_dollars"))
        yb = _cents(m.get("yes_bid_dollars"))
        spread = None
        if ya is not None and yb is not None:
            spread = max(0, ya - yb)
        # Time-to-close in minutes from close_time.
        close_time_str = m.get("close_time") or ""
        mtc_min: float | None = None
        if close_time_str:
            try:
                ct = datetime.fromisoformat(close_time_str.replace("Z", "+00:00"))
                mtc_min = max(0.0, (ct - now).total_seconds() / 60.0)
            except ValueError:
                mtc_min = None
        # Volume / open-interest come as float-precision strings.
        def _intv(v: Any) -> int | None:
            if v is None or v == "":
                return None
            try:
                return int(round(float(v)))
            except (TypeError, ValueError):
                return None
        out.append({
            "ticker": ticker,
            "title": m.get("title") or "",
            "direction": direction,
            "strike_low": strike_low,
            "strike_high": strike_high,
            "minutes_to_close": mtc_min,
            "model_prob_yes": None,
            "raw_model_prob_yes": None,
            "yes_ask_cents": ya,
            "no_ask_cents": na,
            "yes_bid_cents": yb,
            "spread_cents": spread,
            "edge_yes": None,
            "edge_no": None,
            "bot_verdict": None,
            "rejection_reason": None,
            "rules_primary": m.get("rules_primary") or "",
            "rules_secondary": m.get("rules_secondary") or "",
            "event_title": m.get("event_ticker") or "",
            "event_sub_title": m.get("yes_sub_title") or "",
            "volume": _intv(m.get("volume_fp")),
            "open_interest": _intv(m.get("open_interest_fp")),
            "yes_ask_depth": _intv(m.get("yes_ask_size_fp")),
            "captured_at": now.isoformat(),
        })
    # Sort by strike ascending — matches the local fetch_watchlist order.
    out.sort(key=lambda r: (r.get("strike_low")
                            if r.get("strike_low") is not None else 9_999.0,
                            r.get("ticker") or ""))
    return out


def _merge_kalshi_with_local(kalshi_markets: List[dict],
                              local_rows: List[dict]) -> List[dict]:
    """Build the watchlist from Kalshi (the spine), then for each row
    look up the matching ticker in the local DB and copy over the
    bot-computed fields (model_prob_yes, bot_verdict, edge_*, etc.).

    Why Kalshi-spine: the local sim.db can be empty or stale (different
    event's tickers from a prior week), but Kalshi always knows the
    current event's full strike ladder. Layering local data on top
    means the user sees the right ladder AND the bot's view per row,
    when the bot has one.
    """
    base = _watchlist_from_kalshi(kalshi_markets)
    by_ticker = {r["ticker"]: r for r in local_rows if r.get("ticker")}
    bot_fields = (
        "model_prob_yes", "raw_model_prob_yes",
        "edge_yes", "edge_no",
        "bot_verdict", "rejection_reason",
        "rules_primary", "rules_secondary",
    )
    for row in base:
        local = by_ticker.get(row["ticker"])
        if not local:
            continue
        for f in bot_fields:
            if local.get(f) is not None:
                row[f] = local[f]
    return base



def fetch_underlying_history(db_path: str, hours: int = 72,
                              max_points: int = 400) -> List[dict]:
    """Time-series of the bot's underlying value (model_snapshots.current_gas_price)
    over the last N hours. Used to draw the watchlist hero chart.

    Each row carries both ``captured_at`` (ISO string for compatibility
    with older code paths) and ``ts`` (unix seconds, what
    ``svg_kalshi_chart`` reads). Empty list when the bot's DB doesn't
    exist or has no snapshots in the window.
    """
    if not Path(db_path).exists():
        return []
    try:
        with closing(_conn(db_path)) as c:
            rows = c.execute(
                "SELECT captured_at, current_gas_price AS value "
                "FROM model_snapshots "
                "WHERE current_gas_price IS NOT NULL "
                "  AND captured_at >= datetime('now', ?) "
                "ORDER BY captured_at ASC LIMIT ?",
                (f"-{int(hours)} hours", max_points),
            ).fetchall()
    except (sqlite3.OperationalError, sqlite3.DatabaseError):
        return []
    out: List[dict] = []
    for r in rows:
        d = dict(r)
        ts = _iso_to_unix(d.get("captured_at"))
        if ts is not None:
            d["ts"] = ts
        out.append(d)
    return out


def _iso_to_unix(s: str | None) -> float | None:
    """Parse SQLite's `YYYY-MM-DD HH:MM:SS` (UTC) strings into unix
    seconds. Returns None on malformed input — callers fall back to the
    captured_at string.
    """
    if not s:
        return None
    try:
        if "T" in s:
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        else:
            dt = datetime.strptime(s, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except (TypeError, ValueError):
        return None


def fetch_decisions(decisions_path: str, limit: int = 60) -> List[dict]:
    p = Path(decisions_path)
    if not p.exists():
        return []
    # Tail the file efficiently — read last ~256 KB which is ~ last 1k entries.
    size = p.stat().st_size
    with open(p, "rb") as f:
        if size > 262144:
            f.seek(-262144, 2)
            f.readline()  # discard partial line
        tail = f.read().decode("utf-8", errors="replace")
    rows: List[dict] = []
    for line in tail.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows[-limit:][::-1]  # newest first


def build_snapshot(db_path: str, bots: List[dict],
                    edge_cfg: dict,
                    period_days: int | None = None) -> dict:
    """Compact JSON snapshot for live page updates.

    The browser polls this every few seconds and patches DOM cells in
    place — no full page reload. Format is a flat dict where each
    top-level key corresponds to a stable element id in the rendered
    HTML, so the JS can do `document.getElementById(...)` lookups
    directly without needing to reparse layout.

    Includes:
      • Cross-bot summary card values (Total bets / P&L / win % / etc.)
      • The current bot's watchlist rows (one entry per ticker)
      • Active bet mark, entry, current EV, P&L
      • A monotonically-increasing tick counter for the JS to detect
        skipped updates.
    """
    summary = fetch_global_summary(bots, period_days=period_days)
    watchlist = fetch_watchlist(db_path)
    active_bets = fetch_active_bets_with_marks(db_path)

    def _ev_yes(p, ya_c, spread_c):
        if p is None or ya_c is None:
            return None
        return float(p) - (ya_c / 100.0) - ((spread_c or 0) / 200.0)
    def _ev_no(p, na_c, spread_c):
        if p is None or na_c is None:
            return None
        return (1.0 - float(p)) - (na_c / 100.0) - ((spread_c or 0) / 200.0)

    rows = []
    min_ev = edge_cfg.get("min_ev_per_contract", 0.03)
    for v in watchlist:
        ya = v.get("yes_ask_cents"); na = v.get("no_ask_cents")
        sp = v.get("spread_cents")
        p = v.get("model_prob_yes")
        ey = _ev_yes(p, ya, sp); en = _ev_no(p, na, sp)
        rows.append({
            "ticker": v.get("ticker"),
            "kalshi_yes": ya, "kalshi_no": na,
            "spread": sp,
            "volume": v.get("volume"), "open_interest": v.get("open_interest"),
            "minutes_to_close": v.get("minutes_to_close"),
            "model_prob_yes": p,
            # Raw (un-blended) model prob — null on bots that don't
            # surface it. Used by the dashboard to disambiguate "is
            # the displayed edge real raw disagreement, or shrinkage
            # blend noise?"
            "raw_model_prob_yes": v.get("raw_model_prob_yes"),
            "ev_yes": ey, "ev_no": en,
            "bot_verdict": v.get("bot_verdict"),
            "rejection_reason": v.get("rejection_reason"),
        })

    actives = []
    for ab in active_bets:
        actives.append({
            "id": ab.get("id"),
            "ticker": ab.get("ticker"),
            "side": ab.get("side"),
            "entry": ab.get("entry_price_cents"),
            "contracts": ab.get("contracts"),
            "mark_yes_ask": ab.get("mark_yes_ask"),
            "mark_no_ask": ab.get("mark_no_ask"),
            "mark_mid": ab.get("mark_mid"),
            "unreal_pnl_cents": unrealized_pnl_cents(ab),
        })

    period_closed = (summary.get("period_wins", 0)
                     + summary.get("period_losses", 0))
    return {
        "summary": {
            "active_bets": summary.get("active_bets"),
            "period_closed_bets": period_closed,
            "period_money_spent_cents": summary.get("period_money_spent_cents"),
            "period_money_gained_cents": summary.get("period_money_gained_cents"),
            "potential_gain_cents": summary.get("potential_gain_cents"),
            "period_net_pnl_cents": summary.get("period_net_pnl_cents"),
            "period_win_pct": summary.get("period_win_pct"),
            "period_has_closed": period_closed > 0,
        },
        "watchlist": rows,
        "active_bets": actives,
        "min_ev": min_ev,
        "captured_at": datetime.now(timezone.utc).isoformat(),
    }


# --------------------------------------------------------------------------- #
# Computations
# --------------------------------------------------------------------------- #

def unrealized_pnl_cents(pos: dict) -> int | None:
    """For a YES position, mark = current yes ask (what we could resell at).
    For a NO position, mark = current no ask. Returns P&L in cents.
    """
    entry = int(pos["entry_price_cents"])
    contracts = int(pos["contracts"])
    side = (pos["side"] or "").upper()
    if side == "YES":
        mark = pos.get("mark_yes_ask")
    else:
        mark = pos.get("mark_no_ask")
    if mark is None:
        return None
    return (int(mark) - entry) * contracts


def fmt_cents(c: int | float | None) -> str:
    if c is None:
        return "—"
    return f"${c/100:+.2f}" if c < 0 else f"${c/100:.2f}"


def kalshi_fee_cents(price_cents: int | None,
                       contracts: int | None) -> int:
    """Kalshi trading fee per their published formula:

        fee = ceil(0.07 × contracts × price × (1 − price))

    where ``price`` is in dollars and the fee is also in dollars.
    Equivalent in cents: ``ceil(0.07 × contracts × p × (100 − p) /
    100)`` where p is the integer-cents price.

    Charged on entry AND on exit (per side). At settlement (price
    is 0¢ or 100¢) the fee is zero — no risk left to fee.

    Returns 0 cents when inputs are missing or out-of-range so the
    caller can safely add this to any cost calculation.
    """
    if price_cents is None or contracts is None:
        return 0
    try:
        p = int(price_cents)
        n = int(contracts)
    except (TypeError, ValueError):
        return 0
    if n <= 0 or p <= 0 or p >= 100:
        return 0
    raw = 0.07 * n * p * (100 - p) / 100.0
    return int(math.ceil(raw))


def fmt_signed_cents(c: int | None) -> str:
    if c is None:
        return "—"
    sign = "+" if c >= 0 else "−"
    return f"{sign}${abs(c)/100:.2f}"


def cents_or_dash(c: int | None) -> str:
    return f"{c}c" if c is not None else "—"


def confidence_pct(prob: float | None,
                   model_accuracy: float | None = None) -> int | None:
    """How sure should we be that this question resolves one way?

    Combines two signals:
      1. Distance from 50/50 (the model's own view): closer to 0 or 1 = stronger
      2. Model's historical track record (calibrated_classifier_accuracy):
         a model that's right 58% of the time can't be 100% confident on
         any individual call - the displayed confidence is capped by that
         track record.

    Math:
      raw_view = |prob - 0.5| * 2     (0..1, distance from 50/50)
      confidence = raw_view * model_accuracy

    Examples (model_accuracy = 0.58):
      prob=0.50 -> 0%   (no view)
      prob=0.70 -> 23%  (some view, scaled by reliability)
      prob=0.90 -> 46%
      prob=1.00 -> 58%  (full directional view, capped by reliability)
    """
    if prob is None:
        return None
    raw = abs(prob - 0.5) * 2.0
    accuracy = model_accuracy if model_accuracy is not None else 0.55
    # Floor at 0.5 (no info worse than coinflip should still show > 0).
    accuracy = max(0.5, min(1.0, accuracy))
    return int(round(raw * accuracy * 100))


def _empty_chart_frame(width: int = 760, height: int = 220,
                        contract_open_ts: float | None = None,
                        contract_close_ts: float | None = None) -> str:
    """Empty-state chart: just the frame (gridlines + day ticks if a
    contract span is known), no polyline. Used when fewer than 2
    snapshots have been recorded — the user wants to see the chart's
    silhouette even when there's nothing to plot yet.
    """
    pad_l, pad_r, pad_t, pad_b = 12, 64, 14, 30
    inner_w = width - pad_l - pad_r
    out: List[str] = [
        f"<div class='wl-chart-wrap'>"
        f"<svg width='100%' height='{height}' viewBox='0 0 {width} {height}' "
        f"preserveAspectRatio='none' style='display:block'>"
    ]
    # 5 horizontal gridlines, no labels (no data range to anchor them to).
    for i in range(5):
        y = pad_t + (i / 4.0) * (height - pad_t - pad_b)
        out.append(f"<line x1='{pad_l}' y1='{y}' x2='{width-pad_r}' y2='{y}' "
                   f"stroke='#1f2530' stroke-width='1'/>")
    # Day ticks across [contract_open, now] — chart always ends at
    # the current date, never extends into the future even when the
    # contract isn't closed yet. (contract_close_ts is intentionally
    # ignored here for that reason.)
    if contract_open_ts is not None:
        from datetime import timedelta
        t_min = float(contract_open_ts)
        t_max = max(t_min, time.time())
        dt_min = datetime.fromtimestamp(t_min, tz=timezone.utc)
        dt_max = datetime.fromtimestamp(t_max, tz=timezone.utc)
        day_labels: List[str] = [dt_min.strftime("%b %-d")]
        cur = dt_min.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        while cur < dt_max:
            lbl = cur.strftime("%b %-d")
            if lbl != day_labels[-1]:
                day_labels.append(lbl)
            cur += timedelta(days=1)
        last_label = dt_max.strftime("%b %-d")
        if last_label != day_labels[-1]:
            day_labels.append(last_label)
        n = max(1, len(day_labels) - 1)
        for i, label in enumerate(day_labels):
            frac = i / n if n else 0.5
            x = pad_l + frac * inner_w
            out.append(f"<line x1='{x:.1f}' y1='{pad_t}' x2='{x:.1f}' "
                       f"y2='{height-pad_b}' stroke='#1f2530' stroke-width='1' "
                       f"stroke-dasharray='2,3' opacity='0.7'/>")
            anchor = "start" if i == 0 else (
                "end" if i == len(day_labels) - 1 else "middle")
            out.append(f"<text x='{x:.0f}' y='{height-10}' fill='#8b949e' "
                       f"font-size='10' text-anchor='{anchor}'>"
                       f"{html.escape(label)}</text>")
    out.append("</svg></div>")
    return "".join(out)


def svg_kalshi_chart(history: List[dict], display: dict,
                      reference_strike: float | None = None,
                      strike_side: str | None = None,
                      strike_is_active_bet: bool = False,
                      contract_open_ts: float | None = None,
                      contract_close_ts: float | None = None,
                      total_volume: int | None = None,
                      width: int = 760, height: int = 220) -> str:
    """Underlying-price chart, derived from Kalshi's strike ladder.

    Same visual idiom as Kalshi's market-page chart: one line in the
    underlying's native units (USD/MMBtu, USD/gal, K claims), y-axis
    auto-scaled to the data range, optional horizontal strike line
    colored to indicate winning side. Different from the prior 0..100%
    chance chart — the y-axis here is in real-world units so the user
    sees what Kalshi shows on its own market page.
    """
    pts_in: List[Tuple[float, float]] = []
    for r in history:
        ts = r.get("ts")
        v = r.get("value")
        if ts is None or v is None:
            continue
        try:
            pts_in.append((float(ts), float(v)))
        except (TypeError, ValueError):
            continue
    if len(pts_in) < 2:
        return _empty_chart_frame(width=width, height=height,
                                    contract_open_ts=contract_open_ts,
                                    contract_close_ts=contract_close_ts)

    # Y-axis labels go on the right edge (matches Kalshi's market page),
    # so reserve the right padding instead of the left. Bottom padding
    # leaves room for the date-tick row.
    pad_l, pad_r, pad_t, pad_b = 12, 64, 14, 30
    inner_w = width - pad_l - pad_r
    inner_h = height - pad_t - pad_b
    n = len(pts_in)
    # X-axis spans contract open → NOW. Always ends at current date,
    # never extended into the future even when the contract is still
    # open. Future-dated data points (clock skew etc.) are clipped to
    # now since the user's spec is "end at current date".
    now_ts = time.time()
    t_max = now_ts
    t_min = float(contract_open_ts) if contract_open_ts else pts_in[0][0]
    if t_min > pts_in[0][0]:
        t_min = pts_in[0][0]
    t_span = max(1.0, t_max - t_min)

    # The visible polyline plots the raw recorded values — what the
    # underlying actually was at each Kalshi-recorded tick. No smoothing
    # or bucketing: the line and the hover-tooltip values are the same
    # series, so what you see scrubbing matches what you see on the line.
    pts_plot: List[Tuple[float, float]] = list(pts_in)

    # Auto-scale the y-axis to the actual data range with 8% padding.
    # When there's an active bet, also include its strike in the range
    # so the dotted reference line is always visible on the chart.
    values = [v for _, v in pts_in]
    if strike_is_active_bet and reference_strike is not None:
        values = values + [float(reference_strike)]
    vmin = min(values)
    vmax = max(values)
    if vmax == vmin:
        pad_v = max(0.001, abs(vmax) * 0.005)
    else:
        pad_v = (vmax - vmin) * 0.08
    y_lo = vmin - pad_v
    y_hi = vmax + pad_v

    # With the strike included in the value set, the dotted line is
    # always in range when there's a bet. The flag is kept for
    # completeness (callers can pass an out-of-range strike for the
    # closest-to-money case, which we now skip).
    strike_in_range = (reference_strike is not None
                       and y_lo <= float(reference_strike) <= y_hi)

    def x_at(t: float) -> float:
        return pad_l + (t - t_min) / t_span * inner_w

    def y_at(v: float) -> float:
        return pad_t + (1.0 - (v - y_lo) / (y_hi - y_lo)) * inner_h

    # Wrap the SVG in a positioning context for the hover tooltip and
    # expose the chart geometry as data attrs so the JS can map the
    # cursor's x position back to a timestamp without re-deriving it.
    # Compact JSON of (ts, raw_value) pairs for the hover tooltip's
    # interpolation. Server-side stores the raw value (pre-divisor /
    # pre-format); the JS formats it client-side.
    points_payload = json.dumps([(int(t), v) for t, v in pts_in],
                                  separators=(",", ":"))
    fmt_payload = json.dumps({
        "divisor": float(display.get("divisor", 1.0) or 1.0),
        "decimals": int(display.get("underlying_decimals", 2)),
        "unit": display.get("underlying_unit", ""),
        "unit_position": display.get("unit_position", "prefix"),
    }, separators=(",", ":"))
    out: List[str] = [
        f"<div class='wl-chart-wrap' "
        f"data-tmin='{t_min:.0f}' data-tmax='{t_max:.0f}' "
        f"data-padl='{pad_l}' data-innerw='{inner_w}' "
        f"data-padt='{pad_t}' data-padb='{pad_b}' data-h='{height}' "
        f"data-vbw='{width}' "
        f"data-points='{html.escape(points_payload)}' "
        f"data-fmt='{html.escape(fmt_payload)}'>",
        f"<svg width='100%' height='{height}' viewBox='0 0 {width} {height}' "
        f"preserveAspectRatio='none' style='display:block'>"
    ]

    # 5 evenly-spaced y-gridlines, labeled in the underlying's units.
    for i in range(5):
        v = y_lo + (i / 4.0) * (y_hi - y_lo)
        y = y_at(v)
        out.append(f"<line x1='{pad_l}' y1='{y}' x2='{width-pad_r}' y2='{y}' "
                   f"stroke='#1f2530' stroke-width='1'/>")
        # Y-axis label sits to the right of the chart (Kalshi style).
        out.append(f"<text x='{width-pad_r+6}' y='{y+4}' fill='#8b949e' "
                   f"font-size='10' text-anchor='start'>"
                   f"{html.escape(fmt_underlying(v, display))}</text>")

    # Color the line green where it sits on the winning side of the
    # strike, white on the losing side. Same logic as the prior
    # underlying chart — strike-relative segment splitting.
    side = (strike_side or "").upper()
    if side == "NO":
        above_color, below_color = "#c9d1d9", "#3fb950"
    else:
        above_color, below_color = "#3fb950", "#c9d1d9"

    if not strike_in_range or reference_strike is None:
        path = " ".join(f"{x_at(t):.1f},{y_at(v):.1f}" for t, v in pts_plot)
        out.append(f"<polyline points='{path}' stroke='#c9d1d9' "
                   f"stroke-width='2' fill='none'/>")
    else:
        strike = float(reference_strike)
        runs: List[Tuple[bool, List[Tuple[float, float]]]] = []
        cur_above = pts_plot[0][1] >= strike
        cur_run: List[Tuple[float, float]] = [(x_at(pts_plot[0][0]), y_at(pts_plot[0][1]))]
        for i in range(1, n):
            t_prev, v_prev = pts_plot[i - 1]
            t_curr, v_curr = pts_plot[i]
            new_above = v_curr >= strike
            if new_above == cur_above:
                cur_run.append((x_at(t_curr), y_at(v_curr)))
                continue
            denom = v_curr - v_prev
            t = (strike - v_prev) / denom if denom != 0 else 0.5
            t = max(0.0, min(1.0, t))
            cross_x = x_at(t_prev) + t * (x_at(t_curr) - x_at(t_prev))
            cross_y = y_at(strike)
            cur_run.append((cross_x, cross_y))
            runs.append((cur_above, cur_run))
            cur_run = [(cross_x, cross_y), (x_at(t_curr), y_at(v_curr))]
            cur_above = new_above
        runs.append((cur_above, cur_run))
        for is_above, run in runs:
            if len(run) < 2:
                continue
            color = above_color if is_above else below_color
            pts_str = " ".join(f"{x:.1f},{y:.1f}" for x, y in run)
            out.append(f"<polyline points='{pts_str}' stroke='{color}' "
                       f"stroke-width='2' fill='none'/>")

    # Horizontal strike line — dotted, drawn ONLY for an active bet.
    # YES position → green dotted line, label reads "Above $X"
    # NO  position → red dotted line, label reads "Not above $X"
    # The colour communicates "your winning territory": YES bets win
    # when the underlying ends up above the line (green = win), NO
    # bets win when it stays below (red = the threshold you don't
    # want to be above).
    if strike_is_active_bet and strike_in_range and reference_strike is not None:
        ys = y_at(float(reference_strike))
        is_no = (side == "NO")
        line_color = "#f85149" if is_no else "#3fb950"
        label_strike = fmt_underlying(float(reference_strike), display)
        label = (f"Not above {label_strike}" if is_no
                 else f"Above {label_strike}")
        out.append(f"<line x1='{pad_l}' y1='{ys}' x2='{width-pad_r}' y2='{ys}' "
                   f"stroke='{line_color}' stroke-width='1.5' "
                   f"stroke-dasharray='4,4' opacity='0.95'/>")
        label_x = pad_l + inner_w * 0.5
        out.append(f"<text x='{label_x:.0f}' y='{ys-6}' fill='{line_color}' "
                   f"font-size='11' text-anchor='middle' opacity='0.95'>"
                   f"{html.escape(label)}</text>")

    # X-axis: one label per UNIQUE day in the contract span, EVENLY
    # SPACED across the chart's width regardless of where each midnight
    # actually falls in time. Visually decouples label position from
    # the time axis (the polyline still uses real time) so the bottom
    # row stays balanced even on lopsided spans.
    from datetime import timedelta
    dt_min = datetime.fromtimestamp(t_min, tz=timezone.utc)
    dt_max = datetime.fromtimestamp(t_max, tz=timezone.utc)
    day_labels: List[str] = [dt_min.strftime("%b %-d")]
    cur = dt_min.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
    while cur < dt_max:
        lbl = cur.strftime("%b %-d")
        if lbl != day_labels[-1]:
            day_labels.append(lbl)
        cur += timedelta(days=1)
    last_label = dt_max.strftime("%b %-d")
    if last_label != day_labels[-1]:
        day_labels.append(last_label)
    n = max(1, len(day_labels) - 1)
    for i, label in enumerate(day_labels):
        frac = i / n if n else 0.5
        x = pad_l + frac * inner_w
        out.append(f"<line x1='{x:.1f}' y1='{pad_t}' x2='{x:.1f}' "
                   f"y2='{height-pad_b}' stroke='#1f2530' stroke-width='1' "
                   f"stroke-dasharray='2,3' opacity='0.7'/>")
        if i == 0:
            anchor, tx = "start", x
        elif i == len(day_labels) - 1:
            anchor, tx = "end", x
        else:
            anchor, tx = "middle", x
        out.append(f"<text x='{tx:.0f}' y='{height-10}' fill='#8b949e' "
                   f"font-size='10' text-anchor='{anchor}'>"
                   f"{html.escape(label)}</text>")

    # Volume moved to the hero header (top-right under "Closes in")
    # per user request — no longer on the chart frame.

    out.append("</svg>")
    # Hover tooltip — JS in the page polyfills this with a vertical
    # line + "May 1 at 9 AM" label as the cursor moves over the chart.
    out.append("<div class='wl-chart-tooltip' hidden></div>")
    out.append("</div>")
    return "".join(out)



def fmt_underlying(value: float | None, display: dict) -> str:
    """Format an underlying value per the bot's display config:
       prefix → '$2.759';   suffix → '189K';   none → '2.759'.
    Applies `divisor` first so bots that store raw counts (e.g. 189000
    claims) can render in thousands.
    """
    if value is None:
        return "—"
    divisor = float(display.get("divisor", 1.0)) or 1.0
    v = float(value) / divisor
    decimals = int(display.get("underlying_decimals", 2))
    unit = display.get("underlying_unit", "")
    pos = display.get("unit_position", "prefix")
    n = f"{v:,.{decimals}f}"
    if pos == "prefix":
        return f"{unit}{n}"
    if pos == "suffix":
        return f"{n}{unit}"
    return n


def time_left_str(minutes: float | None) -> str:
    """Compact 'closes in 3d 4h' / '12h 30m' / '45m' for the hero header."""
    if minutes is None or minutes <= 0:
        return "—"
    total_min = int(minutes)
    days = total_min // (60 * 24)
    rem = total_min - days * 60 * 24
    hours = rem // 60
    mins = rem - hours * 60
    if days >= 1:
        return f"{days}d {hours}h"
    if hours >= 1:
        return f"{hours}h {mins}m"
    return f"{mins}m"


def question_str(direction: str, low: float | None, high: float | None,
                 display: dict | None = None) -> str:
    """Format the watchlist's Question column. Uses the bot's display
    config when given so unemployment renders "above 175K" instead of
    "above $175000.00"."""
    if display:
        if direction == "between" and low is not None and high is not None:
            return f"{fmt_underlying(low, display)} – {fmt_underlying(high, display)}"
        if low is not None:
            return f"{direction} {fmt_underlying(low, display)}"
        return direction or "—"
    # Legacy default — gas-prices-style $/gal formatting.
    if direction == "between" and low is not None and high is not None:
        return f"${low:.2f} – ${high:.2f}"
    if low is not None:
        return f"{direction} ${low:.2f}"
    return direction or "—"


def time_to_close_str(minutes: float | None) -> str:
    if minutes is None:
        return "—"
    if minutes > 1440:
        return f"{minutes/1440:.1f}d"
    if minutes > 60:
        return f"{minutes/60:.1f}h"
    return f"{int(minutes)}m"


# --------------------------------------------------------------------------- #
# HTML rendering
# --------------------------------------------------------------------------- #

CSS = """
* { box-sizing: border-box; }
body { background: #0d1117; color: #c9d1d9; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 0; padding: 20px; line-height: 1.4; }
h1, h2 { color: #f0f6fc; margin: 0 0 8px 0; }
h1 { font-size: 22px; font-weight: 600; }
h2 { font-size: 14px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; color: #8b949e; margin-top: 28px; margin-bottom: 8px; }
.meta { color: #8b949e; font-size: 12px; margin-bottom: 20px; }
.row { display: flex; gap: 18px; flex-wrap: wrap; margin-bottom: 14px; }
/* Cards sit on a slightly lighter, slightly cooler shade than the
   section background (#161b22). Subtle border + soft drop-shadow gives
   them a gentle elevated appearance against the section panel. */
.card { background: #1d232c; border: 1px solid #30363d; border-radius: 8px; padding: 14px 18px; flex: 1; min-width: 180px; box-shadow: 0 1px 2px rgba(0,0,0,0.35); }
.card .label { font-size: 11px; text-transform: uppercase; color: #9ca5b3; letter-spacing: 0.05em; }
.card .value { font-size: 22px; font-weight: 600; color: #f0f6fc; margin-top: 4px; }
/* Color modifiers — must be more specific than .card .value (0,2,0) so
   the green/red/gray classes actually paint summary card values. */
.card .value.green, .green { color: #56d364; }
.card .value.red, .red { color: #f85149; }
.card .value.gray, .gray { color: #8b949e; }
.card .value.yellow, .yellow { color: #e3b341; }
table { width: 100%; border-collapse: collapse; background: transparent; font-size: 13px; margin: 4px 0; }
th, td { padding: 8px 10px; text-align: left; border-bottom: 1px solid #21262d; }
tr:last-child td { border-bottom: none; }
th { background: #161b22; color: #8b949e; font-weight: 600; font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em; }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
.badge { display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 11px; font-weight: 600; }
.badge-yes { background: rgba(86, 211, 100, 0.2); color: #56d364; }
.badge-no { background: rgba(248, 81, 73, 0.2); color: #f85149; }
.badge-skip { background: rgba(139, 148, 158, 0.2); color: #8b949e; }
.badge-hedge { background: rgba(227, 179, 65, 0.2); color: #e3b341; margin-left: 4px; }
.empty { color: #8b949e; padding: 14px; text-align: center; font-style: italic; }
/* EV diagnostic banner — loud when the trade has gone NEGATIVE EV. */
.ev-warning { background: rgba(248, 81, 73, 0.12); border: 1px solid #f85149;
   color: #ffa6a1; padding: 10px 14px; border-radius: 6px;
   margin: 12px 0; font-size: 13px; line-height: 1.45; }
.ev-warning strong { color: #f85149; }
/* "Why this trade was taken" — two-column grid of audit rows. */
.why-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 4px 24px;
   margin: 8px 0 12px; max-width: 720px; }
.why-row { display: flex; justify-content: space-between;
   border-bottom: 1px dashed #30363d; padding: 4px 0; font-size: 13px; }
.why-row span:first-child { color: #8b949e; }
.why-row span:last-child { color: #c9d1d9; font-variant-numeric: tabular-nums; }
.why-gates { margin: 6px 0 14px; line-height: 1.6; }
.why-gates .mono { font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
   font-size: 11px; color: #c9d1d9; }
/* Status pill — used on watchlist verdict + diagnostics scorecards. */
.status-pill { display: inline-block; padding: 2px 8px; border-radius: 12px;
   font-size: 11px; font-weight: 600; letter-spacing: 0.04em;
   text-transform: uppercase; }
.status-pill.green { background: rgba(86, 211, 100, 0.2); color: #56d364; }
.status-pill.yellow { background: rgba(227, 179, 65, 0.2); color: #e3b341; }
.status-pill.red { background: rgba(248, 81, 73, 0.2); color: #f85149; }
.status-pill.gray { background: rgba(139, 148, 158, 0.2); color: #8b949e; }
/* Brief highlight on cells whose value just updated via the live JS
   poll. Pulses then fades — keeps changes visible without being loud. */
@keyframes cell-flash-fade {
  0%   { background-color: rgba(88, 166, 255, 0.35); }
  100% { background-color: transparent; }
}
.cell-flash { animation: cell-flash-fade 0.8s ease-out; }
/* Buy-criteria reference card. Compact two-column variant + a wider
   three-column variant (with descriptions) used in Section 2. */
table.criteria { max-width: 560px; font-size: 12px; }
table.criteria.criteria-wide { max-width: 1100px; width: 100%; }
table.criteria td { padding: 6px 10px; border-bottom: 1px solid #1f2530;
    vertical-align: top; }
table.criteria td:first-child { color: #c9d1d9; font-weight: 500;
    white-space: nowrap; }
table.criteria.criteria-wide td:nth-child(2) {
    white-space: nowrap; color: #8b949e; }
table.criteria td.criteria-why {
    color: #8b949e; line-height: 1.5; font-size: 12px; }
table.criteria tr.criteria-group td {
    background: #1c2128; color: #c9d1d9; font-size: 11px;
    text-transform: uppercase; letter-spacing: 0.05em;
    padding-top: 6px; padding-bottom: 6px;
}
table.criteria code { background: transparent; color: #c9d1d9; padding: 0; }
.section { background: #161b22; border: 1px solid #30363d; border-radius: 8px; overflow: hidden; margin-bottom: 24px; }
.section h2 { padding: 14px 22px 10px; margin: 0; }
/* Default inner padding so card rows / tables / paragraphs don't touch
   the section edge. Sections that needed different padding (.summary-body,
   .rules) override below. */
.section .body { padding: 14px 22px 18px; }
.bar { display: flex; align-items: baseline; gap: 8px; }
.bar .small, .small { font-size: 11px; color: #8b949e; }
td.mono, code { font-family: ui-monospace, SFMono-Regular, Consolas, monospace; font-size: 12px; }
/* Watchlist ticker links — keep the cell looking like the rest of the
   table at rest, only flip color + underline on hover so the affordance
   is discoverable without making the table feel like a wall of links. */
td.mono a.ticker-link { color: inherit; text-decoration: none; }
td.mono a.ticker-link:hover { color: #58a6ff; text-decoration: underline; }
code { background: #161b22; padding: 1px 6px; border-radius: 3px; color: #c9d1d9; }
/* hero-card sits inside the body which already has padding; only need
   internal vertical breathing room for multi-card scenarios. */
.hero-card { padding: 4px 0 6px 0; border-bottom: 1px solid #21262d; }
.hero-card:last-child { border-bottom: none; padding-bottom: 0; }
.hero-question { font-size: 22px; font-weight: 600; color: #f0f6fc; margin-bottom: 4px; }
.hero-question .badge { font-size: 13px; padding: 4px 10px; vertical-align: middle; }
.hero-question .hero-q-text { margin-left: 6px; }
.hero-question .hero-event-title { color: #f0f6fc; margin-right: 10px; }
.hero-ticker { font-family: ui-monospace, SFMono-Regular, Consolas, monospace; font-size: 12px; color: #8b949e; margin-bottom: 14px; }
.hero-stats { margin-bottom: 14px; }
.hero-chart { padding: 4px 0; }
.rules { padding: 0; }
.rules ol { margin: 0; padding-left: 20px; line-height: 1.8; }
.rules p { margin-top: 0; }
.rules li { color: #c9d1d9; font-size: 13px; }
.rules code { font-size: 12px; }
.summary-body { padding: 18px 22px; }
/* Compact cards: applied wherever we want a tight row of equal-width
   stat cards that fit on one line at desktop widths. Used by Summary
   and the Model-strength row. Centered labels/values for visual alignment.
*/
.summary-body .row,
.row.compact { gap: 10px; flex-wrap: nowrap; }
.summary-body .card,
.row.compact > .card { padding: 14px 14px; min-width: 0; flex: 1 1 0;
    display: flex; flex-direction: column; align-items: center;
    justify-content: center; text-align: center; min-height: 78px; }
.summary-body .card .label,
.row.compact > .card .label { font-size: 10px; margin-bottom: 6px; }
.summary-body .card .value,
.row.compact > .card .value { font-size: 22px; line-height: 1.2; }
.summary-body .card .small,
.row.compact > .card .small { font-size: 10px; margin-top: 4px; }
@media (max-width: 1100px) {
    .summary-body .row,
    .row.compact { flex-wrap: wrap; }
    .summary-body .card,
    .row.compact > .card { flex: 1 1 30%; min-width: 150px; }
}
.subhead { font-size: 12px; text-transform: uppercase; letter-spacing: 0.06em; color: #8b949e; margin: 16px 0 8px 0; font-weight: 600; }
.subsec { padding: 0 0 14px 0; }
.subsec h3 { margin-top: 12px; }
/* Bot filter bar — slim, sits between sections like a real filter, not
   like another content section. Pill-style links per bot. */
.bot-filter-bar { display: flex; align-items: center; gap: 10px;
    padding: 4px 0 18px 0; margin-bottom: 8px; flex-wrap: wrap;
    border-bottom: 1px solid #21262d; margin-top: -8px; }
.bot-filter-bar .filter-label {
    color: #8b949e; font-size: 11px; text-transform: uppercase;
    letter-spacing: 0.06em; font-weight: 600; margin-right: 4px;
}
/* Bot dropdown — native <select> styled to match the rest of the
   dashboard. The chevron is drawn via background-image so the look
   stays consistent across browsers. */
.bot-select {
    background: #0d1117 url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='10' height='10' viewBox='0 0 10 10'><path d='M2 3.5l3 3 3-3' fill='none' stroke='%238b949e' stroke-width='1.5'/></svg>") no-repeat right 10px center;
    color: #c9d1d9; border: 1px solid #30363d; border-radius: 6px;
    padding: 6px 30px 6px 12px;
    font-size: 13px; line-height: 1.4;
    appearance: none; -webkit-appearance: none; -moz-appearance: none;
    cursor: pointer; min-width: 200px;
    transition: border-color 120ms, background-color 120ms;
}
.bot-select:hover { border-color: #40464d; background-color: #161b22; }
.bot-select:focus { outline: none; border-color: #1f6feb;
    box-shadow: 0 0 0 3px rgba(31, 111, 235, 0.18); }
.bot-select option { background: #0d1117; color: #c9d1d9; }
.filter-pill { background: #21262d; color: #c9d1d9; text-decoration: none;
    padding: 6px 14px; border-radius: 999px; font-size: 13px;
    border: 1px solid #30363d; transition: background 120ms, border-color 120ms;
    line-height: 1.4; }
.filter-pill:hover { background: #2d333b; border-color: #40464d; }
.filter-pill-active { background: #1f6feb; color: #fff; border-color: #1f6feb; }
.filter-pill-active:hover { background: #1f6feb; border-color: #1f6feb; }
.filter-pill-disabled { color: #6e7681; cursor: not-allowed; opacity: 0.7; }
.filter-pill-disabled:hover { background: #21262d; border-color: #30363d; }
/* Tab bar for the per-bot detail panes. Same pill idiom as the bot/
   period filters above, slightly slimmer so the visual hierarchy reads
   "filter > tab > content". */
.tab-bar { display: flex; align-items: center; gap: 6px;
    padding: 0 0 10px 0; margin: 4px 0 12px;
    border-bottom: 1px solid #21262d; flex-wrap: wrap; }
.tab-pill { background: transparent; color: #8b949e; cursor: pointer;
    padding: 6px 14px; border-radius: 6px 6px 0 0; font-size: 13px;
    border: 1px solid transparent; line-height: 1.4;
    text-decoration: none; transition: color 120ms, background 120ms; }
.tab-pill:hover { color: #c9d1d9; background: #1c2128; }
.tab-pill-active { color: #f0f6fc; background: #21262d;
    border-color: #30363d; border-bottom-color: #21262d;
    margin-bottom: -1px; font-weight: 600; }
.tab-panel { display: none; }
.tab-panel-active { display: block; }
/* "Why?" button on each active-bets row + the criteria modal it
   opens. Single shared modal at page bottom; JS populates the body
   from data-criteria on the clicked button. */
/* Per-row info button — circle with an italic "i" inside, mirroring
   common information-icon affordances. */
.criteria-btn {
    background: #21262d; color: #8b949e; border: 1px solid #30363d;
    border-radius: 50%; width: 22px; height: 22px; padding: 0;
    font-family: Georgia, "Times New Roman", serif;
    font-style: italic; font-weight: 700;
    font-size: 13px; line-height: 1; cursor: pointer;
    display: inline-flex; align-items: center; justify-content: center;
    transition: background 120ms, border-color 120ms, color 120ms; }
.criteria-btn:hover { background: #2d333b; border-color: #1f6feb;
    color: #f0f6fc; }
/* Used for the "what does the bot need before it'll buy" reference
   popup, rendered inline next to the Active-bet h3 as the same
   circle-i info-icon affordance as the per-row criteria-btn. */
.criteria-rules-btn {
    background: #21262d; color: #8b949e; border: 1px solid #30363d;
    border-radius: 50%; width: 22px; height: 22px; padding: 0;
    font-family: Georgia, "Times New Roman", serif;
    font-style: italic; font-weight: 700;
    font-size: 13px; line-height: 1; cursor: pointer;
    display: inline-flex; align-items: center; justify-content: center;
    transition: background 120ms, border-color 120ms, color 120ms; }
.criteria-rules-btn:hover { background: #2d333b; border-color: #1f6feb;
    color: #f0f6fc; }
.criteria-overlay {
    position: fixed; inset: 0; background: rgba(0,0,0,0.55);
    z-index: 100; }
.criteria-modal {
    position: fixed; top: 50%; left: 50%;
    transform: translate(-50%, -50%);
    background: #0d1117; border: 1px solid #30363d; border-radius: 8px;
    /* Fit-to-content sizing: width sized by what's inside, capped at
       92vw so it never overflows. Min keeps the title/X spacing
       readable on tight payloads. */
    width: max-content; min-width: 320px; max-width: 92vw;
    max-height: 80vh;
    display: flex; flex-direction: column;
    z-index: 101; box-shadow: 0 12px 48px rgba(0,0,0,0.6); }
/* Fee suffix on the entry-cost cell — same red as the base amount
   (it's also a cash outflow). Keep the cell on one line so the
   "−$0.26 + $0.02" pattern stays scannable horizontally. */
.entry-fee { color: #f85149; font-weight: 400; margin-left: 2px; }
td.num.red, td.num.green { white-space: nowrap; }
/* Bot card drift badge — amber pill that lights up when the model's
   training accuracy and live actual-win-% diverge by >10pp on n≥10
   closed bets. Surfaces "this model may have drifted" as a one-look
   signal without forcing users to compare two cells. */
.drift-badge { display: inline-block; margin-left: 6px;
    padding: 1px 6px; border-radius: 4px;
    background: rgba(212, 153, 0, 0.18);
    color: #d49900; border: 1px solid rgba(212, 153, 0, 0.35);
    font-size: 9px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.04em; line-height: 1.5;
    vertical-align: 2px; }
/* The HTML `hidden` attribute applies `display: none` via the UA
   stylesheet (specificity 0,1,0). Our `.criteria-modal { display:
   flex }` rule shares that specificity and wins by source order, so
   the modal kept showing even after JS set `.hidden = true`. These
   attribute selectors (specificity 0,2,0) restore the expected
   behaviour for both the modal and the overlay. */
.criteria-modal[hidden]   { display: none !important; }
.criteria-overlay[hidden] { display: none !important; }
.criteria-modal-head {
    display: flex; align-items: baseline;
    justify-content: space-between;
    padding: 14px 18px; border-bottom: 1px solid #21262d; }
.criteria-modal-head h3 { margin: 0; font-size: 15px; font-weight: 700;
    color: #f0f6fc; }
.criteria-modal-head .ticker { font-family: ui-monospace, SFMono-Regular,
    Consolas, monospace; font-size: 11px; color: #8b949e; }
.criteria-modal-close {
    background: transparent; border: none; color: #8b949e;
    font-size: 20px; cursor: pointer; padding: 0 4px; line-height: 1;
    margin-left: 8px; }
.criteria-modal-close:hover { color: #f0f6fc; }
.criteria-modal-body {
    padding: 14px 18px; overflow-y: auto; font-size: 13px;
    color: #c9d1d9; line-height: 1.55; }
.criteria-modal-body dl { margin: 0; display: grid;
    grid-template-columns: max-content 1fr; gap: 6px 16px; }
.criteria-modal-body dt { color: #8b949e; }
.criteria-modal-body dd { margin: 0; color: #c9d1d9;
    font-variant-numeric: tabular-nums; }
.criteria-modal-body dd.green { color: #3fb950; }
.criteria-modal-body dd.red   { color: #f85149; }
.criteria-modal-body dd.gray  { color: #6e7681; }
.criteria-modal-body .crit-section { margin-top: 14px; }
.criteria-modal-body .crit-section h4 {
    font-size: 11px; text-transform: uppercase; letter-spacing: 0.06em;
    color: #8b949e; font-weight: 600; margin: 0 0 8px 0; }
/* Per-bot performance cards on the Performance tab. Cards align in a
   grid (auto-fit so they reflow at narrow widths) and are clickable —
   the whole card is an anchor to that bot's Watchlist tab. */
.bot-cards-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    grid-auto-rows: 1fr;
    gap: 14px;
}
.bot-card { display: flex; flex-direction: column;
    background: #0d1117; border: 1px solid #21262d;
    border-radius: 8px; padding: 14px 16px;
    color: inherit; text-decoration: none;
    transition: border-color 120ms, background 120ms,
                transform 120ms; }
.bot-card:hover {
    border-color: #1f6feb; background: #11161d;
    transform: translateY(-1px);
}
.bot-card-head { display: flex; align-items: baseline;
    justify-content: space-between; gap: 12px;
    border-bottom: 1px solid #21262d; padding-bottom: 8px;
    margin-bottom: 10px; }
.bot-card-head .bot-name { font-size: 14px; font-weight: 700;
    color: #f0f6fc; letter-spacing: -0.2px; }
.bot-card-head .bot-meta { font-size: 10px; color: #8b949e;
    text-transform: uppercase; letter-spacing: 0.04em; }
.bot-card dl { margin: 0; display: grid;
    grid-template-columns: max-content 1fr max-content 1fr;
    gap: 4px 12px;
    font-size: 12px; line-height: 1.45; }
.bot-card dt { color: #8b949e; }
.bot-card dd { margin: 0; color: #c9d1d9;
    font-variant-numeric: tabular-nums; text-align: right;
    font-weight: 500; }
/* High-specificity + !important so the green/red gain-loss colors
   land regardless of any other .green/.red cascade rules. */
.bot-card dl dd.green { color: #3fb950 !important; font-weight: 600; }
.bot-card dl dd.red   { color: #f85149 !important; font-weight: 600; }
.bot-card dl dd.gray  { color: #6e7681 !important; }
.bot-card-foot {
    margin-top: auto; padding-top: 10px;
    border-top: 1px solid #21262d;
    font-size: 10px; color: #6e7681;
    display: flex; justify-content: space-between;
    text-transform: uppercase; letter-spacing: 0.06em;
}
.bot-card-foot .arrow { color: #8b949e; }
/* Watchlist row that fails one or more validations (horizon mismatch,
   wide spread, edge<cost, etc.). Rendered visible but de-emphasized. */
tr.row-suspect td { opacity: 0.55; }
tr.row-suspect td:nth-last-child(2) { opacity: 0.85; }  /* keep gap legible */
/* Watchlist row matching the strike the bot currently holds an open
   position on. The user can see at a glance which strike is "live" —
   tinted background, a glowing left rail, and a BOUGHT pill prefixing
   the ticker. Wins specificity over row-suspect so a held position is
   never dimmed. */
tr.row-bought td { opacity: 1 !important; }
/* Side-colored treatment: green for YES, red for NO. A 3px colored
   left bar flags the row at a glance, and a faint tint runs across
   every cell so the held strike pops without overpowering the table.
   The first cell carries a slightly stronger tint near the bar so the
   bar reads as anchored, not floating. */
tr.row-bought.bought-yes td { background: rgba(63, 185, 80, 0.06); }
tr.row-bought.bought-no  td { background: rgba(248, 81, 73, 0.06); }
/* First cell keeps the colored left bar but uses the same tint as the
   rest of the row, so the row reads as one even band of color. */
tr.row-bought.bought-yes td:first-child { border-left: 3px solid #3fb950; }
tr.row-bought.bought-no  td:first-child { border-left: 3px solid #f85149; }
tr.row-bought.bought-yes td.mono a.ticker-link,
tr.row-bought.bought-yes td.mono { color: #3fb950; font-weight: 600; }
tr.row-bought.bought-no  td.mono a.ticker-link,
tr.row-bought.bought-no  td.mono { color: #f85149; font-weight: 600; }
/* Watchlist table: fixed scrolling viewport so the strike list never
   pushes the rest of the page off-screen. Sticky header keeps the
   column labels in view as the user scrolls. */
.watchlist-scroll { max-height: 360px; overflow-y: auto;
    border: 1px solid #21262d; border-radius: 6px;
    margin-top: 4px; }
.watchlist-scroll table { margin: 0; border: none; }
.watchlist-scroll thead th {
    position: sticky; top: 0; z-index: 1;
    background: #161b22; box-shadow: 0 1px 0 #30363d;
}
.section h2 .small { text-transform: none; letter-spacing: 0; font-size: 11px; font-weight: 400; }
/* Watchlist hero — Kalshi-style market header above the strikes table.
   Layout mirrors the live Kalshi market page: title + countdown on top,
   then a big current-value, % change, and total volume row, then the
   underlying chart. */
.wl-hero { background: #0d1117; border: 1px solid #21262d; border-radius: 8px;
    padding: 16px 18px; margin-bottom: 18px; }
.wl-hero-top { display: flex; align-items: baseline; justify-content: space-between;
    gap: 12px; margin-bottom: 12px; }
.wl-hero-stats { display: flex; align-items: baseline; gap: 14px;
    flex-wrap: wrap; }
.wl-hero-price { font-size: 24px; font-weight: 700; color: #f0f6fc;
    letter-spacing: -0.3px; }
.wl-hero-price-label { font-size: 12px; font-weight: 500; color: #8b949e;
    text-transform: lowercase; margin-left: 4px; letter-spacing: 0.02em; }
.wl-hero-change { font-size: 14px; font-weight: 600; color: #8b949e; }
.wl-hero-change.pos { color: #3fb950; }
.wl-hero-change.neg { color: #f85149; }
.wl-hero-mtc { font-size: 12px; color: #8b949e; flex: 0 0 auto; }
.wl-hero-mtc .label { color: #8b949e; text-transform: uppercase;
    letter-spacing: 0.04em; margin-right: 6px; font-size: 10px; }
.wl-hero-mtc .value { color: #c9d1d9; font-weight: 600; font-size: 13px; }
/* Hover crosshair on the underlying chart. JS draws the vertical line
   inside the SVG and positions this tooltip via inline `left:`. */
.wl-chart-wrap { position: relative; }
.wl-chart-tooltip {
    position: absolute; top: -8px; transform: translateX(-50%);
    background: #161b22; color: #c9d1d9;
    border: 1px solid #30363d; border-radius: 4px;
    padding: 4px 9px; font-size: 11px; font-weight: 500;
    pointer-events: none; white-space: nowrap; z-index: 2;
    text-align: center; line-height: 1.35;
}
.wl-chart-tooltip .wl-chart-tip-time { color: #8b949e; font-size: 10px; }
.wl-chart-tooltip .wl-chart-tip-value { color: #f0f6fc; font-size: 13px;
    font-weight: 600; }
"""


def render_page(
    model: dict | None,
    global_summary: dict,
    global_active_bets: List[dict],
    global_history: List[dict],
    latest_active: dict | None,
    bot_closed_positions: List[dict],
    watchlist: List[dict],
    underlying_history: List[dict],
    display: dict,
    kalshi_history: List[dict],
    atm_market: dict | None,
    contract_open_ts: float | None,
    contract_close_ts: float | None,
    event_title: str | None,
    risk_caps: dict,
    edge_cfg: dict,
    validator_cfg: dict,
    hedge_cfg: dict,
    available_bots: List[dict],
    current_bot: str,
    period_key: str = "all",
    tab_key: str = "home",
    bot_models: List[dict] | None = None,
) -> str:
    out: List[str] = []
    out.append("<!doctype html><html><head>")
    out.append("<meta charset='utf-8'>")
    # No meta-refresh — JS at the bottom of the page polls /api/snapshot
    # every 5s and patches live cells in place. The page never reloads.
    out.append(f"<title>Kalshi simulation dashboard</title>")
    out.append(f"<style>{CSS}</style>")
    out.append("</head><body>")
    out.append(f"<h1>Kalshi simulation dashboard</h1>")
    out.append(f"<div class='meta'>Loaded {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%SZ')}"
               f" · live updates every 5s · DRY-RUN mode (no real orders)</div>")

    # ── Top-level page tabs ───────────────────────────────────────────
    # All four panels live on the same page; clicks swap which one is
    # visible. URL persists the choice via ?tab=X.
    tabs = [
        ("home", "Home"),
        ("watchlist", "Watchlist"),
        ("history", "History"),
    ]
    valid_tabs = {k for k, _ in tabs}
    active_tab = tab_key if tab_key in valid_tabs else "home"
    out.append("<div class='tab-bar'>")
    for k, label in tabs:
        cls = "tab-pill" + (" tab-pill-active" if k == active_tab else "")
        out.append(
            f"<a class='{cls}' data-tab='{html.escape(k)}' "
            f"href='#tab-{html.escape(k)}'>{html.escape(label)}</a>"
        )
    out.append("</div>")

    def _open_panel(name: str) -> None:
        cls = "tab-panel" + (" tab-panel-active" if name == active_tab else "")
        out.append(f"<div class='{cls}' data-panel='{html.escape(name)}'>")

    period_label = next(
        (lbl for k, lbl, _ in PERIOD_OPTIONS if k == period_key),
        "All-time",
    )

    # ── HOME tab — summary cards + active bets + per-bot perf cards ──
    # Performance and Home were merged per user request; the bot-card
    # grid sits below the summary section as a "what's in each bot"
    # overview that doubles as a click-through to each bot's Watchlist.
    _open_panel("home")
    _render_summary(out, global_summary, global_active_bets, global_history,
                     period_key=period_key, current_bot=current_bot)
    out.append("<div class='section'><h2>Bot performance</h2>"
               "<div class='body'>")
    _render_bot_cards(out, global_summary, bot_models, period_label)
    out.append("</div></div>")
    out.append("</div>")  # /home panel

    # ── WATCHLIST tab — chart + strike ladder + Kalshi rules ─────────
    _open_panel("watchlist")
    if (not watchlist and not latest_active
            and not [b for b in available_bots
                     if b["key"] == current_bot and b.get("available")]):
        _render_bot_unavailable(out, current_bot)
    else:
        # Bot dropdown is rendered inside _render_watchlist (below the
        # section title, above the current-prediction card) so it sits
        # with the section it scopes.
        _render_watchlist(out, watchlist, model,
                          underlying_history=underlying_history,
                          display=display,
                          latest_active=latest_active,
                          kalshi_history=kalshi_history,
                          atm_market=atm_market,
                          contract_open_ts=contract_open_ts,
                          contract_close_ts=contract_close_ts,
                          event_title=event_title,
                          edge_cfg=edge_cfg,
                          validator_cfg=validator_cfg,
                          risk_caps=risk_caps,
                          hedge_cfg=hedge_cfg,
                          available_bots=available_bots,
                          current_bot=current_bot,
                          period_key=period_key)
        _render_contract_rules(out, watchlist, current_bot)
    out.append("</div>")  # /watchlist panel

    # ── HISTORY tab — closed-bet history across all bots ──────────────
    _open_panel("history")
    out.append(
        f"<div class='section'><h2>Contract history "
        f"<span class='small gray'>({html.escape(period_label)})"
        f"</span></h2>"
        f"<div class='body'>"
    )
    # Day/Week/Month/Year/All-time dropdown; clicking keeps the user
    # on the History tab and just narrows the rows.
    _render_period_filter(out, period_key, current_bot=current_bot,
                            tab_key="history")
    # Pass heading="" so the table renders without a duplicate
    # subhead — the section title above already carries the period.
    _render_bet_history_block(
        out, global_history,
        heading="",
        shown_initially=20,
    )
    out.append("</div></div>")
    out.append("</div>")  # /history panel

    # Live-update JS: polls /api/snapshot every 5s and patches summary
    # cards + watchlist cells in place. Pass the period so the live
    # cards keep matching the user's filter selection between polls.
    # Shared "Why?" modal — single instance, populated dynamically by
    # the JS hook when any .criteria-btn is clicked.
    out.append(
        "<div id='criteria-overlay' class='criteria-overlay' hidden></div>"
        "<div id='criteria-modal' class='criteria-modal' hidden>"
        "  <div class='criteria-modal-head'>"
        "    <div>"
        "      <h3>Why was this bet chosen?</h3>"
        "      <div class='ticker' id='criteria-modal-ticker'></div>"
        "    </div>"
        "    <button type='button' id='criteria-close' "
        "      class='criteria-modal-close' aria-label='Close'>×</button>"
        "  </div>"
        "  <div class='criteria-modal-body' id='criteria-modal-body'></div>"
        "</div>"
    )
    # Stash the gating config in a window global so the per-bet popup
    # can list "validators that were met" without bloating every
    # criteria-btn payload. These are global rules (same for all bots),
    # so one publish per page is enough.
    buy_criteria_payload = json.dumps({
        "edge": edge_cfg or {},
        "validators": validator_cfg or {},
        "risk": risk_caps or {},
        "hedge": hedge_cfg or {},
    }, separators=(",", ":"), default=str)
    out.append(
        f"<script>window.__BUY_CRITERIA__ = {buy_criteria_payload};</script>"
    )
    out.append(_live_update_script(current_bot, period_key=period_key))
    out.append("</body></html>")
    return "".join(out)


def _live_update_script(current_bot: str, period_key: str = "all") -> str:
    """Self-contained JS block that fetches /api/snapshot every 5s
    and patches DOM cells with new values. Highlights changed cells
    briefly so updates are visible.
    """
    bot_param = html.escape(current_bot)
    period_param = html.escape(period_key)
    return f"""<script>
(function () {{
  const BOT = "{bot_param}";
  const PERIOD = "{period_param}";
  const POLL_MS = 5000;

  // Format helpers — must mirror the server-side rendering in render_page.
  function fmtSignedCents(c) {{
    if (c === null || c === undefined) return "—";
    const dollars = Math.abs(c) / 100.0;
    const sign = c > 0 ? "+" : (c < 0 ? "−" : "");
    return sign + "$" + dollars.toFixed(2);
  }}
  function fmtPct(p, hasData) {{
    if (!hasData || p === null || p === undefined) return "—";
    return Math.round(p * 100) + "%";
  }}
  function fmtEv(ev) {{
    if (ev === null || ev === undefined) return "—";
    const sign = ev >= 0 ? "+" : "−";
    return "$" + sign + Math.abs(ev).toFixed(3);
  }}
  function evClass(ev, minEv) {{
    if (ev === null || ev === undefined) return "gray";
    if (ev >= minEv) return "green";
    if (ev > 0) return "yellow";
    return "red";
  }}
  function flash(el) {{
    if (!el) return;
    el.classList.add("cell-flash");
    setTimeout(function () {{ el.classList.remove("cell-flash"); }}, 800);
  }}
  function patch(id, newText, newClass) {{
    const el = document.getElementById(id);
    if (!el) return;
    if (el.textContent !== newText) {{
      el.textContent = newText;
      flash(el);
    }}
    if (newClass !== undefined) {{
      // Replace any of green/red/yellow/gray, keep other classes.
      el.classList.remove("green", "red", "yellow", "gray");
      if (newClass) el.classList.add(newClass);
    }}
  }}
  function patchCell(td, newText, newClass) {{
    if (!td) return;
    if (td.textContent !== newText) {{
      td.textContent = newText;
      flash(td);
    }}
    if (newClass !== undefined) {{
      td.classList.remove("green", "red", "yellow", "gray");
      if (newClass) td.classList.add(newClass);
    }}
  }}

  function applySnapshot(snap) {{
    // ── Summary cards ──────────────────────────────────────────────
    // 6 cards: Active bets (live) | Closed bets | Money spent
    // | Money gained | Net gain/loss | Win %. The middle four reflect
    // the period filter; the snapshot was already fetched with the
    // right window so we just patch in.
    const s = snap.summary || {{}};
    patch("card-active-bets", String(s.active_bets ?? 0));
    patch("card-closed-bets", String(s.period_closed_bets ?? 0));
    patch("card-money-spent",
          (s.period_money_spent_cents ?? 0) === 0
            ? "$0.00"
            : fmtSignedCents(-(s.period_money_spent_cents ?? 0)));
    patch("card-money-gained",
          "+" + fmtSignedCents(s.period_money_gained_cents).replace(/^[+−-]/, ""),
          "green");
    patch("card-net-pnl", fmtSignedCents(s.period_net_pnl_cents),
          (s.period_net_pnl_cents > 0) ? "green"
            : (s.period_net_pnl_cents < 0 ? "red" : "gray"));
    patch("card-win-pct", fmtPct(s.period_win_pct, !!s.period_has_closed),
          (s.period_win_pct > 0.5) ? "green"
            : (s.period_has_closed && s.period_win_pct < 0.5 ? "red" : "gray"));

    // ── Watchlist rows ─────────────────────────────────────────────
    const minEv = snap.min_ev || 0.03;
    const tbody = document.getElementById("watchlist-tbody");
    if (tbody && snap.watchlist) {{
      const rowsByTicker = {{}};
      tbody.querySelectorAll("tr[data-ticker]").forEach(function (tr) {{
        rowsByTicker[tr.getAttribute("data-ticker")] = tr;
      }});
      // Keep the "row-bought" highlight in sync with the active-bets
      // list — if a position opens or closes mid-poll, the held strike
      // gets/loses its blue rail without a full page reload. The
      // BOUGHT pill itself is server-rendered; if it's missing the
      // user can refresh to pick it up.
      const boughtBySide = {{}};
      (snap.active_bets || []).forEach(function (ab) {{
        if (ab && ab.ticker) {{
          const s = (ab.side || "").toUpperCase();
          boughtBySide[ab.ticker] = (s === "YES") ? "yes"
                                  : (s === "NO")  ? "no"
                                  : "yes";
        }}
      }});
      tbody.querySelectorAll("tr[data-ticker]").forEach(function (tr) {{
        const t = tr.getAttribute("data-ticker");
        const side = boughtBySide[t];
        tr.classList.remove("row-bought", "bought-yes", "bought-no");
        if (side) {{
          tr.classList.add("row-bought", "bought-" + side);
          tr.classList.remove("row-suspect");
        }}
      }});
      snap.watchlist.forEach(function (r) {{
        const tr = rowsByTicker[r.ticker];
        if (!tr) return;  // server added a new row — page reload would catch
        const ya = r.kalshi_yes, na = r.kalshi_no;
        const kyes = (ya !== null && ya !== undefined) ? (ya + "%")
                   : (na !== null && na !== undefined) ? ("~" + (100 - na) + "%")
                   : "—";
        const kno  = (na !== null && na !== undefined) ? (na + "%")
                   : (ya !== null && ya !== undefined) ? ("~" + (100 - ya) + "%")
                   : "—";
        const myYes = (r.model_prob_yes !== null && r.model_prob_yes !== undefined)
          ? (Math.round(r.model_prob_yes * 100) + "%") : "—";
        const myNo = (r.model_prob_yes !== null && r.model_prob_yes !== undefined)
          ? (Math.round((1 - r.model_prob_yes) * 100) + "%") : "—";
        patchCell(tr.querySelector("[data-field='oi']"),
                  r.open_interest !== null && r.open_interest !== undefined
                    ? Number(r.open_interest).toLocaleString() : "—");
        patchCell(tr.querySelector("[data-field='kyes']"), kyes);
        patchCell(tr.querySelector("[data-field='kno']"), kno);
        patchCell(tr.querySelector("[data-field='my_yes']"), myYes);
        patchCell(tr.querySelector("[data-field='my_no']"), myNo);
        patchCell(tr.querySelector("[data-field='ev_yes']"),
                  fmtEv(r.ev_yes), evClass(r.ev_yes, minEv));
        patchCell(tr.querySelector("[data-field='ev_no']"),
                  fmtEv(r.ev_no), evClass(r.ev_no, minEv));
      }});
    }}
  }}

  function poll() {{
    fetch("/api/snapshot?bot=" + encodeURIComponent(BOT)
          + "&period=" + encodeURIComponent(PERIOD),
          {{cache: "no-store"}})
      .then(function (r) {{ return r.ok ? r.json() : null; }})
      .then(function (snap) {{ if (snap) applySnapshot(snap); }})
      .catch(function () {{ /* swallow — try again next tick */ }});
  }}

  // Initial fetch on load + recurring poll.
  poll();
  setInterval(poll, POLL_MS);

  // ── Bot dropdown (Watchlist tab) + Period dropdowns ─────────────
  // Each <option>'s value carries the target URL; on change we
  // navigate there. Same destinations as the old pill links — the
  // dropdowns are just a quieter UI for the same action. The Period
  // selector appears on both Home and History tabs (one instance
  // each, marked with [data-period-select] so we can wire them all).
  const botSelect = document.getElementById("bot-select");
  if (botSelect) {{
    botSelect.addEventListener("change", function () {{
      const url = botSelect.value;
      if (url) window.location.href = url;
    }});
  }}
  document.querySelectorAll("[data-period-select]").forEach(function (sel) {{
    sel.addEventListener("change", function () {{
      const url = sel.value;
      if (url) window.location.href = url;
    }});
  }});

  // ── "Why?" modal — bet criteria popup ────────────────────────
  // Each .criteria-btn carries data-criteria with the entry-time
  // snapshot. On click we populate one shared modal at the bottom
  // of the page and reveal it; click overlay or × to dismiss.
  const critOverlay = document.getElementById("criteria-overlay");
  const critModal   = document.getElementById("criteria-modal");
  const critBody    = document.getElementById("criteria-modal-body");
  const critTicker  = document.getElementById("criteria-modal-ticker");
  const critClose   = document.getElementById("criteria-close");
  function fmtPct(v) {{
    if (v === null || v === undefined || !isFinite(v)) return "—";
    return (v * 100).toFixed(0) + "%";
  }}
  function fmtCents3(v) {{
    if (v === null || v === undefined || !isFinite(v)) return "—";
    const sign = v >= 0 ? "+" : "−";
    return sign + "$" + Math.abs(v).toFixed(3);
  }}
  function buildCriteriaHTML(c) {{
    // Every value in this popup is rendered green: the bet only
    // exists because each criterion cleared, so every line is a
    // "this passed" datapoint.
    let html = "<div class='crit-section'><h4>Why we took it</h4><dl>";
    html += "<dt>Model probability</dt><dd class='green'>"
         + fmtPct(c.model_p) + "</dd>";
    html += "<dt>Market probability</dt><dd class='green'>"
         + fmtPct(c.kalshi_p) + "</dd>";
    const edgeStr = (c.edge_pts === null || !isFinite(c.edge_pts))
      ? "—"
      : (c.edge_pts >= 0 ? "+" : "−")
        + Math.abs(c.edge_pts).toFixed(0) + " pts";
    html += "<dt>Edge</dt><dd class='green'>" + edgeStr + "</dd>";
    html += "<dt>Entry EV / contract</dt><dd class='green'>"
         + fmtCents3(c.entry_ev) + "</dd>";
    html += "<dt>Break-even probability</dt><dd class='green'>"
         + fmtPct(c.break_even) + "</dd>";
    html += "<dt>Validators met</dt><dd class='green'>100%</dd>";
    html += "</dl></div>";
    return html;
  }}
  function showCriteria(btn) {{
    if (!critOverlay || !critModal) return;
    let data = {{}};
    try {{ data = JSON.parse(btn.dataset.criteria || "{{}}"); }} catch (e) {{}}
    if (critTicker) critTicker.textContent = data.ticker || "";
    if (critBody)   critBody.innerHTML     = buildCriteriaHTML(data);
    critOverlay.hidden = false;
    critModal.hidden   = false;
  }}
  function hideCriteria() {{
    if (critOverlay) critOverlay.hidden = true;
    if (critModal)   critModal.hidden   = true;
  }}
  // Build the "buy criteria + validators" reference popup body from
  // the bot's edge/validator/risk/hedge configs serialised on the
  // button as data-rules.
  function fmtCents(c) {{
    if (c === null || c === undefined || !isFinite(c)) return "—";
    return "$" + (c / 100).toFixed(2);
  }}
  function fmtMin(m) {{
    if (m === null || m === undefined || !isFinite(m)) return "—";
    if (m >= 1440) return (m / 1440).toFixed(0) + "d";
    if (m >= 60)   return (m / 60).toFixed(0) + "h";
    return m + "min";
  }}
  function buildRulesHTML(r) {{
    const e = r.edge || {{}};
    const v = r.validators || {{}};
    const k = r.risk || {{}};
    const h = r.hedge || {{}};
    let html = "";
    html += "<div class='crit-section'><h4>Edge / EV thresholds</h4><dl>";
    html += "<dt>Min model edge (YES)</dt><dd>" + (e.min_edge_yes != null ? (e.min_edge_yes * 100).toFixed(0) + " pts" : "—") + "</dd>";
    html += "<dt>Min model edge (NO)</dt><dd>"  + (e.min_edge_no  != null ? (e.min_edge_no  * 100).toFixed(0) + " pts" : "—") + "</dd>";
    html += "<dt>Min model confidence</dt><dd>" + fmtPct(e.min_model_confidence) + "</dd>";
    html += "<dt>Min model accuracy</dt><dd>"   + fmtPct(e.min_model_accuracy) + "</dd>";
    html += "<dt>Min EV per contract</dt><dd>"  + (e.min_ev_per_contract != null ? "$" + Number(e.min_ev_per_contract).toFixed(2) : "—") + "</dd>";
    html += "<dt>Min edge over BE</dt><dd>"     + (e.min_prob_edge_over_breakeven != null ? (e.min_prob_edge_over_breakeven * 100).toFixed(0) + " pts" : "—") + "</dd>";
    html += "</dl></div>";

    html += "<div class='crit-section'><h4>Validators (must all pass)</h4><dl>";
    html += "<dt>Min book depth</dt><dd>"       + (v.min_book_depth_contracts != null ? v.min_book_depth_contracts + " contracts" : "—") + "</dd>";
    html += "<dt>Max spread</dt><dd>"           + (v.max_spread_cents != null ? v.max_spread_cents + "¢" : "—") + "</dd>";
    let ttc = "—";
    if (v.min_minutes_to_close != null && v.max_minutes_to_close != null)
      ttc = fmtMin(v.min_minutes_to_close) + " – " + fmtMin(v.max_minutes_to_close);
    html += "<dt>Time to close</dt><dd>" + ttc + "</dd>";
    let pb = "—";
    if (Array.isArray(v.prob_bounds_cents) && v.prob_bounds_cents.length === 2)
      pb = v.prob_bounds_cents[0] + "¢ – " + v.prob_bounds_cents[1] + "¢";
    html += "<dt>Probability bounds</dt><dd>" + pb + "</dd>";
    html += "<dt>Min volume</dt><dd>"           + (v.min_volume != null ? v.min_volume : "—") + "</dd>";
    html += "<dt>Min open interest</dt><dd>"    + (v.min_open_interest != null ? v.min_open_interest : "—") + "</dd>";
    html += "<dt>Min ask depth</dt><dd>"        + (v.min_depth_at_best_ask != null ? v.min_depth_at_best_ask : "—") + "</dd>";
    html += "<dt>Basis-risk strike window</dt><dd>" + (v.basis_risk_strike_window_dollars != null ? "±$" + Number(v.basis_risk_strike_window_dollars).toFixed(2) : "—") + "</dd>";
    html += "<dt>Basis-risk max hours</dt><dd>" + (v.basis_risk_max_hours_to_close != null ? v.basis_risk_max_hours_to_close + "h" : "—") + "</dd>";
    html += "</dl></div>";

    html += "<div class='crit-section'><h4>Risk caps</h4><dl>";
    html += "<dt>Bet size</dt><dd>"             + fmtCents(k.bet_size_cents) + "</dd>";
    html += "<dt>Max open positions</dt><dd>"   + (k.max_open_positions ?? "—") + "</dd>";
    html += "<dt>Max total exposure</dt><dd>"   + fmtCents(k.max_total_exposure_cents) + "</dd>";
    html += "<dt>Max bets per day</dt><dd>"     + (k.max_bets_per_day ?? "—") + "</dd>";
    html += "<dt>Cooldown (same market)</dt><dd>" + (k.cooldown_seconds_same_market != null ? Math.round(k.cooldown_seconds_same_market / 60) + " min" : "—") + "</dd>";
    html += "</dl></div>";

    html += "<div class='crit-section'><h4>Hedging</h4><dl>";
    html += "<dt>Enabled</dt><dd>"              + (h.enabled ? "Yes" : "No") + "</dd>";
    html += "<dt>Profit-lock</dt><dd>"          + (h.profit_lock_cents != null ? h.profit_lock_cents + "¢" : "—") + "</dd>";
    html += "<dt>Stop-loss</dt><dd>"            + (h.stop_loss_cents != null ? h.stop_loss_cents + "¢" : "—") + "</dd>";
    html += "<dt>Hedge size fraction</dt><dd>"  + (h.hedge_size_fraction != null ? Number(h.hedge_size_fraction).toFixed(2) : "—") + "</dd>";
    html += "</dl></div>";

    html += "<div class='crit-section' style='font-size:11px;color:#8b949e;'>"
         + "Every contract the bot considers must clear all of these gates "
         + "before a bet is placed. The Why? button on each open position "
         + "shows what the bot saw at entry-time for that specific bet."
         + "</div>";
    return html;
  }}
  function showRules(btn) {{
    if (!critOverlay || !critModal) return;
    let data = {{}};
    try {{ data = JSON.parse(btn.dataset.rules || "{{}}"); }} catch (e) {{}}
    const h3 = critModal.querySelector("h3");
    if (h3) h3.textContent = "Buy criteria & validators";
    if (critTicker) critTicker.textContent = "";
    if (critBody)   critBody.innerHTML = buildRulesHTML(data);
    critOverlay.hidden = false;
    critModal.hidden   = false;
  }}

  document.addEventListener("click", function (e) {{
    const ruleBtn = e.target.closest(".criteria-rules-btn");
    if (ruleBtn) {{
      e.preventDefault();
      showRules(ruleBtn);
      return;
    }}
    const btn = e.target.closest(".criteria-btn");
    if (btn) {{
      e.preventDefault();
      // Restore the per-bet header — the rules popup may have changed it.
      const h3 = critModal && critModal.querySelector("h3");
      if (h3) h3.textContent = "Why was this bet chosen?";
      showCriteria(btn);
    }}
  }});
  if (critOverlay) critOverlay.addEventListener("click", hideCriteria);
  if (critClose)   critClose.addEventListener("click", hideCriteria);
  document.addEventListener("keydown", function (e) {{
    if (e.key === "Escape") hideCriteria();
  }});

  // ── Tab switcher ────────────────────────────────────────────────
  // Clicks on a tab pill toggle the .tab-pill-active class on the bar
  // and the .tab-panel-active class on the matching panel. Updates
  // ?tab=X via history.replaceState so reloads + the period filter
  // preserve the active tab.
  const tabBar = document.querySelector(".tab-bar");
  if (tabBar) {{
    tabBar.querySelectorAll(".tab-pill").forEach(function (pill) {{
      pill.addEventListener("click", function (e) {{
        e.preventDefault();
        const key = pill.getAttribute("data-tab");
        if (!key) return;
        // Special-case History: full-page navigate WITHOUT preserving
        // the period — the History tab should default to all-time
        // every time the user opens it. Other tabs JS-swap (snappy
        // and preserves period from elsewhere on the page).
        if (key === "history") {{
          window.location.href = "?tab=history";
          return;
        }}
        tabBar.querySelectorAll(".tab-pill").forEach(function (p) {{
          p.classList.toggle("tab-pill-active",
                              p.getAttribute("data-tab") === key);
        }});
        document.querySelectorAll(".tab-panel").forEach(function (panel) {{
          panel.classList.toggle("tab-panel-active",
                                   panel.getAttribute("data-panel") === key);
        }});
        try {{
          const url = new URL(window.location.href);
          url.searchParams.set("tab", key);
          history.replaceState(null, "", url.toString());
        }} catch (err) {{ /* old browser; skip */ }}
      }});
    }});
  }}

  // ── Hover crosshair on the underlying chart ───────────────────
  // The SVG carries data-* attrs with t_min/t_max + chart geometry.
  // On mousemove we draw a vertical line and position a "May 1 at 9 AM"
  // tooltip; on mouseleave we hide them. Pure DOM, no chart library.
  document.querySelectorAll(".wl-chart-wrap").forEach(function (wrap) {{
    const svg = wrap.querySelector("svg");
    const tip = wrap.querySelector(".wl-chart-tooltip");
    if (!svg || !tip) return;
    const tmin = parseFloat(wrap.dataset.tmin);
    const tmax = parseFloat(wrap.dataset.tmax);
    const padL = parseFloat(wrap.dataset.padl);
    const innerW = parseFloat(wrap.dataset.innerw);
    const padT = parseFloat(wrap.dataset.padt);
    const padB = parseFloat(wrap.dataset.padb);
    const h = parseFloat(wrap.dataset.h);
    const vbW = parseFloat(wrap.dataset.vbw);
    if (![tmin, tmax, padL, innerW, padT, padB, h, vbW].every(isFinite)) return;

    const ns = "http://www.w3.org/2000/svg";
    // Dim overlay covers the right portion of the chart (everything
    // past the cursor) so the line in that region greys out as the
    // user "scrubs" through. Appended before the cursor line so it
    // sits beneath it in z-order. Using the panel background color
    // (#0d1117) at 0.65 opacity gives a clean greyed-out look without
    // hiding the line entirely.
    const dimRect = document.createElementNS(ns, "rect");
    dimRect.setAttribute("y", padT);
    dimRect.setAttribute("height", h - padB - padT);
    dimRect.setAttribute("fill", "#0d1117");
    dimRect.setAttribute("opacity", "0");
    dimRect.setAttribute("pointer-events", "none");
    svg.appendChild(dimRect);

    const cursor = document.createElementNS(ns, "line");
    cursor.setAttribute("stroke", "#c9d1d9");
    cursor.setAttribute("stroke-width", "1");
    cursor.setAttribute("stroke-dasharray", "2,3");
    cursor.setAttribute("opacity", "0");
    cursor.setAttribute("pointer-events", "none");
    svg.appendChild(cursor);

    // The hero forecast price + change-indicator elements (top-left
    // of the chart card). While the user scrubs the chart, we swap
    // the price for the value at the cursor AND the change indicator
    // for (cursor − earliest); on mouseleave we restore both from
    // their data-current-text / data-current-class attrs.
    const hero = wrap.closest(".wl-hero");
    const heroPrice = hero ? hero.querySelector(".wl-hero-price") : null;
    const heroPriceText = heroPrice ? heroPrice.querySelector(
        ".wl-hero-price-text") : null;
    const heroChange = hero ? hero.querySelector(".wl-hero-change") : null;

    function fmtTs(ts) {{
      const d = new Date(ts * 1000);
      const months = ["Jan","Feb","Mar","Apr","May","Jun",
                      "Jul","Aug","Sep","Oct","Nov","Dec"];
      const month = months[d.getUTCMonth()];
      const day = d.getUTCDate();
      let hour = d.getUTCHours();
      const minute = d.getUTCMinutes();
      const ampm = hour >= 12 ? "PM" : "AM";
      hour = hour % 12 || 12;
      const minStr = minute === 0 ? "" : ":" + (minute < 10 ? "0" : "") + minute;
      return month + " " + day + " at " + hour + minStr + " " + ampm;
    }}

    // (ts, value) pairs + the bot's display formatting for the hover
    // tooltip. Always snaps to the nearest recorded point so scrubbing
    // anywhere across the chart shows the time + value of the closest
    // forecast Kalshi recorded — no tolerance check, no interpolation.
    let points = [];
    let fmt = {{ divisor: 1.0, decimals: 2, unit: "", unit_position: "prefix" }};
    try {{ points = JSON.parse(wrap.dataset.points || "[]"); }} catch (e) {{}}
    try {{ fmt = Object.assign(fmt, JSON.parse(wrap.dataset.fmt || "{{}}")); }}
    catch (e) {{}}
    // Earliest recorded value — anchor for the (Δ from start of chart)
    // delta the change indicator displays. Computed AFTER points are
    // parsed (let-declared above; would TDZ-throw if accessed earlier).
    const earliestValue = points.length ? points[0][1] : null;

    function fmtValue(raw) {{
      if (raw === null || raw === undefined || !isFinite(raw)) return "—";
      const v = raw / (fmt.divisor || 1);
      const n = v.toLocaleString("en-US", {{
        minimumFractionDigits: fmt.decimals,
        maximumFractionDigits: fmt.decimals,
      }});
      if (fmt.unit_position === "prefix") return (fmt.unit || "") + n;
      if (fmt.unit_position === "suffix") return n + (fmt.unit || "");
      return n;
    }}

    // Snap to the nearest recorded point; always returns one as long
    // as the points array is non-empty. The cursor's mapping is to the
    // closest recorded forecast — both the popup date stamp and value
    // come from that point, so they always agree.
    function nearestPoint(ts) {{
      if (!points.length) return null;
      let lo = 0, hi = points.length - 1;
      if (ts <= points[lo][0]) {{
        return {{ ts: points[lo][0], value: points[lo][1] }};
      }}
      if (ts >= points[hi][0]) {{
        return {{ ts: points[hi][0], value: points[hi][1] }};
      }}
      while (hi - lo > 1) {{
        const mid = (lo + hi) >> 1;
        if (points[mid][0] <= ts) lo = mid; else hi = mid;
      }}
      const dLo = Math.abs(ts - points[lo][0]);
      const dHi = Math.abs(ts - points[hi][0]);
      const closer = dLo <= dHi ? points[lo] : points[hi];
      return {{ ts: closer[0], value: closer[1] }};
    }}

    // Format an absolute delta in the bot's units, no sign. The arrow
    // (▲/▼) is added separately so we can color it via class.
    function fmtDeltaAbs(raw) {{
      if (raw === null || raw === undefined || !isFinite(raw)) return "—";
      const v = Math.abs(raw) / (fmt.divisor || 1);
      const n = v.toLocaleString("en-US", {{
        minimumFractionDigits: fmt.decimals,
        maximumFractionDigits: fmt.decimals,
      }});
      if (fmt.unit_position === "prefix") return (fmt.unit || "") + n;
      if (fmt.unit_position === "suffix") return n + (fmt.unit || "");
      return n;
    }}

    function restoreHero() {{
      if (heroPrice && heroPriceText) {{
        heroPriceText.textContent = heroPrice.dataset.currentText || "";
      }}
      if (heroChange) {{
        heroChange.textContent = heroChange.dataset.currentText || "";
        const cls = heroChange.dataset.currentClass || "";
        heroChange.className = "wl-hero-change" + (cls ? " " + cls : "");
      }}
    }}

    svg.addEventListener("mousemove", function (e) {{
      const rect = svg.getBoundingClientRect();
      // Cursor's x in viewBox space (the SVG scales to the wrap's width).
      const x = (e.clientX - rect.left) * vbW / rect.width;
      if (x < padL || x > padL + innerW) {{
        cursor.setAttribute("opacity", "0");
        dimRect.setAttribute("opacity", "0");
        tip.hidden = true;
        restoreHero();
        return;
      }}
      cursor.setAttribute("x1", x);
      cursor.setAttribute("x2", x);
      cursor.setAttribute("y1", padT);
      cursor.setAttribute("y2", h - padB);
      cursor.setAttribute("opacity", "0.7");
      // Grey out the line to the right of the cursor.
      dimRect.setAttribute("x", x);
      dimRect.setAttribute("width", Math.max(0, padL + innerW - x));
      dimRect.setAttribute("opacity", "0.65");

      const frac = (x - padL) / innerW;
      const cursorTs = tmin + frac * (tmax - tmin);
      const np = nearestPoint(cursorTs);
      // When a recorded point is in range, stamp the popup AND swap
      // the hero forecast price for the value at the cursor. When no
      // point is near (gap in the data) we still show the cursor date
      // in the popup, but leave the hero on the live current so we
      // never display an unsourced value up top.
      if (np !== null) {{
        tip.innerHTML =
          "<div class='wl-chart-tip-time'>" + fmtTs(np.ts) + "</div>"
          + "<div class='wl-chart-tip-value'>" + fmtValue(np.value) + "</div>";
        if (heroPriceText) heroPriceText.textContent = fmtValue(np.value);
        // Update the ▲/▼ change indicator to (cursor value − earliest)
        // — same Δ semantics as the live indicator, just anchored to
        // wherever the user is hovering instead of "now".
        if (heroChange && earliestValue !== null) {{
          const delta = np.value - earliestValue;
          const arrow = delta >= 0 ? "▲" : "▼";
          const cls = delta >= 0 ? "pos" : "neg";
          heroChange.textContent = arrow + " " + fmtDeltaAbs(delta);
          heroChange.className = "wl-hero-change " + cls;
        }}
      }} else {{
        tip.innerHTML =
          "<div class='wl-chart-tip-time'>" + fmtTs(cursorTs) + "</div>";
        restoreHero();
      }}
      tip.hidden = false;
      // Anchor the tooltip in pixel space relative to the wrap so it
      // tracks the cursor regardless of how the SVG is scaled.
      const ratio = rect.width / vbW;
      tip.style.left = (x * ratio) + "px";
    }});

    svg.addEventListener("mouseleave", function () {{
      cursor.setAttribute("opacity", "0");
      dimRect.setAttribute("opacity", "0");
      restoreHero();
      tip.hidden = true;
    }});
  }});
}})();
</script>"""


# --------------------------------------------------------------------------- #
# Section helpers
# --------------------------------------------------------------------------- #

PERIOD_OPTIONS = [
    ("day", "Day", 1),
    ("week", "Week", 7),
    ("month", "Month", 30),
    ("year", "Year", 365),
    ("all", "All-time", None),
]


def _period_days(period_key: str) -> int | None:
    """Map ``?period=X`` query value to the rolling-window day count
    used by the SQL filters. Unknown / missing → None (lifetime).
    """
    for key, _label, days in PERIOD_OPTIONS:
        if key == period_key:
            return days
    return None


def _render_period_filter(out: List[str], period_key: str,
                            current_bot: str = "",
                            tab_key: str = "home") -> None:
    """Period filter dropdown (Day · Week · Month · Year · All-time).
    Uses the same wrapper + select styling as the Watchlist tab's bot
    selector so the three filters across the dashboard read as one
    consistent UI control.
    """
    select_id = f"period-select-{html.escape(tab_key)}"
    out.append("<div class='bot-filter-bar'>")
    out.append(f"<label for='{select_id}' class='filter-label'>"
               f"Period</label>")
    out.append(
        f"<select id='{select_id}' class='bot-select' "
        f"data-period-select>"
    )
    for key, label, _days in PERIOD_OPTIONS:
        bot_qs = (f"&bot={html.escape(current_bot)}"
                  if current_bot else "")
        tab_qs = (f"&tab={html.escape(tab_key)}"
                  if tab_key and tab_key != "home" else "")
        href = f"?period={key}{bot_qs}{tab_qs}"
        sel = " selected" if key == period_key else ""
        out.append(
            f"<option value='{html.escape(href)}'{sel}>"
            f"{html.escape(label)}</option>"
        )
    out.append("</select>")
    out.append("</div>")


def _render_summary(out: List[str], rollup: dict, active_bets: List[dict],
                    history: List[dict],
                    period_key: str = "all",
                    current_bot: str = "") -> None:
    """Section 1 — global cross-bot summary. Period filter pills control
    the "Bets made / Net gain / Total win %" cards; "Active bets" is
    always live (per user spec). Switching bots in the per-bot filter
    below leaves these cards unchanged.
    """
    period_label = next(
        (lbl for k, lbl, _ in PERIOD_OPTIONS if k == period_key),
        "All-time",
    )
    bets_made = rollup.get("period_bets_made", 0)
    net = rollup.get("period_net_pnl_cents", 0)
    pnl_cls = "green" if net > 0 else ("red" if net < 0 else "gray")
    win_pct = rollup.get("period_win_pct", 0.0)
    has_closed = (rollup.get("period_wins", 0)
                  + rollup.get("period_losses", 0)) > 0
    win_cls = ("green" if win_pct > 0.5
               else ("red" if has_closed and win_pct < 0.5 else "gray"))
    win_pct_str = f"{win_pct*100:.0f}%" if has_closed else "—"

    out.append("<div class='section'><h2>1 · Summary — across all bots</h2>"
               "<div class='body summary-body'>")

    # ── Period filter pills (shared helper, also used on History) ────
    _render_period_filter(out, period_key, current_bot=current_bot,
                            tab_key="home")

    # ── Headline cards ────────────────────────────────────────────────
    # 6 cards (Potential gains was dropped per user request). Card
    # labels carry no "(period)" tag — the dropdown above scopes
    # everything visible on the page already, so the parenthetical
    # was redundant.
    closed_bets = (rollup.get("period_wins", 0)
                   + rollup.get("period_losses", 0))
    money_spent = rollup.get("period_money_spent_cents", 0)
    money_gained = rollup.get("period_money_gained_cents", 0)
    out.append("<div class='row'>")
    out.append(f"<div class='card'><div class='label' "
               f"title='Live count of currently-open positions across "
               f"all bots. Not affected by the period filter.'>"
               f"Active bets</div>"
               f"<div class='value' id='card-active-bets'>"
               f"{rollup['active_bets']}</div></div>")
    out.append(f"<div class='card'><div class='label'>"
               f"Closed bets</div>"
               f"<div class='value' id='card-closed-bets'>"
               f"{closed_bets}</div></div>")
    out.append(f"<div class='card'><div class='label' "
               f"title='Total cost basis of every position opened in "
               f"the period (entry × contracts).'>"
               f"Money spent</div>"
               f"<div class='value' id='card-money-spent'>"
               f"{fmt_signed_cents(-money_spent)}</div></div>")
    out.append(f"<div class='card'><div class='label' "
               f"title='Total payout received from positions closed in "
               f"the period (entry × contracts + realized P&amp;L).'>"
               f"Money gained</div>"
               f"<div class='value green' id='card-money-gained'>"
               f"+{fmt_signed_cents(money_gained).lstrip('+')}</div></div>")
    out.append(f"<div class='card'><div class='label'>"
               f"Net gain / loss</div>"
               f"<div class='value {pnl_cls}' id='card-net-pnl'>"
               f"{fmt_signed_cents(net)}</div></div>")
    out.append(f"<div class='card'><div class='label' "
               f"title='Wins divided by closed bets in the selected "
               f"period. 0-100%; above 50% means winning more than "
               f"losing.'>"
               f"Total win %</div>"
               f"<div class='value {win_cls}' id='card-win-pct'>"
               f"{win_pct_str}</div></div>")
    out.append("</div>")

    # Active bets list — same table used in the per-bot view below.
    out.append("<h3 class='subhead'>Active bets — currently open</h3>")
    _render_active_bets_table(out, active_bets, empty_msg="No active bets right now.")

    out.append("</div></div>")


def _render_bot_cards(out: List[str], rollup: dict,
                        bot_models: List[dict] | None,
                        period_label: str) -> None:
    """Per-bot card grid for the Performance tab. Compact, clickable —
    each card is an anchor to the bot's Watchlist tab. Cards align on
    a fixed grid (auto-fit minmax 280px) so they share row + column
    edges. Contract rules live on the Watchlist tab to keep these
    cards skimmable.
    """
    if not bot_models:
        out.append("<div class='empty'>No bot data yet.</div>")
        return

    # Per-bot perf rows — used for the period-scoped Gain/loss cell.
    perf_by_name = {name: s for name, s in (rollup.get("per_bot") or [])}

    def _fmt_pct(v, decimals=0):
        if v is None:
            return "—"
        try:
            return f"{float(v)*100:.{decimals}f}%"
        except (TypeError, ValueError):
            return "—"

    out.append("<div class='bot-cards-grid'>")
    for entry in bot_models:
        b = entry.get("bot") or {}
        m = entry.get("model") or {}
        name = b.get("name", "—")
        bot_key = b.get("key", "")
        series_ticker = b.get("series_ticker") or "—"
        # Period-scoped net P&L from this bot's per-bot summary row.
        perf = perf_by_name.get(name, {})
        gain_loss = perf.get("period_net_pnl_cents", 0) or 0
        gl_cls = ("green" if gain_loss > 0
                   else ("red" if gain_loss < 0 else "gray"))
        gl_str = fmt_signed_cents(gain_loss)
        # Each card is a link to the bot's Watchlist tab.
        href = (f"?tab=watchlist&bot={html.escape(bot_key)}"
                if bot_key else "#")

        # Compute drift-badge HTML (if any) up-front so it can be
        # rendered inline with the bot name in the card header.
        # Drift = |training accuracy − live actual-win-%| > 10pp on
        # n ≥ 10 closed bets.
        ACTUAL_WIN_MIN_N = 10
        DRIFT_PP_THRESHOLD = 0.10
        drift_html = ""
        if m:
            a_wins_pre = int(m.get("actual_wins") or 0)
            a_losses_pre = int(m.get("actual_losses") or 0)
            a_total_pre = a_wins_pre + a_losses_pre
            try:
                acc_train_pre = float(m.get("classifier_accuracy") or 0)
            except (TypeError, ValueError):
                acc_train_pre = 0.0
            if a_total_pre >= ACTUAL_WIN_MIN_N and acc_train_pre > 0:
                a_pct_pre = a_wins_pre / a_total_pre
                if abs(acc_train_pre - a_pct_pre) > DRIFT_PP_THRESHOLD:
                    gap_pp = int(round(abs(acc_train_pre - a_pct_pre) * 100))
                    drift_html = (
                        f"<span class='drift-badge' "
                        f"title='Training accuracy ({acc_train_pre*100:.0f}%) "
                        f"and live actual-win-% ({a_pct_pre*100:.0f}%) differ "
                        f"by {gap_pp}pp on {a_total_pre} closed bets — model "
                        f"may have drifted; a retrain is likely overdue.'"
                        f">⚠ drift</span>"
                    )
        out.append(f"<a class='bot-card' href='{href}'>")
        out.append("<div class='bot-card-head'>")
        out.append(
            f"<div class='bot-name'>{html.escape(name)}{drift_html}</div>"
        )
        out.append(f"<div class='bot-meta'>{html.escape(series_ticker)}</div>")
        out.append("</div>")

        if not m:
            out.append("<dl><dt class='gray'>Model</dt>"
                       "<dd class='gray' style='grid-column:span 3;text-align:left;'>"
                       "no snapshot yet</dd></dl>")
        else:
            a_wins = int(m.get("actual_wins") or 0)
            a_losses = int(m.get("actual_losses") or 0)
            a_total = a_wins + a_losses
            # Sample-size guard: hide the % on n<10 — a single closed
            # loss reading "0%" is misleading. Show "learning (n=X)"
            # so users know the metric is warming up.
            a_pct = a_wins / a_total if a_total > 0 else None
            if a_total >= ACTUAL_WIN_MIN_N:
                a_str = f"{a_pct*100:.0f}%"
                a_cls = ("green" if a_pct > 0.55
                         else ("red" if a_pct < 0.45 else ""))
            elif a_total > 0:
                a_str = f"learning (n={a_total})"
                a_cls = "gray"
            else:
                a_str = "—"
                a_cls = "gray"
            features = int(m.get("feature_count") or 0)
            out.append("<dl>")
            out.append(f"<dt>Accuracy</dt><dd>{_fmt_pct(m.get('classifier_accuracy'), 1)}</dd>"
                        f"<dt>F1</dt><dd>{_fmt_pct(m.get('training_f1'))}</dd>")
            out.append(f"<dt>Precision</dt><dd>{_fmt_pct(m.get('training_precision'))}</dd>"
                        f"<dt>ROC AUC</dt><dd>{_fmt_pct(m.get('training_roc_auc'))}</dd>")
            out.append(f"<dt>Recall</dt><dd>{_fmt_pct(m.get('training_recall'))}</dd>"
                        f"<dt>Features</dt><dd>{features}</dd>")
            out.append(f"<dt>Actual win %</dt><dd class='{a_cls}'>{a_str}</dd>"
                        f"<dt>Gain / loss</dt><dd class='{gl_cls}'>{gl_str}</dd>")
            out.append("</dl>")

        # Footer hints at the click affordance — same idiom as the
        # ticker cells in the watchlist (subtle "go here" signal).
        out.append("<div class='bot-card-foot'>"
                   "<span>View watchlist</span>"
                   "<span class='arrow'>›</span>"
                   "</div>")
        out.append("</a>")  # /bot-card
    out.append("</div>")  # /bot-cards-grid


def _render_active_bets_table(out: List[str], bets: List[dict],
                              empty_msg: str = "No active bets.",
                              show_bot: bool = True) -> None:
    """Shared renderer used by both Section 1 (cross-bot summary) and
    the per-bot view inside the Watchlist tab. Columns:
        Opened | [Bot] | Ticker | Question | Contracts | Side
        | Entry cost | Current | Potential gain | Closes in
    The Bot column is skipped when ``show_bot`` is False (per-bot view
    where the bot is implied by the surrounding section). Entry cost /
    Current / Potential gain are in dollars (per-position totals).
    """
    if not bets:
        out.append(f"<div class='empty'>{html.escape(empty_msg)}</div>")
        return
    bot_th = "<th>Bot</th>" if show_bot else ""
    # Last column is the per-row info button — no header label needed
    # (the icon is self-explanatory; tooltip on hover spells it out).
    out.append("<table><thead><tr>"
               f"<th>Opened</th>{bot_th}<th>Ticker</th><th>Question</th>"
               "<th class='num'>Contracts</th><th>Side</th>"
               "<th class='num' title='Implied probability of our side at entry (= entry price in ¢).'>Entry prob</th>"
               "<th class='num' title='Implied probability of our side right now, taken from the market mid.'>Current prob</th>"
               "<th class='num' title='Entry prob × contracts + Kalshi entry fee — total cash out at open'>Entry cost</th>"
               "<th class='num' title='(100¢ − entry) × contracts − entry fee — gross profit if our side wins'>Potential gain</th>"
               "<th class='num' title='Time until the contract resolves'>Closes in</th>"
               "<th></th>"
               "</tr></thead><tbody>")
    for b in bets:
        opened = (b.get("opened_at") or "")[:19].replace("T", " ")
        side = (b.get("side") or "").upper()
        badge_cls = "badge-yes" if side == "YES" else "badge-no"
        entry = b.get("entry_price_cents") or 0
        contracts = b.get("contracts", 0) or 0
        bot_name = b.get("_bot_name", "—")
        # Question — rendered in the bot's native display units
        # ($/gal, K claims, $/MMBtu) when display config is attached.
        floor = b.get("floor_strike")
        cap = b.get("cap_strike")
        try:
            strike_low = float(floor) if floor is not None else None
        except (TypeError, ValueError):
            strike_low = None
        try:
            strike_high = float(cap) if cap is not None else None
        except (TypeError, ValueError):
            strike_high = None
        direction = "between" if (strike_low is not None
                                   and strike_high is not None) else "above"
        question = question_str(direction, strike_low, strike_high,
                                  display=b.get("_display"))
        # Probability columns — entry prob is just entry_price_cents
        # (1 cent = 1% implied probability for the side bet on).
        # Current prob is the market's view of "this side wins" right
        # now, derived from the mid where available with graceful
        # fallbacks for bots that don't write position_marks.
        entry_prob_pct = entry  # cents == percent
        # Compute mid for the YES side first, then flip for NO bets.
        mid_yes = b.get("mark_mid")
        if mid_yes is None:
            ya = b.get("mark_yes_ask")
            yb = b.get("mark_yes_bid")
            if ya is not None and yb is not None:
                mid_yes = (int(ya) + int(yb)) / 2.0
            elif ya is not None:
                mid_yes = int(ya)
            else:
                # Derive from the opposing side's ask (no_ask ≈ 100−yes)
                na = b.get("mark_no_ask")
                if na is not None:
                    mid_yes = max(0, 100 - int(na))
        if mid_yes is None:
            current_prob_pct = None
        else:
            current_prob_pct = (float(mid_yes) if side == "YES"
                                 else 100.0 - float(mid_yes))
        # Dollar columns — all incorporate Kalshi trading fees:
        #   Entry cost     = entry prob × contracts + entry fee
        #   Potential gain = (100 − entry) × contracts − entry fee
        #                    (entry fee already paid; settlement at
        #                     100¢ has zero exit fee)
        entry_fee_c = kalshi_fee_cents(entry, contracts)
        entry_cost_base = entry * contracts / 100.0
        entry_fee_dollars = entry_fee_c / 100.0
        potential_gain = ((100 - entry) * contracts - entry_fee_c) / 100.0
        # Entry-cost cell shows base + fee inline so the user sees
        # how much of the cost is fee. Tooltip explains.
        entry_cost_cell = (
            f"<td class='num red' title='Entry prob × contracts + "
            f"Kalshi entry fee — total cash out at open'>"
            f"−${entry_cost_base:.2f}"
            f"<span class='entry-fee'> + ${entry_fee_dollars:.2f}</span>"
            f"</td>"
        )
        # Probability cells — both rendered in the default white text;
        # the user can compare entry vs current at a glance without
        # the color cue (which was tracking direction of market move).
        entry_prob_cell = f"<td class='num'>{entry_prob_pct}%</td>"
        if current_prob_pct is None:
            current_prob_cell = "<td class='num gray'>—</td>"
        else:
            current_prob_cell = (
                f"<td class='num' title='Market mid for our side right "
                f"now. Compare to Entry prob to see how the market has "
                f"moved.'>{current_prob_pct:.0f}%</td>"
            )
        mtc = b.get("minutes_to_close")
        # Sign / color logic for potential gain — usually positive
        # (winning side pays $1 minus entry minus fees), but very
        # high entry prices on extreme strikes can flip negative.
        pg_sign = "+" if potential_gain >= 0 else "−"
        pg_cls  = "green" if potential_gain >= 0 else "red"
        bot_td = (f"<td>{html.escape(bot_name)}</td>" if show_bot else "")
        # Build the "why was this bet chosen" payload from entry-time
        # snapshot fields recorded on the position. JS reads this from
        # data-criteria on click and populates the shared modal.
        m_yes = b.get("model_yes_prob_at_entry")
        k_yes = b.get("kalshi_yes_prob_at_entry")
        # Selected-side probabilities — YES bet uses model_yes / kalshi_yes
        # directly; NO bet uses 1 - model_yes / 1 - kalshi_yes.
        if m_yes is not None:
            try:
                m_yes_f = float(m_yes)
                model_p = m_yes_f if side == "YES" else (1.0 - m_yes_f)
            except (TypeError, ValueError):
                model_p = None
        else:
            model_p = None
        if k_yes is not None:
            try:
                k_yes_f = float(k_yes)
                kalshi_p = k_yes_f if side == "YES" else (1.0 - k_yes_f)
            except (TypeError, ValueError):
                kalshi_p = None
        else:
            kalshi_p = None
        edge_pts = (
            (model_p - kalshi_p) * 100.0
            if (model_p is not None and kalshi_p is not None) else None
        )
        criteria = {
            "ticker": b.get("ticker"),
            "side": side,
            "entry": entry,
            "contracts": contracts,
            "question": question,
            "bot": bot_name if show_bot else b.get("_bot_name", ""),
            "model_p": model_p,
            "kalshi_p": kalshi_p,
            "edge_pts": edge_pts,
            "entry_ev": b.get("expected_ev_at_entry"),
            "break_even": b.get("break_even_probability"),
            "opened": opened,
        }
        criteria_json = html.escape(json.dumps(
            criteria, separators=(",", ":"), default=str))
        out.append(
            f"<tr><td>{html.escape(opened)}</td>"
            f"{bot_td}"
            f"<td class='mono'>{html.escape(b['ticker'])}</td>"
            f"<td>{html.escape(question)}</td>"
            f"<td class='num'>{contracts}</td>"
            f"<td><span class='badge {badge_cls}'>{side}</span></td>"
            f"{entry_prob_cell}"
            f"{current_prob_cell}"
            f"{entry_cost_cell}"
            f"<td class='num {pg_cls}' title='(100¢ − entry) × contracts "
            f"− entry fee. Entry fee already paid; settlement at 100¢ "
            f"or 0¢ has zero exit fee.'>"
            f"{pg_sign}${abs(potential_gain):.2f}</td>"
            f"<td class='num'>{time_to_close_str(mtc)}</td>"
            f"<td><button type='button' class='criteria-btn' "
            f"title='Why was this bet chosen?' "
            f"data-criteria='{criteria_json}'>i</button></td>"
            f"</tr>"
        )
    out.append("</tbody></table>")


def _render_bet_history_block(out: List[str], history: List[dict],
                               heading: str = "Historical bets — closed",
                               shown_initially: int = 5) -> None:
    """Subsection: closed bets with entry/exit/P&L. Used inline under
    Section 1 (Summary) and Section 5 (Active bet) so each view shows
    the lifetime trade ledger directly under its active-bets table.

    Uses HTML <details>/<summary> so the first ``shown_initially`` rows
    are visible and the rest are collapsible — no JS. Pass an empty
    ``heading`` to render the table without a subhead — useful when the
    enclosing section's title already names the period.
    """
    if heading:
        out.append(f"<h3 class='subhead'>{html.escape(heading)}</h3>")
    if not history:
        out.append("<div class='empty'>No closed bets yet.</div>")
        return

    head = (
        "<table><thead><tr>"
        "<th>Closed</th><th>Bot</th><th>Ticker</th><th>Question</th>"
        "<th>Side</th>"
        "<th class='num'>Entry</th><th class='num'>Exit</th>"
        "<th class='num'>Contracts</th>"
        "<th class='num' title='Model probability for the side we bet on, recorded at entry.'>Model p</th>"
        "<th class='num' title='Net EV per contract at entry: (model_p − entry_price) − half-spread. "
        "Positive = +EV trade.'>Entry EV</th>"
        "<th class='num' title='Underlying value at the moment this bet closed (in the bot’s native units).'>Value at close</th>"
        "<th class='num'>P&amp;L</th>"
        "<th>Outcome</th>"
        "</tr></thead><tbody>"
    )

    def render_row(b):
        closed = (b.get("exited_at") or "")[:19].replace("T", " ")
        side = (b.get("side") or "").upper()
        badge_cls = "badge-yes" if side == "YES" else "badge-no"
        entry = b.get("entry_price_cents")
        exit_c = b.get("exit_price_cents")
        contracts = b.get("contracts", 0) or 0
        pnl = b.get("realized_pnl_cents") or 0
        pnl_cls_ = "green" if pnl > 0 else ("red" if pnl < 0 else "gray")
        outcome = "WON" if pnl > 0 else ("LOST" if pnl < 0 else "FLAT")
        # Value at close — uses the bot's display config so jobless
        # renders "189K" (no decimals) and gas/natgas render "$2.79".
        display = b.get("_display") or {}
        value_at_close = b.get("gas_price_at_close")
        value_str = (fmt_underlying(value_at_close, display)
                     if value_at_close is not None else "—")
        bot_name = b.get("_bot_name", "—")
        # Question — rendered in the bot's native units when display
        # config is attached. Strikes pulled via market_views subquery
        # in fetch_bet_history.
        floor = b.get("floor_strike")
        cap = b.get("cap_strike")
        try:
            strike_low = float(floor) if floor is not None else None
        except (TypeError, ValueError):
            strike_low = None
        try:
            strike_high = float(cap) if cap is not None else None
        except (TypeError, ValueError):
            strike_high = None
        direction = ("between" if (strike_low is not None
                                    and strike_high is not None)
                     else "above")
        question = question_str(direction, strike_low, strike_high,
                                  display=display)
        # Selected-side model prob: YES bet = model_yes_prob; NO bet = 1 - that.
        m_yes = b.get("model_yes_prob_at_entry")
        if m_yes is not None:
            p_sel = float(m_yes) if side == "YES" else (1.0 - float(m_yes))
            mp_str = f"{p_sel*100:.0f}%"
        else:
            mp_str = "—"
        ev = b.get("expected_ev_at_entry")
        if ev is None:
            ev_str = "—"
            ev_cls = "gray"
        else:
            ev_str, ev_cls = (f"${ev:+.3f}", _ev_status(ev)[0])
        return (f"<tr><td>{html.escape(closed)}</td>"
                f"<td>{html.escape(bot_name)}</td>"
                f"<td class='mono'>{html.escape(b['ticker'])}</td>"
                f"<td>{html.escape(question)}</td>"
                f"<td><span class='badge {badge_cls}'>{side}</span></td>"
                f"<td class='num'>{entry}c</td>"
                f"<td class='num'>{cents_or_dash(exit_c)}</td>"
                f"<td class='num'>{contracts}</td>"
                f"<td class='num'>{mp_str}</td>"
                f"<td class='num {ev_cls}'>{ev_str}</td>"
                f"<td class='num'>{value_str}</td>"
                f"<td class='num {pnl_cls_}'>{fmt_signed_cents(pnl)}</td>"
                f"<td class='{pnl_cls_}'>{outcome}</td></tr>")

    out.append(head)
    for b in history[:shown_initially]:
        out.append(render_row(b))
    out.append("</tbody></table>")

    if len(history) > shown_initially:
        out.append(f"<details style='margin-top:8px;'>"
                   f"<summary style='cursor:pointer; padding:10px 0; "
                   f"color:#58a6ff;'>Show {len(history) - shown_initially} more</summary>")
        out.append(head)
        for b in history[shown_initially:]:
            out.append(render_row(b))
        out.append("</tbody></table></details>")


def _render_bot_filter(out: List[str], available_bots: List[dict],
                       current_bot: str,
                       period_key: str = "all") -> None:
    """Bot selector dropdown for the Watchlist tab. Native <select>
    so it's keyboard-friendly and feels like part of the dashboard
    rather than another row of pills. Switching navigates to
    ?bot=<key>&tab=watchlist; the active period is preserved.
    """
    period_qs = (f"&period={html.escape(period_key)}"
                 if period_key and period_key != "all" else "")
    out.append("<div class='bot-filter-bar'>")
    out.append("<label for='bot-select' class='filter-label'>Bot</label>")
    out.append("<select id='bot-select' class='bot-select'>")
    for b in available_bots:
        avail = b.get("available", True)
        suffix = "" if avail else " (no data)"
        sel = " selected" if b["key"] == current_bot else ""
        # data-href carries the target URL; the JS at the bottom of
        # the page wires the <select>'s onchange event to navigate
        # there (same pattern as the prior pill links).
        href = (f"?bot={html.escape(b['key'])}&tab=watchlist{period_qs}")
        out.append(
            f"<option value='{html.escape(href)}'{sel}>"
            f"{html.escape(b['name'])}{html.escape(suffix)}</option>"
        )
    out.append("</select>")
    out.append("</div>")


def _render_bot_unavailable(out: List[str], bot_key: str) -> None:
    out.append("<div class='section'><h2>Bot data unavailable</h2><div class='body'>"
               f"<div class='empty'>The <b>{html.escape(bot_key)}</b> bot is registered "
               f"but has no data on this host yet. Switch to a different bot above, "
               f"or run that bot's service to populate <code>data/sim.db</code>.</div>"
               "</div></div>")


def _fmt_signed_underlying(value: float | None, display: dict) -> str:
    """Like fmt_underlying but with a leading +/- sign — for delta
    values (median_change). Respects the bot's divisor + decimals."""
    if value is None:
        return "—"
    divisor = float(display.get("divisor", 1.0)) or 1.0
    v = float(value) / divisor
    decimals = int(display.get("underlying_decimals", 2))
    unit = display.get("underlying_unit", "")
    pos = display.get("unit_position", "prefix")
    sign = "+" if v >= 0 else "−"
    n = f"{abs(v):,.{decimals}f}"
    if pos == "prefix":
        return f"{sign}{unit}{n}"
    if pos == "suffix":
        return f"{sign}{n}{unit}"
    return f"{sign}{n}"


def _render_current_prediction(out: List[str], model: dict | None,
                                 display: dict | None = None) -> None:
    """Renders the 'Current prediction' card row.

    Lives at the top of the Watchlist section now (per user request) — it
    fits there better than under the Model section because it's the
    immediate context for reading the watchlist rows below it.

    Number formatting follows the bot's display config so unemployment
    shows "189K" / "+1K" instead of "$189000.00" / "+1000.00".
    """
    if not model:
        return
    display = display or {}
    # Subhead removed per user request — the cards label themselves.
    out.append("<div class='subsec'>")
    prob_up = float(model.get("prob_up") or 0)
    change = float(model.get("median_change") or 0)
    q05 = model.get("quantile_05")
    q95 = model.get("quantile_95")
    out.append("<div class='row compact'>")
    out.append(f"<div class='card'><div class='label'>Current price</div>"
               f"<div class='value'>"
               f"{html.escape(fmt_underlying(model.get('current_gas_price'), display))}"
               f"</div></div>")
    out.append(f"<div class='card'><div class='label'>Predicted next week</div>"
               f"<div class='value'>"
               f"{html.escape(fmt_underlying(model.get('median_price'), display))}"
               f"</div></div>")
    out.append(f"<div class='card'><div class='label'>Median change</div>"
               f"<div class='value'>"
               f"{html.escape(_fmt_signed_underlying(change, display))}"
               f"</div></div>")
    out.append(f"<div class='card'><div class='label'>P(price goes up)</div>"
               f"<div class='value'>{prob_up:.0%}</div></div>")
    q05_str = (html.escape(fmt_underlying(q05, display))
               if q05 is not None else "—")
    q95_str = (html.escape(fmt_underlying(q95, display))
               if q95 is not None else "—")
    out.append(f"<div class='card'><div class='label'>Lower 5%</div>"
               f"<div class='value'>{q05_str}</div></div>")
    out.append(f"<div class='card'><div class='label'>Upper 95%</div>"
               f"<div class='value'>{q95_str}</div></div>")
    out.append("</div></div>")


def _ev_status(ev: float | None) -> tuple[str, str]:
    """Return (css_class, label) for an EV value. Drives the red/yellow/
    green pill on every EV-bearing card."""
    if ev is None:
        return "gray", "—"
    if ev >= 0.03:
        return "green", "POSITIVE EV"
    if ev > 0:
        return "yellow", "MARGINAL EV"
    return "red", "NEGATIVE EV"


def _render_watchlist_hero(out: List[str],
                            watchlist: List[dict],
                            model: dict | None,
                            underlying_history: List[dict],
                            display: dict,
                            latest_active: dict | None,
                            kalshi_history: List[dict] | None = None,
                            atm_market: dict | None = None,
                            contract_open_ts: float | None = None,
                            contract_close_ts: float | None = None,
                            event_title: str | None = None) -> None:
    """Kalshi-style hero block: current implied-underlying forecast,
    value delta, time-to-close on the soonest market, and a chart of
    the forecast over the contract life. The hero forecast value
    updates as the user scrubs the line — the JS swaps it for the
    value at the cursor's timestamp, then restores the live current
    when the cursor leaves the chart.
    """
    # Pull current + earliest from the Kalshi-forecast series (the
    # implied-underlying derived from the strike ladder). That's the
    # series the chart plots, so the hero forecast and the chart
    # endpoints line up. Falls back to local snapshots / latest model
    # snapshot only if the Kalshi feed is unavailable.
    current: float | None = None
    earliest_value: float | None = None
    if kalshi_history:
        for r in reversed(kalshi_history):
            v = r.get("value")
            if v is None:
                continue
            try:
                current = float(v)
                break
            except (TypeError, ValueError):
                continue
        for r in kalshi_history:
            v = r.get("value")
            if v is None:
                continue
            try:
                earliest_value = float(v)
                break
            except (TypeError, ValueError):
                continue
    if current is None and model is not None and model.get("current_gas_price") is not None:
        try:
            current = float(model["current_gas_price"])
        except (TypeError, ValueError):
            current = None
    if earliest_value is None and underlying_history:
        for r in underlying_history:
            v = r.get("value")
            if v is None:
                continue
            try:
                earliest_value = float(v)
                break
            except (TypeError, ValueError):
                continue

    # Raw value delta over the visible chart window (e.g. "▼ 9.05K").
    # Replaces the prior percent-change indicator: the user wants to see
    # the actual underlying delta in native units, not a normalized %.
    value_change: float | None = None
    if current is not None and earliest_value is not None:
        value_change = current - earliest_value

    # Total Kalshi volume across the visible watchlist + soonest close.
    vols = [int(r.get("volume") or 0) for r in watchlist
            if r.get("volume") is not None]
    total_volume = sum(vols)
    mtc_values = [float(r.get("minutes_to_close")) for r in watchlist
                  if r.get("minutes_to_close") is not None
                  and float(r.get("minutes_to_close")) > 0]
    soonest_mtc = min(mtc_values) if mtc_values else None

    # Per-bot display formatting + active-strike overlay (if any).
    label = display.get("underlying_label", "Underlying") if display else "Underlying"
    current_str = fmt_underlying(current, display)
    # Format the raw delta in the bot's native units, then strip the
    # leading sign (the arrow already conveys direction).
    if value_change is None:
        change_body = "—"
        change_cls = ""
    else:
        signed = _fmt_signed_underlying(value_change, display)
        # _fmt_signed_underlying emits "+" or "−" as the first char.
        change_body = signed.lstrip("+−-")
        change_cls = "pos" if value_change >= 0 else "neg"

    active_strike = None
    active_side = None
    if latest_active:
        # The positions table doesn't carry strike_low / strike_high
        # directly — those live on market_views. Look them up by ticker.
        # For "between" markets we plot the midpoint; for "above $X" we
        # plot the lower bound.
        wl_row = next(
            (w for w in watchlist
             if w.get("ticker") == latest_active.get("ticker")),
            None,
        )
        if wl_row is not None:
            sl = wl_row.get("strike_low")
            sh = wl_row.get("strike_high")
            if sl is not None and sh is not None:
                try:
                    active_strike = (float(sl) + float(sh)) / 2
                except (TypeError, ValueError):
                    pass
            elif sl is not None:
                try:
                    active_strike = float(sl)
                except (TypeError, ValueError):
                    pass
        active_side = (latest_active.get("side") or "").upper() or None

    # Header: big price + change on the left, time-to-close on the right.
    # Title and volume removed per user request — title is in the chart's
    # underlying ladder anyway and volume duplicates per-row Contracts.
    out.append("<div class='wl-hero'>")
    out.append("<div class='wl-hero-top'>")
    out.append("<div class='wl-hero-stats'>")
    # data-current-text holds the live forecast in display form. The
    # hover JS swaps the price span's text for the value at the cursor
    # while scrubbing, then restores this string on mouseleave.
    out.append(
        f"<div class='wl-hero-price' data-current-text='"
        f"{html.escape(current_str)}'>"
        f"<span class='wl-hero-price-text'>{html.escape(current_str)}</span>"
        f"<span class='wl-hero-price-label'>forecast</span>"
        f"</div>"
    )
    arrow = ""
    if value_change is not None:
        arrow = "▲" if value_change >= 0 else "▼"
    change_display = (change_body if not arrow
                      else f"{arrow} {change_body}")
    # Tag the change indicator with its live text + class so the hover
    # JS can swap to (cursor − earliest) while scrubbing and restore
    # the live current on leave. The class encodes pos/neg coloring.
    out.append(
        f"<div class='wl-hero-change {change_cls}' "
        f"data-current-text='{html.escape(change_display)}' "
        f"data-current-class='{html.escape(change_cls)}'>"
        f"{html.escape(change_display)}"
        f"</div>"
    )
    out.append("</div>")  # /wl-hero-stats
    out.append(f"<div class='wl-hero-mtc'>"
               f"<span class='label'>Closes in</span> "
               f"<span class='value'>{time_left_str(soonest_mtc)}</span>"
               f"</div>")
    out.append("</div>")  # /wl-hero-top

    # Pick a reference strike to color the line against. Active bet's
    # Reference strike for chart coloring: only set when there's an
    # active bet. The dotted strike line + green-above-or-below split
    # only make sense relative to a real position; closest-to-money
    # was noisy without one.
    reference_strike = active_strike
    strike_side = active_side
    strike_is_active = active_strike is not None

    # Chart plots Kalshi's implied-underlying forecast. Empty frame
    # renders when the strike ladder hasn't produced enough data points
    # yet (svg_kalshi_chart handles the <2 case internally).
    out.append(svg_kalshi_chart(
        kalshi_history or [], display,
        reference_strike=reference_strike,
        strike_side=strike_side,
        strike_is_active_bet=strike_is_active,
        contract_open_ts=contract_open_ts,
        contract_close_ts=contract_close_ts,
        total_volume=total_volume,
    ))
    out.append("</div>")


def _render_watchlist(out: List[str], watchlist: List[dict],
                      model: dict | None,
                      underlying_history: List[dict] | None = None,
                      display: dict | None = None,
                      latest_active: dict | None = None,
                      kalshi_history: List[dict] | None = None,
                      atm_market: dict | None = None,
                      contract_open_ts: float | None = None,
                      contract_close_ts: float | None = None,
                      event_title: str | None = None,
                      edge_cfg: dict | None = None,
                      validator_cfg: dict | None = None,
                      risk_caps: dict | None = None,
                      hedge_cfg: dict | None = None,
                      available_bots: List[dict] | None = None,
                      current_bot: str = "",
                      period_key: str = "all") -> None:
    accuracy = float(model["classifier_accuracy"]) if model and model.get("classifier_accuracy") else None
    accuracy_label = (f"{accuracy*100:.0f}%" if accuracy else "untrained")
    out.append(f"<div class='section'><h2>4 · Watchlist — model vs market "
               f"<span class='small gray'>(model historical accuracy {accuracy_label}; "
               f"confidence is scaled by it)</span></h2><div class='body'>")
    # Bot dropdown sits between the watchlist title and the current
    # prediction so the active bot is clearly tied to the section it
    # scopes (per user request).
    if available_bots:
        _render_bot_filter(out, available_bots, current_bot,
                            period_key=period_key)
    # Current-prediction card row (no subtitle — the cards label
    # themselves).
    _render_current_prediction(out, model, display=display)

    # Buy-criteria reference button — rendered as a small circle-i info
    # icon inline with the Active-bet h3 so it sits next to the
    # section title (compact, doesn't take a row of its own). Click
    # opens the same shared modal with the full rules.
    rules_payload = json.dumps({
        "edge": edge_cfg or {},
        "validators": validator_cfg or {},
        "risk": risk_caps or {},
        "hedge": hedge_cfg or {},
    }, separators=(",", ":"), default=str)
    rules_icon_html = (
        "<button type='button' class='criteria-rules-btn' "
        f"data-rules='{html.escape(rules_payload)}' "
        f"title=\"What does this bot need before it'll buy?\">"
        "i</button>"
    )

    # ── This bot's active bet ────────────────────────────────────────
    # Active bet h3 → rules button → bet table (or empty state). The
    # rules button always renders so the rule-set context is one click
    # away even when the bot has no open position right now.
    out.append(
        "<h3 class='subhead' "
        "style='display:flex;align-items:center;gap:8px;'>"
        f"Active bet {rules_icon_html}</h3>"
    )
    if latest_active:
        enriched = dict(latest_active)
        # Strike data from the matching watchlist row (keyed by ticker).
        wl_match = next(
            (w for w in (watchlist or [])
             if w.get("ticker") == latest_active.get("ticker")),
            None,
        )
        if wl_match:
            enriched.setdefault("floor_strike", wl_match.get("strike_low"))
            enriched.setdefault("cap_strike", wl_match.get("strike_high"))
            enriched.setdefault("minutes_to_close",
                                  wl_match.get("minutes_to_close"))
            # mark fallback for bots that don't write position_marks —
            # use the latest watchlist mark so Current renders.
            if enriched.get("mark_yes_ask") is None:
                enriched["mark_yes_ask"] = wl_match.get("yes_ask_cents")
            if enriched.get("mark_no_ask") is None:
                enriched["mark_no_ask"] = wl_match.get("no_ask_cents")
        enriched["_display"] = display or {}
        _render_active_bets_table(out, [enriched],
                                    show_bot=False)

    # ── Hero header + chart (Kalshi-style) ────────────────────────────────
    # Top-line metrics for the underlying the bot tracks: current value,
    # % change vs the start of the chart window, total Kalshi volume
    # across the watchlist, and time-to-close on the soonest market.
    # Chart pulls candlesticks live from Kalshi when configured.
    _render_watchlist_hero(out, watchlist, model,
                           underlying_history or [],
                           display or {}, latest_active,
                           kalshi_history=kalshi_history,
                           atm_market=atm_market,
                           contract_open_ts=contract_open_ts,
                           contract_close_ts=contract_close_ts,
                           event_title=event_title)

    if not watchlist:
        out.append("<div class='empty'>No fully-priced markets right now.</div>")
        out.append("</div></div>")
        return

    # ── Pre-pass: enrich each row with EV/BE numbers, then sort by best
    # EV. Sorting by EV (not by gap or by alphabetical ticker) puts the
    # genuinely-actionable opportunities at the top of the table.
    for v in watchlist:
        ya = v.get("yes_ask_cents")
        na = v.get("no_ask_cents")
        spread = v.get("spread_cents") or 0
        half_spread_d = (spread / 2.0) / 100.0
        p_yes_blend = v.get("model_prob_yes")
        be_yes = (ya / 100.0) if ya is not None else None
        be_no = (na / 100.0) if na is not None else None
        ev_yes = ((p_yes_blend - be_yes) - half_spread_d
                  if p_yes_blend is not None and be_yes is not None else None)
        ev_no = (((1.0 - p_yes_blend) - be_no) - half_spread_d
                 if p_yes_blend is not None and be_no is not None else None)
        # Best side by EV (only among the sides we have prices for).
        candidates = [(s, e) for s, e in (("YES", ev_yes), ("NO", ev_no))
                      if e is not None]
        best_side, best_ev = (None, None)
        if candidates:
            best_side, best_ev = max(candidates, key=lambda x: x[1])
        v["_ev_yes"] = ev_yes
        v["_ev_no"] = ev_no
        v["_be_yes"] = be_yes
        v["_be_no"] = be_no
        v["_best_side"] = best_side
        v["_best_ev"] = best_ev
    # Filter to rows that have at least 1 open contract — markets with
    # zero open interest aren't tradeable and clutter the table.
    watchlist = [r for r in watchlist
                 if (r.get("open_interest") or 0) > 0]
    # Sort by strike ascending — natural order ($4.00, $4.02, $4.04 ...).
    # Falls back to ticker for rows missing a strike (shouldn't happen for
    # KXAAAGASW but defends against partial parses).
    watchlist = sorted(
        watchlist,
        key=lambda r: (r.get("strike_low")
                       if r.get("strike_low") is not None else 9_999.0,
                       r.get("ticker") or ""),
    )

    # Column layout (per user spec): Ticker | Question | Contracts |
    # Kalshi YES + NO grouped | My YES + NO grouped | EV YES + NO
    # grouped | Verdict (rightmost). Chance was redundant with Kalshi
    # YES (same midpoint of the bid/ask); volume and closes-in live in
    # the hero header instead of being repeated per row.
    out.append("<div class='watchlist-scroll'>"
               "<table><thead><tr>"
               "<th>Ticker</th><th>Question</th>"
               "<th class='num' title='Open interest — number of contracts currently held open on this strike.'>Contracts</th>"
               "<th class='num'>Kalshi YES %</th>"
               "<th class='num'>Kalshi NO %</th>"
               "<th class='num'>My YES %</th>"
               "<th class='num'>My NO %</th>"
               "<th class='num' title='Expected value per $1 contract on YES, net of half-spread. Positive = profitable in expectation.'>EV YES</th>"
               "<th class='num' title='Expected value per $1 contract on NO, net of half-spread.'>EV NO</th>"
               "<th>Verdict</th></tr></thead><tbody id='watchlist-tbody'>")
    for v in watchlist:
        ticker = v.get("ticker", "")
        qstr = question_str(v.get("direction", ""), v.get("strike_low"),
                             v.get("strike_high"), display=display)
        ya_c = v.get("yes_ask_cents"); na_c = v.get("no_ask_cents")
        spread_cents = v.get("spread_cents")
        # Volume still drives the "thin volume" row-suspect flag below
        # but the column itself is gone — the hero shows the watchlist
        # total instead.
        volume = v.get("volume")
        oi = v.get("open_interest")
        oi_str = f"{int(oi):,}" if oi is not None else "—"
        # Derive missing side from the other when only one ask is quoted.
        if ya_c is not None:
            kyes_str = f"{ya_c}%"
        elif na_c is not None:
            kyes_str = f"~{100 - na_c}%"
        else:
            kyes_str = "—"
        if na_c is not None:
            kno_str = f"{na_c}%"
        elif ya_c is not None:
            kno_str = f"~{100 - ya_c}%"
        else:
            kno_str = "—"
        p = v.get("model_prob_yes")
        raw_p = v.get("raw_model_prob_yes")
        my_yes_str = f"{int(round(float(p)*100))}%" if p is not None else "—"
        my_no_str = f"{int(round((1-float(p))*100))}%" if p is not None else "—"
        # Tooltip exposes the un-blended raw model probability so a 1pt
        # blended display doesn't hide a 30pt raw disagreement (or
        # vice versa). Lets the user audit "is the bot's actual view
        # justified, or is the blend doing all the work?"
        my_yes_tt = ""
        my_no_tt = ""
        if raw_p is not None and p is not None:
            raw_yes_pct = int(round(float(raw_p) * 100))
            raw_no_pct = int(round((1 - float(raw_p)) * 100))
            blended_yes_pct = int(round(float(p) * 100))
            blended_no_pct = int(round((1 - float(p)) * 100))
            my_yes_tt = (f" title='Raw model: {raw_yes_pct}% · "
                         f"Blended (vs Kalshi, skill-weighted): {blended_yes_pct}%'")
            my_no_tt = (f" title='Raw model: {raw_no_pct}% · "
                        f"Blended: {blended_no_pct}%'")

        # Validation flags — surfaced via row dimming + tooltip only.
        # The Gap column was removed; EV YES + EV NO already convey the
        # same edge information with spread cost baked in.
        flags = []
        if ya_c is None or na_c is None:
            flags.append("one-sided book")
        if spread_cents is not None and spread_cents > 8:
            flags.append("wide spread")
        if p is not None and 0.40 <= p <= 0.60:
            flags.append("low confidence")
        if volume is not None and volume < 50:
            flags.append("thin volume")

        # Color my-yes / my-no cells when one side is meaningfully better
        # than the other (using EV directly, not gap-based heuristics).
        ev_yes_v = v.get("_ev_yes")
        ev_no_v = v.get("_ev_no")
        my_yes_cls = ""
        my_no_cls = ""
        if not flags and ev_yes_v is not None and ev_no_v is not None:
            if ev_yes_v >= 0.03 and ev_yes_v > ev_no_v:
                my_yes_cls = "green"
                my_no_cls = "red"
            elif ev_no_v >= 0.03 and ev_no_v > ev_yes_v:
                my_yes_cls = "red"
                my_no_cls = "green"

        # ── Verdict — driven by EV first, gates second ─────────────────
        # Rules:
        #   TRADE  — best EV positive AND bot_verdict is BUY_*
        #            (both EV and all gates passed)
        #   WATCH  — best EV positive but bot_verdict is WATCH/SKIP
        #            (model likes it, but a gate is blocking — e.g.
        #             thin volume, low confidence, basis-risk zone)
        #   SKIP   — best EV is non-positive (don't trade against EV)
        bot_verdict = v.get("bot_verdict", "SKIP")
        reason = v.get("rejection_reason") or ""
        best_ev_v = v.get("_best_ev")
        best_side_v = v.get("_best_side")
        tt = f" title='{html.escape(reason)}'" if reason else ""
        if best_ev_v is None or best_ev_v <= 0:
            badge = f"<span class='badge badge-skip'{tt}>SKIP</span>"
        elif bot_verdict in ("BUY_YES", "BUY_NO"):
            cls = "badge-yes" if best_side_v == "YES" else "badge-no"
            badge = (f"<span class='badge {cls}'{tt}>"
                     f"BUY {best_side_v}</span>")
        else:
            badge = f"<span class='badge badge-hedge'{tt}>WATCH</span>"

        # Bought rows (the strike the bot currently holds) win over
        # suspect-row dimming — we always want the held position to
        # pop visually, even if its book is currently thin/one-sided.
        is_bought = bool(latest_active and
                         latest_active.get("ticker") == ticker)
        bought_side = ((latest_active.get("side") or "").upper()
                       if is_bought else "")
        classes: List[str] = []
        title_attr = ""
        if is_bought:
            classes.append("row-bought")
            classes.append("bought-yes" if bought_side == "YES"
                           else "bought-no" if bought_side == "NO"
                           else "")
            entry_c = latest_active.get("entry_price_cents")
            contracts = latest_active.get("contracts")
            tip_parts = ["You are holding this strike"]
            if bought_side:
                tip_parts.append(f"on {bought_side}")
            if contracts is not None:
                tip_parts.append(f"({contracts} contracts")
                if entry_c is not None:
                    tip_parts.append(f"@ {entry_c}c)")
                else:
                    tip_parts[-1] = tip_parts[-1] + ")"
            elif entry_c is not None:
                tip_parts.append(f"(entry {entry_c}c)")
            title_attr = (" title='"
                          + html.escape(" ".join(tip_parts)) + "'")
        elif flags:
            classes.append("row-suspect")
            title_attr = (" title='"
                          + html.escape("Suspect: " + ", ".join(flags))
                          + "'")
        row_cls = (f" class='{' '.join(classes)}'" if classes else "") + title_attr

        # Pre-format EV cells (BE YES + Best columns removed — EV cells
        # already convey the same information without the real-estate).
        def _ev_cell(ev: float | None) -> tuple[str, str]:
            if ev is None:
                return "—", "gray"
            cls_, _ = _ev_status(ev)
            return f"${ev:+.3f}", cls_
        ev_yes_str, ev_yes_cls = _ev_cell(ev_yes_v)
        ev_no_str, ev_no_cls = _ev_cell(ev_no_v)

        # data-ticker on the row + data-field on each live cell so the
        # snapshot poller can patch them in place without re-rendering.
        # mtc cell isn't tagged because it doesn't refresh on a 30s
        # cadence (advances naturally with wall clock time).
        tt_esc = html.escape(ticker)
        # Kalshi uses lowercased series tickers in its market URLs. The
        # full market ticker has the form "<SERIES>-<EVENT>-<STRIKE>", so
        # the series is everything before the first hyphen. Linking to
        # the series page lands on the same market group the row is
        # describing; Kalshi resolves it to the active event.
        series_lower = (ticker.split("-", 1)[0] if ticker else "").lower()
        ticker_url = (f"https://kalshi.com/markets/{series_lower}"
                      if series_lower else "")
        ticker_cell = (
            f"<a href='{html.escape(ticker_url)}' target='_blank' "
            f"rel='noopener noreferrer' class='ticker-link'>{tt_esc}</a>"
            if ticker_url else tt_esc
        )
        # The "BOUGHT YES/NO" inline pill was retired — the row's
        # side-colored left bar + colored ticker text already convey
        # the bet at a glance.
        out.append(f"<tr{row_cls} data-ticker='{tt_esc}'>"
                   f"<td class='mono'>{ticker_cell}</td>"
                   f"<td>{html.escape(qstr)}</td>"
                   f"<td class='num' data-field='oi'>{oi_str}</td>"
                   f"<td class='num' data-field='kyes'>{kyes_str}</td>"
                   f"<td class='num' data-field='kno'>{kno_str}</td>"
                   f"<td class='num {my_yes_cls}' data-field='my_yes'{my_yes_tt}>{my_yes_str}</td>"
                   f"<td class='num {my_no_cls}' data-field='my_no'{my_no_tt}>{my_no_str}</td>"
                   f"<td class='num {ev_yes_cls}' data-field='ev_yes'>{ev_yes_str}</td>"
                   f"<td class='num {ev_no_cls}' data-field='ev_no'>{ev_no_str}</td>"
                   f"<td data-field='verdict'>{badge}</td></tr>")
    out.append("</tbody></table></div>")
    out.append("</div></div>")


def _render_contract_rules(out: List[str], watchlist: List[dict],
                           current_bot: str) -> None:
    """Section 7 — actual Kalshi resolution rule, one paragraph.

    All KXAAAGASW contracts share the same template (only the strike
    differs). We pick the first market with a populated rules_primary
    and render that single paragraph verbatim. The strike count and
    range are noted so the user knows it applies across the series.
    """
    out.append("<div class='section'><h2>6 · Kalshi rules — how the market resolves</h2>"
               "<div class='body rules'>")

    # Find a representative rules_primary string.
    paragraph = ""
    for v in watchlist:
        rp = (v.get("rules_primary") or "").strip()
        if rp:
            paragraph = rp
            break

    if not paragraph:
        out.append("<div class='empty'>Rules text not cached yet — "
                   "the next bot tick will populate it from Kalshi.</div>")
        out.append("</div></div>")
        return

    # Strike range across the active contracts in this series.
    strikes = [v.get("strike_low") for v in watchlist
               if v.get("strike_low") is not None]
    if strikes:
        lo, hi = min(strikes), max(strikes)
        sub = (f"This rule applies to <b>{len(strikes)}</b> active contracts "
               f"in this series, with strikes from <b>${lo:.2f}</b> to "
               f"<b>${hi:.2f}</b>.")
    else:
        sub = ""

    out.append(f"<p style='font-size:15px; line-height:1.6;'>"
               f"{html.escape(paragraph)}</p>")
    if sub:
        out.append(f"<p class='small gray'>{sub}</p>")
    out.append("</div></div>")



# --------------------------------------------------------------------------- #
# Server
# --------------------------------------------------------------------------- #

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

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        log.info("%s - %s", self.address_string(), format % args)

    def _resolve_bot(self, query: str) -> dict:
        from urllib.parse import parse_qs
        qs = parse_qs(query)
        requested = qs.get("bot", [None])[0]
        if requested:
            for b in self.bots:
                if b["key"] == requested:
                    return b
        return self.bots[0]

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
                if tab_key not in {"home", "watchlist", "history"}:
                    tab_key = "home"

                # Whale-watcher uses a different page entirely — JSONL
                # source, signal-analysis-style render. Dispatch early so
                # the standard render path stays focused on sim.db bots.
                if bot.get("dashboard_type") == "whale":
                    from . import whale
                    qs = parse_qs(parsed.query)
                    sort_by = qs.get("sort", ["recent"])[0]
                    events = whale.load_events(bot.get("signals_path"))
                    orders = whale.load_orders(bot.get("orders_path"))
                    body = whale.render_page(
                        events=events,
                        orders=orders,
                        available_bots=self.bots,
                        current_bot_key=bot["key"],
                        sort_by=sort_by,
                    )
                    payload = body.encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                    return

                db_path = bot["db_path"]

                # Bot-scoped fetches.
                model = fetch_latest_model(db_path)
                latest_active = fetch_latest_open_position(db_path)
                watchlist = fetch_watchlist(db_path)
                # Will fall back to Kalshi markets below if `watchlist`
                # comes up empty (bot service not writing market_views,
                # or the bot is currently between events). Done after
                # the Kalshi fetch since both share the cache.
                # Local snapshots — kept around as the secondary source
                # for the hero current-value (used as a final fallback
                # if Kalshi creds are missing).
                underlying_history = fetch_underlying_history(
                    db_path, hours=7 * 24, max_points=5000,
                )
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
                if series_ticker:
                    from . import kalshi_client
                    try:
                        (kalshi_history, atm_market, kalshi_markets,
                         contract_open_ts, contract_close_ts,
                         event_title) = (
                            kalshi_client.fetch_underlying_history(
                                series_ticker,
                                period_minutes=chart_period,
                            )
                        )
                    except Exception:  # noqa: BLE001
                        log.exception("kalshi candlestick fetch failed")
                        kalshi_history, atm_market = [], None
                        kalshi_markets, contract_open_ts = [], None
                        contract_close_ts = None
                        event_title = None
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

                # Global cross-bot fetches (these power the Summary section
                # which is identical regardless of which bot is selected).
                global_summary = fetch_global_summary(self.bots,
                                                       period_days=period_days)
                global_active_bets: List[dict] = []
                global_history: List[dict] = []
                # Per-bot models for the Performance tab — one card-row
                # per bot showing accuracy / precision / recall / F1.
                bot_models: List[dict] = []
                for b in self.bots:
                    if not b.get("available"):
                        continue
                    if b.get("dashboard_type") and b["dashboard_type"] != "standard":
                        continue
                    for ab in fetch_active_bets_with_marks(b["db_path"]):
                        ab["_bot_name"] = b["name"]
                        # Attach the bot's display config so the
                        # question column can be formatted in the bot's
                        # native units (K claims vs $ vs ...).
                        ab["_display"] = b.get("display") or {}
                        global_active_bets.append(ab)
                    for h in fetch_bet_history(b["db_path"], limit=50):
                        h["_bot_name"] = b["name"]
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
                # Period-filter the history so the History tab agrees
                # with the rest of the period-aware UI. None → keep all.
                if period_days is not None:
                    cutoff_ts = (datetime.now(timezone.utc).timestamp()
                                  - period_days * 86400)
                    def _within(h):
                        ex = h.get("exited_at") or ""
                        try:
                            t = datetime.fromisoformat(
                                ex.replace("Z", "+00:00")
                            ).timestamp()
                        except (TypeError, ValueError):
                            try:
                                t = datetime.strptime(
                                    ex[:19], "%Y-%m-%d %H:%M:%S"
                                ).replace(tzinfo=timezone.utc).timestamp()
                            except (TypeError, ValueError):
                                return False
                        return t >= cutoff_ts
                    global_history = [h for h in global_history if _within(h)]

                # Bot-scoped closed positions — used in Section 5 underneath
                # the active-bet table per request.
                bot_closed_positions = fetch_bet_history(db_path, limit=100)

                body = render_page(
                    model=model,
                    global_summary=global_summary,
                    global_active_bets=global_active_bets,
                    global_history=global_history,
                    latest_active=latest_active,
                    bot_closed_positions=bot_closed_positions,
                    watchlist=watchlist,
                    underlying_history=underlying_history,
                    display=bot.get("display") or {},
                    kalshi_history=kalshi_history,
                    atm_market=atm_market,
                    contract_open_ts=contract_open_ts,
                    contract_close_ts=contract_close_ts,
                    event_title=event_title,
                    risk_caps=self.risk_caps,
                    edge_cfg=self.edge_cfg,
                    validator_cfg=self.validator_cfg,
                    hedge_cfg=self.hedge_cfg,
                    available_bots=self.bots,
                    current_bot=bot["key"],
                    period_key=period_key,
                    tab_key=tab_key,
                    bot_models=bot_models,
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
                if bot.get("dashboard_type") == "whale":
                    # Whale page uses meta-refresh, not the JS poller.
                    # Return a minimal stub so any client polling this
                    # endpoint gets a clean 200.
                    payload_dict = {"bot": bot["key"], "type": "whale"}
                else:
                    db_path = bot["db_path"]
                    payload_dict = build_snapshot(db_path, self.bots,
                                                   self.edge_cfg,
                                                   period_days=snap_period_days)
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
        elif self.path == "/healthz":
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"ok\n")
        else:
            self.send_error(404)


def serve(host: str, port: int, bots: List[dict], risk_caps: dict,
          edge_cfg: dict, validator_cfg: dict, hedge_cfg: dict) -> None:
    Handler.bots = bots
    Handler.risk_caps = risk_caps
    Handler.edge_cfg = edge_cfg
    Handler.validator_cfg = validator_cfg
    Handler.hedge_cfg = hedge_cfg
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
        "min_edge_yes": cfg.edge.min_edge_yes,
        "min_edge_no": cfg.edge.min_edge_no,
        "min_model_confidence": cfg.edge.min_model_confidence,
        "min_confidence": cfg.edge.min_confidence,
        "min_model_accuracy": cfg.edge.min_model_accuracy,
        "min_ev_per_contract": cfg.edge.min_ev_per_contract,
        "min_prob_edge_over_breakeven": cfg.edge.min_prob_edge_over_breakeven,
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

    # Bot registry comes from the dashboard YAML. Each entry's "available"
    # flag reflects whether the bot's sim.db exists on disk — selecting an
    # unavailable bot in the dropdown shows a friendly stub.
    bots: list[dict] = []
    for b in cfg.bots:
        # For whale-type bots `db_path` points at a JSONL (signal_tracking),
        # so the same `available = path exists` check works.
        bots.append({
            "key": b.key,
            "name": b.name,
            "db_path": b.db_path,
            "decisions_path": b.decisions_path,
            "dashboard_type": b.dashboard_type,
            "signals_path": b.signals_path,
            "orders_path": b.orders_path,
            "series_ticker": b.series_ticker,
            "display": {
                "underlying_label": b.display.underlying_label,
                "underlying_unit": b.display.underlying_unit,
                "underlying_decimals": b.display.underlying_decimals,
                "unit_position": b.display.unit_position,
                "divisor": b.display.divisor,
                "chart_period_minutes": b.display.chart_period_minutes,
            },
            "available": Path(b.db_path).exists(),
        })

    host = args.host or cfg.host
    port = args.port or cfg.port
    serve(host, port, bots, risk_caps, edge_cfg, validator_cfg, hedge_cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
