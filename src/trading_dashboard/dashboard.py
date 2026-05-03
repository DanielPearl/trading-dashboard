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


def fetch_open_positions(db_path: str) -> List[dict]:
    return _safe_query(
        db_path,
        "SELECT p.*, m.yes_ask_cents AS mark_yes_ask, m.no_ask_cents AS mark_no_ask, "
        "m.yes_bid_cents AS mark_yes_bid, m.mid_cents AS mark_mid, "
        "m.spread_cents AS mark_spread, m.updated_at AS mark_updated_at "
        "FROM positions p LEFT JOIN position_marks m ON p.id = m.position_id "
        "WHERE p.status = 'open' ORDER BY p.opened_at DESC")


def fetch_closed_positions(db_path: str, limit: int = 100) -> List[dict]:
    return _safe_query(
        db_path,
        "SELECT * FROM positions WHERE status = 'closed' "
        "ORDER BY exited_at DESC LIMIT ?", (limit,))


def fetch_summary(db_path: str) -> dict:
    """Lifetime + recent stats used by the Summary section."""
    empty = {
        "total_bets": 0, "open_count": 0, "exposure_cents": 0,
        "closed_count": 0, "realized_pnl_cents": 0,
        "wins_lifetime": 0, "losses_lifetime": 0,
        "avg_win_cents": 0, "avg_loss_cents": 0,
        "bets_today": 0, "this_week_pnl_cents": 0,
        "biggest_win_cents": 0, "biggest_loss_cents": 0,
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


def fetch_recent_bets(db_path: str, limit: int = 25) -> List[dict]:
    """Most recent bets (open + closed)."""
    return _safe_query(
        db_path,
        "SELECT id, ticker, side, entry_price_cents, contracts, opened_at, "
        "       status, exit_price_cents, realized_pnl_cents, exited_at "
        "FROM positions ORDER BY opened_at DESC LIMIT ?", (limit,))


def fetch_active_bets_with_marks(db_path: str) -> List[dict]:
    """Open positions joined with their latest mark + latest mtc."""
    return _safe_query(
        db_path,
        "SELECT p.*, "
        "       m.yes_ask_cents AS mark_yes_ask, "
        "       m.no_ask_cents AS mark_no_ask, "
        "       (SELECT mv.minutes_to_close FROM market_views mv "
        "          WHERE mv.ticker = p.ticker "
        "          ORDER BY mv.id DESC LIMIT 1) AS minutes_to_close "
        "FROM positions p LEFT JOIN position_marks m ON p.id = m.position_id "
        "WHERE p.status = 'open' ORDER BY p.opened_at DESC")


def fetch_bet_history(db_path: str, limit: int = 100) -> List[dict]:
    """Closed positions only — for the Bet History section.

    Tolerates schema drift across bots. ``gas_price_at_close`` only exists
    on the gas-prices simulator schema; for other bots (e.g. whale-watcher)
    we still want their closed bets to appear in the cross-bot Summary,
    just with an empty Gas-at-close cell.
    """
    if not Path(db_path).exists():
        return []
    base_cols = ("id, ticker, side, entry_price_cents, exit_price_cents, "
                 "contracts, realized_pnl_cents, opened_at, exited_at")
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
            extras = [c_ for c_ in (
                "gas_price_at_close",
                "model_yes_prob_at_entry",
                "kalshi_yes_prob_at_entry",
                "break_even_probability",
                "expected_ev_at_entry",
                "error_type",
            ) if c_ in cols]
            select_cols = base_cols + (", " + ", ".join(extras) if extras else "")
            rows = c.execute(
                f"SELECT {select_cols} FROM positions WHERE status='closed' "
                f"ORDER BY exited_at DESC LIMIT ?", (limit,),
            ).fetchall()
    except (sqlite3.OperationalError, sqlite3.DatabaseError):
        return []
    return [dict(r) for r in rows]


def fetch_global_summary(bots: List[dict]) -> dict:
    """Cross-bot rollup. The summary card row at the top of the dashboard
    does NOT change when the user switches bots in the filter — these are
    totals across every registered bot.

    Best-bot strategy: the bot with the highest realized P&L; ties go to
    the one with the most lifetime bets (more proof of edge).
    """
    rollup = {
        "total_bets": 0,
        "active_bets": 0,
        "net_pnl_cents": 0,
        "weekly_pnl_cents": 0,
        "wins": 0,
        "losses": 0,
        "weekly_wins": 0,
        "weekly_losses": 0,
        "bets_today": 0,
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
        s = fetch_summary(b["db_path"])
        rollup["total_bets"] += s.get("total_bets", 0)
        rollup["active_bets"] += s.get("open_count", 0)
        rollup["net_pnl_cents"] += s.get("realized_pnl_cents", 0)
        rollup["weekly_pnl_cents"] += s.get("this_week_pnl_cents", 0)
        rollup["wins"] += s.get("wins_lifetime", 0)
        rollup["losses"] += s.get("losses_lifetime", 0)
        rollup["weekly_wins"] += s.get("this_week_wins", 0)
        rollup["weekly_losses"] += s.get("this_week_losses", 0)
        rollup["bets_today"] += s.get("bets_today", 0)
        rollup["per_bot"].append((b["name"], s))
        # Track the best one by realized P&L; ties broken by total_bets.
        better = (
            s.get("realized_pnl_cents", 0) > rollup["best_bot_pnl_cents"]
            or (s.get("realized_pnl_cents", 0) == rollup["best_bot_pnl_cents"]
                and s.get("total_bets", 0) > 0
                and rollup["best_bot_name"] == "—")
        )
        if better:
            rollup["best_bot_name"] = b["name"]
            rollup["best_bot_pnl_cents"] = s.get("realized_pnl_cents", 0)
    total_closed = rollup["wins"] + rollup["losses"]
    # Plain win percent: wins / total closed. 0–100% scale, no losses term.
    rollup["win_pct"] = (
        rollup["wins"] / total_closed if total_closed else 0.0
    )
    weekly_closed = rollup["weekly_wins"] + rollup["weekly_losses"]
    rollup["weekly_win_pct"] = (
        rollup["weekly_wins"] / weekly_closed if weekly_closed else 0.0
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

    Returns rows oldest-first as `[{"captured_at": "...", "value": float}, ...]`.
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
        return [dict(r) for r in rows]
    except (sqlite3.OperationalError, sqlite3.DatabaseError):
        return []


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
                    edge_cfg: dict) -> dict:
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
    summary = fetch_global_summary(bots)
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

    return {
        "summary": {
            "total_bets": summary.get("total_bets"),
            "active_bets": summary.get("active_bets"),
            "net_pnl_cents": summary.get("net_pnl_cents"),
            "weekly_pnl_cents": summary.get("weekly_pnl_cents"),
            "win_pct": summary.get("win_pct"),
            "weekly_win_pct": summary.get("weekly_win_pct"),
            "wins": summary.get("wins"),
            "losses": summary.get("losses"),
            "weekly_wins": summary.get("weekly_wins"),
            "weekly_losses": summary.get("weekly_losses"),
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


def _smooth_values(values: List[float], window: int = 3) -> List[float]:
    """Centered moving average for the chart polyline.

    Smooths just the visual line — the data-points payload used by the
    hover tooltip stays on the raw values so hovering still surfaces the
    actual underlying. Window=3 is intentionally small: enough to take
    the edge off jitter, not enough to lag perceptibly.
    """
    n = len(values)
    if n < 3 or window < 2:
        return list(values)
    half = window // 2
    out: List[float] = []
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        seg = values[lo:hi]
        out.append(sum(seg) / len(seg))
    return out


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
        return ("<div class='empty' style='padding:24px'>"
                "Live Kalshi chart is loading… (or the strike ladder "
                "doesn't have enough trade history yet to derive an "
                "implied underlying price). Refreshes every 60 seconds."
                "</div>")

    # Y-axis labels go on the right edge (matches Kalshi's market page),
    # so reserve the right padding instead of the left. Bottom padding
    # leaves room for the date-tick row.
    pad_l, pad_r, pad_t, pad_b = 12, 64, 14, 30
    inner_w = width - pad_l - pad_r
    inner_h = height - pad_t - pad_b
    n = len(pts_in)
    # X-axis spans contract open → NOW. Each chart shows only the
    # current event's lifetime, no multi-event stitching.
    now_ts = time.time()
    t_max = max(now_ts, pts_in[-1][0])
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
    # YES position → segments above the dotted line are green;
    # NO  position → segments below the dotted line are green.
    # (The closest-to-money strike highlight was retired — it's noisy
    # without a real position to anchor it to.)
    if strike_is_active_bet and strike_in_range and reference_strike is not None:
        ys = y_at(float(reference_strike))
        line_color = above_color if side != "NO" else below_color
        out.append(f"<line x1='{pad_l}' y1='{ys}' x2='{width-pad_r}' y2='{ys}' "
                   f"stroke='{line_color}' stroke-width='1.5' "
                   f"stroke-dasharray='4,4' opacity='0.95'/>")
        label_strike = fmt_underlying(float(reference_strike), display)
        label = f"Above {label_strike}"
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


def svg_underlying_chart(history: List[dict], current_value: float | None,
                          display: dict, reference_strike: float | None = None,
                          strike_side: str | None = None,
                          strike_is_active_bet: bool = False,
                          width: int = 760, height: int = 220) -> str:
    """Watchlist hero chart: underlying commodity / metric value over time,
    rendered in the same style Kalshi uses on its market page.

    Kalshi colors the price line GREEN where it sits above the selected
    strike and white where it sits below. We mirror that: walk the
    polyline, split it at strike crossings, and render each run in the
    appropriate color.

    `reference_strike` controls which strike is treated as the dividing
    line. The watchlist hero passes (a) the active position's strike if
    any, else (b) the closest-to-money strike from the watchlist.
    `strike_side` decides which direction is "winning":
       YES → green above; NO → green below; None → green above (default).
    `strike_is_active_bet` makes the strike line bolder + relabels it.
    """
    pts_in: List[Tuple[str, float]] = []
    for r in history:
        v = r.get("value")
        ts = r.get("captured_at")
        if v is None or ts is None:
            continue
        try:
            pts_in.append((ts, float(v)))
        except (TypeError, ValueError):
            continue
    if len(pts_in) < 2:
        return ("<div class='empty' style='padding:24px'>"
                "Not enough underlying-price history yet (need 2+ snapshots). "
                "Once the bot runs for a few ticks this chart populates."
                "</div>")

    pad_l, pad_r, pad_t, pad_b = 56, 16, 12, 28
    inner_w = width - pad_l - pad_r
    inner_h = height - pad_t - pad_b
    n = len(pts_in)

    # Extend the y-range to include the strike, so an in-the-money strike
    # that sits outside the recent price band still appears on screen.
    values_for_range = [v for _, v in pts_in]
    if reference_strike is not None:
        values_for_range.append(float(reference_strike))
    vmin = min(values_for_range)
    vmax = max(values_for_range)
    if vmax == vmin:
        vmax = vmin + 1.0
    span = vmax - vmin
    pad = span * 0.10
    y_lo = vmin - pad
    y_hi = vmax + pad

    def x_at(i: int) -> float:
        return pad_l + (i / max(1, n - 1)) * inner_w

    def y_at(v: float) -> float:
        return pad_t + (1.0 - (v - y_lo) / (y_hi - y_lo)) * inner_h

    out: List[str] = [
        f"<svg width='100%' height='{height}' viewBox='0 0 {width} {height}' "
        f"preserveAspectRatio='none' style='display:block'>"
    ]

    # Y gridlines: 4 evenly spaced ticks across the visible range.
    for i in range(5):
        v = y_lo + (i / 4.0) * (y_hi - y_lo)
        y = y_at(v)
        out.append(f"<line x1='{pad_l}' y1='{y}' x2='{width-pad_r}' y2='{y}' "
                   f"stroke='#1f2530' stroke-width='1'/>")
        out.append(f"<text x='{pad_l-6}' y='{y+4}' fill='#8b949e' font-size='10' "
                   f"text-anchor='end'>{fmt_underlying(v, display)}</text>")

    # ── Price line, split at strike crossings ──────────────────────────
    side = (strike_side or "").upper()
    if side == "NO":
        above_color, below_color = "#c9d1d9", "#3fb950"
    else:  # YES, or no active side → default green-above
        above_color, below_color = "#3fb950", "#c9d1d9"

    if reference_strike is None:
        # No strike reference → single white line.
        pts = [(x_at(i), y_at(v)) for i, (_, v) in enumerate(pts_in)]
        path = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        out.append(f"<polyline points='{path}' stroke='#c9d1d9' "
                   f"stroke-width='2' fill='none'/>")
    else:
        strike = float(reference_strike)
        # Walk the polyline; whenever a segment crosses the strike,
        # interpolate the crossing point and split the run there.
        runs: List[Tuple[bool, List[Tuple[float, float]]]] = []
        # Each run is (is_above, [(x_px, y_px), ...]).
        cur_above = pts_in[0][1] >= strike
        cur_run: List[Tuple[float, float]] = [(x_at(0), y_at(pts_in[0][1]))]
        for i in range(1, n):
            v_prev = pts_in[i - 1][1]
            v_curr = pts_in[i][1]
            new_above = v_curr >= strike
            if new_above == cur_above:
                cur_run.append((x_at(i), y_at(v_curr)))
                continue
            # Crossing — split at the strike line.
            denom = v_curr - v_prev
            t = (strike - v_prev) / denom if denom != 0 else 0.5
            t = max(0.0, min(1.0, t))
            cross_x = x_at(i - 1) + t * (x_at(i) - x_at(i - 1))
            cross_y = y_at(strike)
            cur_run.append((cross_x, cross_y))
            runs.append((cur_above, cur_run))
            cur_run = [(cross_x, cross_y), (x_at(i), y_at(v_curr))]
            cur_above = new_above
        runs.append((cur_above, cur_run))

        for is_above, run in runs:
            if len(run) < 2:
                continue
            color = above_color if is_above else below_color
            pts_str = " ".join(f"{x:.1f},{y:.1f}" for x, y in run)
            out.append(f"<polyline points='{pts_str}' stroke='{color}' "
                       f"stroke-width='2' fill='none'/>")

    # ── Strike line ────────────────────────────────────────────────────
    if reference_strike is not None:
        ys = y_at(float(reference_strike))
        # Dimmer for the inferred (closest-to-money) strike, brighter +
        # slightly thicker for the strike of an actual active bet.
        line_color = above_color if side != "NO" else below_color
        stroke_w = 1.8 if strike_is_active_bet else 1.2
        opacity = 0.95 if strike_is_active_bet else 0.55
        out.append(f"<line x1='{pad_l}' y1='{ys}' x2='{width-pad_r}' y2='{ys}' "
                   f"stroke='{line_color}' stroke-width='{stroke_w}' "
                   f"opacity='{opacity}'/>")
        label_strike = fmt_underlying(float(reference_strike), display)
        if strike_is_active_bet:
            label = f"Above {label_strike}"
        else:
            label = f"Above {label_strike}"
        out.append(f"<text x='{width-pad_r-6}' y='{ys-6}' fill='{line_color}' "
                   f"font-size='11' text-anchor='end' opacity='{opacity}'>"
                   f"{html.escape(label)}</text>")

    # X axis: timestamp at each end.
    first_ts = pts_in[0][0][:16].replace("T", " ")
    last_ts = pts_in[-1][0][:16].replace("T", " ")
    out.append(f"<text x='{pad_l}' y='{height-8}' fill='#8b949e' font-size='10'>"
               f"{html.escape(first_ts)}</text>")
    out.append(f"<text x='{width-pad_r}' y='{height-8}' fill='#8b949e' font-size='10' "
               f"text-anchor='end'>{html.escape(last_ts)} UTC</text>")
    out.append("</svg>")
    return "".join(out)


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
.bot-filter-bar { display: flex; align-items: center; gap: 8px;
    padding: 4px 0 18px 0; margin-bottom: 8px; flex-wrap: wrap;
    border-bottom: 1px solid #21262d; margin-top: -8px; }
.bot-filter-bar .filter-label {
    color: #8b949e; font-size: 11px; text-transform: uppercase;
    letter-spacing: 0.06em; font-weight: 600; margin-right: 4px;
}
.filter-pill { background: #21262d; color: #c9d1d9; text-decoration: none;
    padding: 6px 14px; border-radius: 999px; font-size: 13px;
    border: 1px solid #30363d; transition: background 120ms, border-color 120ms;
    line-height: 1.4; }
.filter-pill:hover { background: #2d333b; border-color: #40464d; }
.filter-pill-active { background: #1f6feb; color: #fff; border-color: #1f6feb; }
.filter-pill-active:hover { background: #1f6feb; border-color: #1f6feb; }
.filter-pill-disabled { color: #6e7681; cursor: not-allowed; opacity: 0.7; }
.filter-pill-disabled:hover { background: #21262d; border-color: #30363d; }
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
tr.row-bought.bought-yes td:first-child {
    border-left: 3px solid #3fb950;
    background: rgba(63, 185, 80, 0.12);
}
tr.row-bought.bought-no td:first-child {
    border-left: 3px solid #f85149;
    background: rgba(248, 81, 73, 0.12);
}
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


def _bot_label(bot_key: str, available_bots: List[dict]) -> str:
    for b in available_bots:
        if b["key"] == bot_key:
            return b["name"]
    return bot_key


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

    # SECTION 1 — Global summary (cards + active bets + history). Identical
    # regardless of which bot is selected in the filter below.
    _render_summary(out, global_summary, global_active_bets, global_history)

    # SECTION 2 — Bot filter.
    _render_bot_filter(out, available_bots, current_bot)

    # If the selected bot doesn't have a populated DB, stop here.
    if (not watchlist and not latest_active
            and not [b for b in available_bots if b["key"] == current_bot and b.get("available")]):
        _render_bot_unavailable(out, current_bot)
        out.append("</body></html>")
        return "".join(out)

    # SECTION 2 — Model (strength + current prediction).
    _render_model_section(out, model)

    # SECTION 3 — Active bet (just the latest one) + this bot's closed history.
    _render_active_bet(out, latest_active, watchlist, bot_closed_positions)

    # SECTION 4 — Watchlist + the bot's full buy-criteria table directly
    # underneath it (so verdicts and rules live side-by-side). The hero
    # header at the top of this section shows the underlying value and
    # an SVG chart with the active-bet strike overlaid (if any).
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
                      hedge_cfg=hedge_cfg)

    # SECTION 5 — Diagnostics (Translation, Calibration, TTC buckets,
    # Quality Score, Market Flow). Driven by closed-bet history; small
    # sample sizes are surfaced honestly instead of hidden.
    _render_diagnostics(out, bot_closed_positions)

    # SECTION 6 — Kalshi rules per contract.
    _render_contract_rules(out, watchlist, current_bot)

    # Live-update JS: polls /api/snapshot every 5s and patches summary
    # cards + watchlist cells in place. No page reload, no scroll loss.
    out.append(_live_update_script(current_bot))
    out.append("</body></html>")
    return "".join(out)


def _live_update_script(current_bot: str) -> str:
    """Self-contained JS block that fetches /api/snapshot every 5s
    and patches DOM cells with new values. Highlights changed cells
    briefly so updates are visible.
    """
    bot_param = html.escape(current_bot)
    return f"""<script>
(function () {{
  const BOT = "{bot_param}";
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
    const s = snap.summary || {{}};
    const wins = s.wins || 0, losses = s.losses || 0;
    const wWins = s.weekly_wins || 0, wLosses = s.weekly_losses || 0;
    patch("card-total-bets", String(s.total_bets ?? 0));
    patch("card-active-bets", String(s.active_bets ?? 0));
    patch("card-net-pnl", fmtSignedCents(s.net_pnl_cents),
          (s.net_pnl_cents > 0) ? "green"
            : (s.net_pnl_cents < 0 ? "red" : "gray"));
    patch("card-win-pct", fmtPct(s.win_pct, (wins+losses) > 0),
          (s.win_pct > 0.5) ? "green"
            : ((wins+losses) > 0 && s.win_pct < 0.5 ? "red" : "gray"));
    patch("card-weekly-pnl", fmtSignedCents(s.weekly_pnl_cents),
          (s.weekly_pnl_cents > 0) ? "green"
            : (s.weekly_pnl_cents < 0 ? "red" : "gray"));
    patch("card-weekly-win-pct", fmtPct(s.weekly_win_pct, (wWins+wLosses) > 0),
          (s.weekly_win_pct > 0.5) ? "green"
            : ((wWins+wLosses) > 0 && s.weekly_win_pct < 0.5 ? "red" : "gray"));

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
    fetch("/api/snapshot?bot=" + encodeURIComponent(BOT),
          {{cache: "no-store"}})
      .then(function (r) {{ return r.ok ? r.json() : null; }})
      .then(function (snap) {{ if (snap) applySnapshot(snap); }})
      .catch(function () {{ /* swallow — try again next tick */ }});
  }}

  // Initial fetch on load + recurring poll.
  poll();
  setInterval(poll, POLL_MS);

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
    // tooltip. Snaps to the nearest recorded data point; if the cursor
    // is more than half a typical interval away from any recorded
    // point, returns null so the tooltip hides the value line. The
    // user wants the popup to show only values that were actually
    // captured — no interpolation between data points.
    let points = [];
    let fmt = {{ divisor: 1.0, decimals: 2, unit: "", unit_position: "prefix" }};
    try {{ points = JSON.parse(wrap.dataset.points || "[]"); }} catch (e) {{}}
    try {{ fmt = Object.assign(fmt, JSON.parse(wrap.dataset.fmt || "{{}}")); }}
    catch (e) {{}}

    // Median spacing between recorded points. Used as the "is the
    // cursor near a recorded point" yardstick — if the closest point
    // is more than ~60% of the median spacing away, treat it as a gap
    // and hide the value (the user explicitly asked: don't show a value
    // for time ranges where no value was captured).
    let medianGapSec = 0;
    if (points.length > 1) {{
      const gaps = [];
      for (let i = 1; i < points.length; i++) {{
        gaps.push(points[i][0] - points[i - 1][0]);
      }}
      gaps.sort(function (a, b) {{ return a - b; }});
      medianGapSec = gaps[Math.floor(gaps.length / 2)] || 0;
    }}
    // Tolerance: closest point must be within this many seconds of
    // the cursor's timestamp to count as "captured at that time".
    // Floor so daily series get ~14h tolerance (roomy enough to cover
    // the whole day, tight enough to flag a fully-missing day).
    const tolSec = Math.max(60, medianGapSec * 0.6);

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

    // Snap to the nearest recorded point; return null if no point is
    // within the tolerance window. Returns {{ts, value}} so the popup
    // can also stamp the actual recorded timestamp (not the cursor's
    // hovered position) — that's what the user is reading off the line.
    function nearestPoint(ts) {{
      if (!points.length) return null;
      let lo = 0, hi = points.length - 1;
      if (ts <= points[lo][0]) {{
        return Math.abs(points[lo][0] - ts) <= tolSec
          ? {{ ts: points[lo][0], value: points[lo][1] }} : null;
      }}
      if (ts >= points[hi][0]) {{
        return Math.abs(points[hi][0] - ts) <= tolSec
          ? {{ ts: points[hi][0], value: points[hi][1] }} : null;
      }}
      while (hi - lo > 1) {{
        const mid = (lo + hi) >> 1;
        if (points[mid][0] <= ts) lo = mid; else hi = mid;
      }}
      const dLo = Math.abs(ts - points[lo][0]);
      const dHi = Math.abs(ts - points[hi][0]);
      const closer = dLo <= dHi ? points[lo] : points[hi];
      const dist = Math.min(dLo, dHi);
      if (dist > tolSec) return null;
      return {{ ts: closer[0], value: closer[1] }};
    }}

    svg.addEventListener("mousemove", function (e) {{
      const rect = svg.getBoundingClientRect();
      // Cursor's x in viewBox space (the SVG scales to the wrap's width).
      const x = (e.clientX - rect.left) * vbW / rect.width;
      if (x < padL || x > padL + innerW) {{
        cursor.setAttribute("opacity", "0");
        dimRect.setAttribute("opacity", "0");
        tip.hidden = true;
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
      // When a recorded point is in range, stamp the popup with the
      // recorded timestamp + value; otherwise show only the cursor's
      // date so the user knows where they are without implying a value
      // was captured.
      if (np !== null) {{
        tip.innerHTML =
          "<div class='wl-chart-tip-time'>" + fmtTs(np.ts) + "</div>"
          + "<div class='wl-chart-tip-value'>" + fmtValue(np.value) + "</div>";
      }} else {{
        tip.innerHTML =
          "<div class='wl-chart-tip-time'>" + fmtTs(cursorTs) + "</div>";
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
      tip.hidden = true;
    }});
  }});
}})();
</script>"""


# --------------------------------------------------------------------------- #
# Section helpers
# --------------------------------------------------------------------------- #

def _render_summary(out: List[str], rollup: dict, active_bets: List[dict],
                    history: List[dict]) -> None:
    """Section 1 — global cross-bot summary. Does NOT change when the
    user switches bots in the filter. Active bets table on top, full
    closed-bet history collapsed underneath (per request)."""
    net = rollup["net_pnl_cents"]
    pnl_cls = "green" if net > 0 else ("red" if net < 0 else "gray")
    weekly = rollup["weekly_pnl_cents"]
    weekly_cls = "green" if weekly > 0 else ("red" if weekly < 0 else "gray")
    # Plain win percent: wins / closed. 0-100% scale. Above 50% = green
    # (the bot is winning more than it loses), below 50% = red.
    win_pct = rollup["win_pct"]
    has_closed = (rollup["wins"] + rollup["losses"]) > 0
    win_cls = ("green" if win_pct > 0.5
               else ("red" if has_closed and win_pct < 0.5 else "gray"))
    win_pct_str = f"{win_pct*100:.0f}%" if has_closed else "—"
    wwin_pct = rollup["weekly_win_pct"]
    has_weekly_closed = (rollup["weekly_wins"] + rollup["weekly_losses"]) > 0
    wwin_cls = ("green" if wwin_pct > 0.5
                else ("red" if has_weekly_closed and wwin_pct < 0.5 else "gray"))
    wwin_pct_str = f"{wwin_pct*100:.0f}%" if has_weekly_closed else "—"

    out.append("<div class='section'><h2>1 · Summary — across all bots</h2>"
               "<div class='body summary-body'>")
    out.append("<div class='small' style='margin-bottom:14px;'>"
               "Lifetime totals across every registered bot. This panel "
               "does not change when you switch bots in the filter.</div>")
    # Single-value cards only — no sub-labels. Each shows just the headline
    # number; the value font is sized up to fill the freed space.
    out.append("<div class='row'>")
    out.append(f"<div class='card'><div class='label'>Total bets made</div>"
               f"<div class='value' id='card-total-bets'>{rollup['total_bets']}</div></div>")
    out.append(f"<div class='card'><div class='label'>Active bets</div>"
               f"<div class='value' id='card-active-bets'>{rollup['active_bets']}</div></div>")
    out.append(f"<div class='card'><div class='label'>Total net gain / loss</div>"
               f"<div class='value {pnl_cls}' id='card-net-pnl'>{fmt_signed_cents(net)}</div></div>")
    out.append(f"<div class='card'><div class='label' "
               f"title='Lifetime wins divided by closed bets. 0-100%; "
               f"above 50% means winning more than losing.'>"
               f"Total win percent</div>"
               f"<div class='value {win_cls}' id='card-win-pct'>{win_pct_str}</div></div>")
    # Weekly sits second-from-right; both P&L cards stay green/red.
    out.append(f"<div class='card'><div class='label' "
               f"title='Realized P&amp;L from bets closed in the last "
               f"7 days.'>Weekly net gain / loss</div>"
               f"<div class='value {weekly_cls}' id='card-weekly-pnl'>{fmt_signed_cents(weekly)}</div></div>")
    out.append(f"<div class='card'><div class='label' "
               f"title='Wins divided by closed bets in the last 7 days. "
               f"0-100%, same scale as lifetime win percent.'>"
               f"Weekly win percent</div>"
               f"<div class='value {wwin_cls}' id='card-weekly-win-pct'>{wwin_pct_str}</div></div>")
    out.append("</div>")

    # Active bets list. Same table used in Section 5 below for consistency.
    out.append("<h3 class='subhead'>Active bets — currently open</h3>")
    _render_active_bets_table(out, active_bets, empty_msg="No active bets right now.")

    # Historical (closed) bets directly under active bets, per user request.
    _render_bet_history_block(out, history, heading="Historical bets — closed")
    out.append("</div></div>")


def _render_active_bets_table(out: List[str], bets: List[dict],
                              empty_msg: str = "No active bets.") -> None:
    """Shared renderer used by both Section 1 (cross-bot summary) and
    Section 5 (the currently-filtered bot's active bet). One source of
    truth so the columns stay in lockstep.
    """
    if not bets:
        out.append(f"<div class='empty'>{html.escape(empty_msg)}</div>")
        return
    out.append("<table><thead><tr>"
               "<th>Opened</th><th>Bot</th><th>Ticker</th><th>Side</th>"
               "<th class='num'>Entry</th><th class='num'>Current</th>"
               "<th class='num'>Contracts</th>"
               "<th class='num' title='Entry price × contracts — cash at risk'>Cost</th>"
               "<th class='num' title='(100 − entry) × contracts — gross profit if our side wins'>Potential gain</th>"
               "<th class='num' title='Time until the contract resolves'>Closes in</th>"
               "</tr></thead><tbody>")
    for b in bets:
        opened = (b.get("opened_at") or "")[:19].replace("T", " ")
        side = (b.get("side") or "").upper()
        badge_cls = "badge-yes" if side == "YES" else "badge-no"
        entry = b.get("entry_price_cents") or 0
        contracts = b.get("contracts", 0) or 0
        current = b.get("mark_yes_ask") if side == "YES" else b.get("mark_no_ask")
        bot_name = b.get("_bot_name", "—")
        cost = entry * contracts / 100.0
        potential_gain = (100 - entry) * contracts / 100.0
        mtc = b.get("minutes_to_close")
        out.append(
            f"<tr><td>{html.escape(opened)}</td>"
            f"<td>{html.escape(bot_name)}</td>"
            f"<td class='mono'>{html.escape(b['ticker'])}</td>"
            f"<td><span class='badge {badge_cls}'>{side}</span></td>"
            f"<td class='num'>{entry}c</td>"
            f"<td class='num'>{cents_or_dash(current)}</td>"
            f"<td class='num'>{contracts}</td>"
            f"<td class='num red'>−${cost:.2f}</td>"
            f"<td class='num green'>+${potential_gain:.2f}</td>"
            f"<td class='num'>{time_to_close_str(mtc)}</td></tr>"
        )
    out.append("</tbody></table>")


def _render_bet_history_block(out: List[str], history: List[dict],
                               heading: str = "Historical bets — closed",
                               shown_initially: int = 5) -> None:
    """Subsection: closed bets with entry/exit/P&L. Used inline under
    Section 1 (Summary) and Section 5 (Active bet) so each view shows
    the lifetime trade ledger directly under its active-bets table.

    Uses HTML <details>/<summary> so the first ``shown_initially`` rows
    are visible and the rest are collapsible — no JS.
    """
    out.append(f"<h3 class='subhead'>{html.escape(heading)}</h3>")
    if not history:
        out.append("<div class='empty'>No closed bets yet.</div>")
        return

    head = (
        "<table><thead><tr>"
        "<th>Closed</th><th>Ticker</th><th>Side</th>"
        "<th class='num'>Entry</th><th class='num'>Exit</th>"
        "<th class='num'>Contracts</th>"
        "<th class='num' title='Model probability for the side we bet on, recorded at entry.'>Model p</th>"
        "<th class='num' title='Net EV per contract at entry: (model_p − entry_price) − half-spread. "
        "Positive = +EV trade. Compare with realized P&amp;L to spot model→trade translation gaps.'>Entry EV</th>"
        "<th class='num' title='Underlying gas price ($/gal, EIA national "
        "average) at the moment this bet closed. EIA differs from AAA "
        "(Kalshi resolution) by ~1-3¢.'>Gas at close</th>"
        "<th class='num'>P&amp;L</th>"
        "<th>Outcome</th>"
        "<th title='Post-trade error classification — see tooltip on each cell. "
        "Helps separate process errors (we shouldn’t have taken the trade) "
        "from variance (good bet, bad outcome).'>Error type</th>"
        "</tr></thead><tbody>"
    )

    error_explainers = {
        "BAD_BET": "Entry-time EV was already negative. Should not have been taken.",
        "EXECUTION_BAD_PRICE": "Break-even probability was extreme (>85%); paid too much for too thin an edge.",
        "LOW_CONFIDENCE_TRADE": "Model probability for the chosen side was 50–60%; signal too weak.",
        "MODEL_OVERCONFIDENT": "Model said >75% on this side, but the trade lost. Calibration miss.",
        "GOOD_BET_BAD_OUTCOME": "Entry EV was positive — this is variance, not a process error.",
    }

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
        gas_close = b.get("gas_price_at_close")
        gas_str = f"${gas_close:.3f}" if gas_close is not None else "—"
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
        err = b.get("error_type") or ""
        if err:
            err_cls = ("red" if err in ("BAD_BET", "EXECUTION_BAD_PRICE",
                                        "MODEL_OVERCONFIDENT")
                       else ("yellow" if err == "LOW_CONFIDENCE_TRADE"
                             else "gray"))
            err_tt = error_explainers.get(err, err)
            err_html = (f"<span class='status-pill {err_cls}' "
                        f"title='{html.escape(err_tt)}'>{html.escape(err)}</span>")
        else:
            err_html = "<span class='small gray'>—</span>"
        return (f"<tr><td>{html.escape(closed)}</td>"
                f"<td class='mono'>{html.escape(b['ticker'])}</td>"
                f"<td><span class='badge {badge_cls}'>{side}</span></td>"
                f"<td class='num'>{entry}c</td>"
                f"<td class='num'>{cents_or_dash(exit_c)}</td>"
                f"<td class='num'>{contracts}</td>"
                f"<td class='num'>{mp_str}</td>"
                f"<td class='num {ev_cls}'>{ev_str}</td>"
                f"<td class='num'>{gas_str}</td>"
                f"<td class='num {pnl_cls_}'>{fmt_signed_cents(pnl)}</td>"
                f"<td class='{pnl_cls_}'>{outcome}</td>"
                f"<td>{err_html}</td></tr>")

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
                       current_bot: str) -> None:
    """Slim filter bar: pill buttons, one per bot. Selected pill is
    highlighted; others are clickable links that switch via the
    ?bot= query param. No section box / heading — this is a filter,
    not a content section."""
    out.append("<div class='bot-filter-bar'>")
    out.append("<span class='filter-label'>Bot</span>")
    for b in available_bots:
        classes = ["filter-pill"]
        if b["key"] == current_bot:
            classes.append("filter-pill-active")
        if not b.get("available", True):
            classes.append("filter-pill-disabled")
        cls = " ".join(classes)
        avail_marker = "" if b.get("available", True) else " <span class='small gray'>(no data)</span>"
        out.append(
            f"<a href='?bot={html.escape(b['key'])}' class='{cls}'>"
            f"{html.escape(b['name'])}{avail_marker}</a>"
        )
    out.append("</div>")


def _render_bot_unavailable(out: List[str], bot_key: str) -> None:
    out.append("<div class='section'><h2>Bot data unavailable</h2><div class='body'>"
               f"<div class='empty'>The <b>{html.escape(bot_key)}</b> bot is registered "
               f"but has no data on this host yet. Switch to a different bot above, "
               f"or run that bot's service to populate <code>data/sim.db</code>.</div>"
               "</div></div>")


def _render_buy_criteria_table(out: List[str],
                                edge_cfg: dict | None,
                                validator_cfg: dict | None,
                                risk_caps: dict | None,
                                hedge_cfg: dict | None) -> None:
    """Standalone renderer for the buy-criteria table.

    Three-column table (gate / rule / why-it-matters). Values pull
    live from config so anything you change in config.yaml shows up
    here on next reload — no separate display string to keep in sync.

    Designed to live underneath the watchlist table so a row's
    verdict can be cross-referenced against the rules in one glance.
    """
    if not edge_cfg or not validator_cfg or not risk_caps:
        return
    out.append("<h3 class='subhead' style='margin-top:22px;'>"
               "Buy criteria — what the bot needs before placing a bet</h3>")
    out.append("<div class='small gray' style='margin-bottom:14px;'>"
               "Every gate below must clear before the bot places a (sim) bet. "
               "Expected value is the headline check; the rest are guardrails "
               "against trading into thin / noisy / stale-anchor markets.</div>")
    out.append("<table class='criteria criteria-wide'><thead><tr>"
               "<th>Gate</th><th>Rule</th><th>Why it matters</th>"
               "</tr></thead><tbody>")

    def _row(label: str, rule: str, why: str) -> str:
        return (f"<tr><td>{html.escape(label)}</td>"
                f"<td><code>{html.escape(rule)}</code></td>"
                f"<td class='criteria-why'>{html.escape(why)}</td></tr>")

    def _group(label: str) -> str:
        return (f"<tr class='criteria-group'><td colspan='3'>"
                f"<b>{html.escape(label)}</b></td></tr>")

    # ── Expected value ────────────────────────────────────────────────
    out.append(_group("Expected value"))
    out.append(_row(
        "Net EV per contract",
        f"≥ ${edge_cfg.get('min_ev_per_contract', 0.03):.2f}",
        "Expected dollar profit per $1 contract after the round-trip "
        "spread cost. Required to be positive by a meaningful margin "
        "so noise can't flip a marginal bet into a losing one. Below "
        "this threshold the spread eats whatever edge the model thought "
        "it had."))
    out.append(_row(
        "Model prob over break-even",
        f"model_p − entry_price ≥ "
        f"{edge_cfg.get('min_prob_edge_over_breakeven', 0.05)*100:.0f} pts",
        "A 5pt cushion above pure break-even probability. Without it a "
        "small probability error would tip a 'positive EV' bet into "
        "negative; the cushion absorbs that uncertainty."))
    out.append(_row(
        "Confidence band (skip middle)",
        f"model_p outside "
        f"[{int(edge_cfg.get('min_model_confidence', 0.40)*100)}%, "
        f"{int((1-edge_cfg.get('min_model_confidence', 0.40))*100)}%]",
        "When the model says 40–60% it has no clear directional view. "
        "Apparent edges in this band are mostly noise; the bot stays out."))
    out.append(_row(
        "Per-market confidence",
        f"|model_p − 0.5| × 2 × accuracy ≥ "
        f"{edge_cfg.get('min_confidence', 0.30)*100:.0f}%",
        "Combines the model's distance-from-coinflip with its track "
        "record. A confident view (90%) from a weak model (52% accuracy) "
        "fails this gate; so does a 65% view from any model."))
    out.append(_row(
        "Global model accuracy",
        f"directional accuracy ≥ "
        f"{edge_cfg.get('min_model_accuracy', 0.55):.2f}",
        "If the model's training-holdout accuracy can't beat 55%, it "
        "shouldn't be trading anything live. Hard kill switch — defends "
        "against acting on a freshly-retrained model that just got worse."))

    # ── Liquidity ─────────────────────────────────────────────────────
    out.append(_group("Liquidity"))
    out.append(_row(
        "Volume (lifetime trades)",
        f"≥ {validator_cfg.get('min_volume', 50)} contracts",
        "Total contracts traded since the market opened. A market with "
        "only 5 trades isn't a wisdom-of-crowds signal — it's whatever "
        "the last few people happened to do. The Kalshi YES% only means "
        "something on a market with real flow."))
    out.append(_row(
        "Open interest",
        f"≥ {validator_cfg.get('min_open_interest', 50)} contracts",
        "Number of currently-open contracts. Volume measures activity; "
        "open interest measures conviction (people actually holding). "
        "Both signals matter — markets dominated by a single whale's "
        "position fail this gate."))
    out.append(_row(
        "Book depth within 3¢",
        f"≥ {validator_cfg.get('min_book_depth_contracts', 100)} contracts",
        "Total contracts available within 3¢ of the mid price. Ensures "
        "the displayed price isn't a single-tick mirage; we need depth "
        "so a small fill doesn't move the price 10pt against us."))
    out.append(_row(
        "Depth at exact best ask",
        f"≥ {validator_cfg.get('min_depth_at_best_ask', 25)} contracts",
        "Contracts available at the price we'd actually pay. Without "
        "this the displayed 9c ask might have only 1 contract while "
        "the next price level is 17c — your effective fill is much "
        "worse than the headline price."))
    out.append(_row(
        "Spread",
        f"≤ {validator_cfg.get('max_spread_cents', 8)}¢",
        "Round-trip spread is the largest transaction cost. An 8c "
        "spread eats 4c on entry and 4c on exit, killing most edges "
        "before they can pay out."))
    pb = validator_cfg.get('prob_bounds_cents') or [5, 95]
    out.append(_row(
        "YES ask price band",
        f"in {pb[0]}¢ – {pb[1]}¢",
        "Avoid deep tails priced ~0¢ or ~100¢. Asymmetric payoffs at "
        "the extremes (you risk 99c to make 1c, or vice versa) make "
        "edges fragile and spread cost dominate."))

    # ── Time-to-close ─────────────────────────────────────────────────
    out.append(_group("Time-to-close"))
    mn = validator_cfg.get('min_minutes_to_close', 30)
    mx = validator_cfg.get('max_minutes_to_close', 10080)
    out.append(_row(
        "TTC window",
        f"{mn} min ≤ TTC ≤ {mx//1440} days",
        "Under the floor: model's weekly forecast is meaningless and "
        "resolution is too noisy. Above the ceiling: the model's "
        "1-week-ahead training horizon doesn't apply — different "
        "macro features matter at multi-week timescales."))
    bk = validator_cfg.get('basis_risk_strike_window_dollars', 0.0)
    bh = validator_cfg.get('basis_risk_max_hours_to_close', 0.0)
    if bk > 0 and bh > 0:
        out.append(_row(
            "Basis-risk zone (skip)",
            f"strike within ±${bk:.2f} of model median AND TTC < {bh:.0f}h",
            "At near-resolution short distances, the EIA-vs-AAA "
            "anchor mismatch (~1-3¢) dominates apparent edges. Strikes "
            "in this zone look like big mispricings but are systematically "
            "wrong — the model is anchored to EIA, Kalshi resolves on AAA."))

    # ── Risk caps ─────────────────────────────────────────────────────
    out.append(_group("Risk caps"))
    out.append(_row(
        "Open positions",
        f"≤ {risk_caps.get('max_open_positions', 1)} at a time",
        "Keeps total dollars at risk small while we validate the "
        "strategy. Increase only after sustained positive realized P&L."))
    out.append(_row(
        "Total exposure",
        f"≤ ${risk_caps.get('max_total_exposure_cents', 200)/100:.2f}",
        "Hard guardrail on total cash at risk in open bets. The sim "
        "should mirror real-money risk discipline so behavior translates."))
    out.append(_row(
        "Bets per day",
        f"≤ {risk_caps.get('max_bets_per_day', 50)}",
        "Defends against a tilt streak if a bug or data hiccup makes "
        "everything look like a winner — caps damage from any single day."))
    cd_min = risk_caps.get('cooldown_seconds_same_market', 1800) // 60
    out.append(_row(
        "Same-market cooldown",
        f"{cd_min} min between bets on the same ticker",
        "Avoids piling into a single market on consecutive ticks if the "
        "price hovers near our trigger. Forces some price discovery "
        "between attempts."))

    # ── Hedging ───────────────────────────────────────────────────────
    out.append(_group("Hedging — when the bot exits early"))
    out.append(_row(
        "EV inverted",
        "current_p − live_break_even ≤ −5 pts",
        "If the model's probability for our side has fallen far enough "
        "below the live break-even (the current ask), we're now in a "
        "negative-EV trade. Close it instead of waiting for resolution."))
    out.append(_row(
        "EV-realized lock",
        "mark ≥ entry + 20¢ AND remaining EV < 2¢",
        "We've already won most of the available edge. Why hold for "
        "a few more cents of remaining EV? Lock the gain, free the "
        "position slot for a better opportunity."))

    out.append("</tbody></table>")


def _render_model_section(out: List[str], model: dict | None) -> None:
    out.append("<div class='section'><h2>2 · Model — what the bot believes right now</h2>"
               "<div class='body'>")
    if not model:
        out.append("<div class='empty'>No model snapshot yet — wait for the first tick.</div>")
        out.append("</div></div>")
        return

    accuracy = float(model.get("classifier_accuracy") or 0)

    # 4a · Model strength (historical performance). Compact: 6 cards / row.
    out.append("<div class='subsec'><h3 class='subhead'>Model strength (from training holdout)</h3>")
    out.append("<div class='row compact'>")
    out.append(f"<div class='card'><div class='label'>Directional accuracy</div>"
               f"<div class='value'>{accuracy*100:.1f}%</div></div>")

    def _metric_card(label: str, value: float | None, suffix: str = "%") -> str:
        if value is None:
            return (f"<div class='card'><div class='label'>{label}</div>"
                    f"<div class='value gray'>—</div></div>")
        return (f"<div class='card'><div class='label'>{label}</div>"
                f"<div class='value'>{value*100:.0f}{suffix}</div></div>")

    out.append(_metric_card("Precision", model.get("training_precision")))
    out.append(_metric_card("Recall", model.get("training_recall")))
    out.append(_metric_card("F1", model.get("training_f1")))
    out.append(_metric_card("ROC AUC", model.get("training_roc_auc")))
    out.append(f"<div class='card'><div class='label'>Features used</div>"
               f"<div class='value'>{int(model.get('feature_count') or 0)}</div></div>")
    # Actual win % from real closed bets — the model's ground-truth
    # performance, distinct from the training-holdout directional
    # accuracy card to its left. Compare the two: they should converge
    # as closed-bet sample size grows.
    a_wins = int(model.get("actual_wins") or 0)
    a_losses = int(model.get("actual_losses") or 0)
    a_total = a_wins + a_losses
    if a_total > 0:
        a_pct = a_wins / a_total
        a_str = f"{a_pct*100:.0f}%"
        a_cls = "green" if a_pct > 0.55 else ("red" if a_pct < 0.45 else "")
    else:
        a_str = "—"
        a_cls = "gray"
    out.append(f"<div class='card'><div class='label' "
               f"title='Actual win % from this bot’s closed bets ({a_wins}W / "
               f"{a_losses}L). The training-holdout Directional Accuracy is "
               f"a backtest estimate; this is the live result. They should "
               f"converge as the sample grows.'>"
               f"Actual win %</div>"
               f"<div class='value {a_cls}'>{a_str}</div></div>")
    out.append("</div></div>")

    out.append("</div></div>")


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
    out.append("<div class='subsec'>"
               "<h3 class='subhead' style='margin-left:0;'>Current prediction</h3>")
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


def _accuracy_verdict(acc: float) -> str:
    if acc < 0.52:
        return "essentially a coinflip — don't deploy"
    if acc < 0.58:
        return "weakly positive — easily eaten by spread"
    if acc < 0.65:
        return "tradeable signal at sufficient edge"
    return "strong signal — validate carefully"


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


def _render_ev_diagnostic(out: List[str], pos: dict, side: str, entry_cents: int) -> None:
    """Show the four numbers that determine whether this trade should
    still be open right now: break-even prob, model prob (selected
    side), entry-time EV, and a current-mark EV recomputed using the
    live ask price (so the user sees how the trade has decayed/
    improved since entry)."""
    be_prob = pos.get("break_even_probability")
    if be_prob is None:
        be_prob = entry_cents / 100.0
    model_p_yes = pos.get("model_yes_prob_at_entry")
    if model_p_yes is None:
        return  # legacy bet — no entry-time stats; skip the panel cleanly
    p_selected = float(model_p_yes) if side == "YES" else (1.0 - float(model_p_yes))
    entry_ev = pos.get("expected_ev_at_entry")

    # Current EV: same formula but using the LIVE mark price as the
    # exit-equivalent breakeven. This answers "if I closed right now,
    # is this still a +EV position?". Mark = the side's current ask.
    mark_cents = pos.get("mark_yes_ask") if side == "YES" else pos.get("mark_no_ask")
    cur_be = (mark_cents / 100.0) if mark_cents is not None else None
    cur_ev = (p_selected - cur_be) if cur_be is not None else None

    entry_cls, entry_lbl = _ev_status(entry_ev)
    cur_cls, cur_lbl = _ev_status(cur_ev)
    contracts = int(pos.get("contracts") or 1)

    out.append("<h3 class='subhead'>EV check — is this still a good trade?</h3>")
    out.append("<div class='row compact'>")
    out.append(f"<div class='card'><div class='label' "
               f"title='Entry price as a probability — the model must beat this for the bet to have positive EV.'>"
               f"Break-even prob</div>"
               f"<div class='value'>{be_prob*100:.0f}%</div></div>")
    out.append(f"<div class='card'><div class='label' "
               f"title='The model’s probability for the side we bought ({side}). "
               f"Compare this against break-even — model > break-even means positive expected value.'>"
               f"Model prob ({side})</div>"
               f"<div class='value'>{p_selected*100:.0f}%</div></div>")
    if entry_ev is not None:
        out.append(f"<div class='card'><div class='label' "
                   f"title='Net EV per contract at entry: (model_prob − entry_price) − half-spread. "
                   f"Total stake EV ≈ ${entry_ev * contracts:+.2f}.'>"
                   f"Entry-time EV / contract</div>"
                   f"<div class='value {entry_cls}'>${entry_ev:+.3f}</div>"
                   f"<div class='small {entry_cls}'>{entry_lbl}</div></div>")
    if cur_ev is not None:
        out.append(f"<div class='card'><div class='label' "
                   f"title='Live EV recomputed against the current mark. If this has gone red, "
                   f"the trade has decayed — consider an EV-aware hedge or exit.'>"
                   f"Current EV / contract</div>"
                   f"<div class='value {cur_cls}'>${cur_ev:+.3f}</div>"
                   f"<div class='small {cur_cls}'>{cur_lbl}</div></div>")
    out.append("</div>")
    # Loud banner when the model says we're now upside-down.
    if cur_ev is not None and cur_ev < 0:
        out.append(
            "<div class='ev-warning' role='alert'>"
            "⚠ <strong>NEGATIVE EV right now.</strong> Model probability "
            f"({p_selected*100:.0f}%) is below the live break-even "
            f"({(cur_be or 0)*100:.0f}%). A large gap column does NOT make a "
            "trade good — EV does. Consider a hedge or exit on the next tick."
            "</div>"
        )


def _render_why_panel(out: List[str], pos: dict, mtc_min, wl: dict | None) -> None:
    """The 'why this trade was taken' audit. All numbers come from the
    Decision row stored on the position at open time, so this answers
    'why did we do this?' even days later when prices have moved."""
    model_p_yes = pos.get("model_yes_prob_at_entry")
    kalshi_p_yes = pos.get("kalshi_yes_prob_at_entry")
    if model_p_yes is None or kalshi_p_yes is None:
        return  # legacy bet
    side = (pos.get("side") or "").upper()
    model_p_no = 1.0 - float(model_p_yes)
    kalshi_p_no = 1.0 - float(kalshi_p_yes)
    selected_p = float(model_p_yes) if side == "YES" else model_p_no
    selected_k = float(kalshi_p_yes) if side == "YES" else kalshi_p_no
    edge_pts = (selected_p - selected_k) * 100
    spread = pos.get("mark_spread")
    contracts = int(pos.get("contracts") or 1)

    try:
        gates_passed = json.loads(pos.get("gates_passed_json") or "[]")
        gates_failed = json.loads(pos.get("gates_failed_json") or "[]")
    except Exception:  # noqa: BLE001
        gates_passed, gates_failed = [], []

    out.append("<h3 class='subhead'>Why this trade was taken</h3>")
    out.append("<div class='why-grid'>")
    out.append(f"<div class='why-row'><span>Model YES</span><span>{model_p_yes*100:.0f}%</span></div>")
    out.append(f"<div class='why-row'><span>Model NO</span><span>{model_p_no*100:.0f}%</span></div>")
    out.append(f"<div class='why-row'><span>Kalshi YES</span><span>{kalshi_p_yes*100:.0f}%</span></div>")
    out.append(f"<div class='why-row'><span>Kalshi NO</span><span>{kalshi_p_no*100:.0f}%</span></div>")
    out.append(f"<div class='why-row'><span>Selected side</span><span>{side}</span></div>")
    out.append(f"<div class='why-row'><span>Edge (pts)</span><span>{edge_pts:+.0f}</span></div>")
    out.append(f"<div class='why-row'><span>Spread at entry</span>"
               f"<span>{spread if spread is not None else '—'}c</span></div>")
    out.append(f"<div class='why-row'><span>Contracts</span><span>{contracts}</span></div>")
    if mtc_min is not None:
        out.append(f"<div class='why-row'><span>Time to close (now)</span>"
                   f"<span>{time_to_close_str(mtc_min)}</span></div>")
    out.append("</div>")
    if gates_passed or gates_failed:
        out.append("<div class='why-gates'>")
        if gates_passed:
            passed_str = ", ".join(html.escape(g) for g in gates_passed)
            out.append(f"<div><span class='small green'>Passed:</span> "
                       f"<span class='small mono'>{passed_str}</span></div>")
        if gates_failed:
            failed_str = ", ".join(html.escape(g) for g in gates_failed)
            out.append(f"<div><span class='small red'>Failed:</span> "
                       f"<span class='small mono'>{failed_str}</span></div>")
        out.append("</div>")


def _render_active_bet(out: List[str], pos: dict | None,
                       watchlist: List[dict],
                       closed_history: List[dict]) -> None:
    out.append("<div class='section'><h2>3 · Active bet — currently invested</h2>"
               "<div class='body'>")
    if not pos:
        out.append("<div class='empty'>No active bets right now.</div>")
        # Still show this bot's closed history below the empty notice.
        _render_bet_history_block(out, closed_history,
                                  heading="Historical bets — closed (this bot)")
        out.append("</div></div>")
        return

    # Pull the canonical Kalshi event title (e.g. "US gas prices this week")
    # from the active position's matching watchlist row. Used to prefix the
    # question line below.
    event_title = ""
    for w in watchlist:
        if w.get("ticker") == pos.get("ticker") and (w.get("event_title") or "").strip():
            event_title = w["event_title"].strip()
            break
    if not event_title:
        # Fallback: any watchlist row in the same series.
        for w in watchlist:
            if (w.get("event_title") or "").strip():
                event_title = w["event_title"].strip()
                break

    entry = int(pos["entry_price_cents"])
    contracts = int(pos["contracts"])
    side = (pos["side"] or "").upper()
    badge_cls = "badge-yes" if side == "YES" else "badge-no"
    mark = pos.get("mark_yes_ask") if side == "YES" else pos.get("mark_no_ask")
    pnl = unrealized_pnl_cents(pos)
    pnl_cls = "green" if (pnl or 0) > 0 else ("red" if (pnl or 0) < 0 else "gray")
    ticker = pos["ticker"]
    wl = next((w for w in watchlist if w["ticker"] == ticker), None)
    question = "(question unknown)"
    if wl:
        question = question_str(wl.get("direction", ""), wl.get("strike_low"),
                                 wl.get("strike_high"))
    opened = (pos.get("opened_at") or "")[:19].replace("T", " ")
    pct_pnl = (pnl or 0) / max(1, entry * contracts) * 100

    # The duplicate stat boxes (Entry/Current/Contracts/Cost/Potential
    # gain/Closes in) were removed per request — the same data is in the
    # active-bets table directly below, and the question line + chart
    # carry the visual context.
    mtc = wl.get("minutes_to_close") if wl else None

    out.append("<div class='hero-card'>")
    # Question line: "US gas prices this week  [NO]  above $4.10"
    title_prefix = (f"<span class='hero-event-title'>{html.escape(event_title)}</span> "
                    if event_title else "")
    out.append(f"<div class='hero-question'>{title_prefix}"
               f"<span class='badge {badge_cls}'>{side}</span> "
               f"<span class='hero-q-text'>{html.escape(question)}</span></div>")
    out.append(f"<div class='hero-ticker'>{html.escape(ticker)} · "
               f"opened {html.escape(opened)} UTC</div>")

    # Active-bets table (same columns as Section 1's summary view).
    # Bot-name fallback: if not already attached, infer from the
    # ticker prefix.
    enriched = dict(pos)
    enriched.setdefault("minutes_to_close", mtc)
    if not enriched.get("_bot_name"):
        ticker = (enriched.get("ticker") or "")
        if ticker.startswith("KXJOBLESSCLAIMS"):
            enriched["_bot_name"] = "Jobless Claims"
        elif ticker.startswith("KXAAAGASW") or ticker.startswith("KXAAAGASD"):
            enriched["_bot_name"] = "Retail Gas Prices"
        elif ticker.startswith("KXNATGAS"):
            enriched["_bot_name"] = "Natural Gas Prices"
        else:
            enriched["_bot_name"] = "—"
    out.append("<div style='padding:6px 0 14px 0;'>")
    _render_active_bets_table(out, [enriched])
    out.append("</div>")

    # ── EV / break-even diagnostic block ───────────────────────────────
    # Pulled from columns mirrored at open from the Decision dataclass.
    # Recomputes "current EV" using the live mark; that's the live-tick
    # answer to "is this still a positive-EV trade right now?" rather
    # than what it looked like at entry.
    _render_ev_diagnostic(out, pos, side, entry)

    # ── "Why this trade was taken" panel ───────────────────────────────
    _render_why_panel(out, pos, mtc, wl)

    # The per-position market-price chart that lived here was retired:
    # the watchlist hero chart now shows the underlying with a horizontal
    # line at this position's strike, which is more useful for tracking
    # whether the bet is going to resolve YES or NO.
    out.append("</div>")  # /hero-card

    # Per request: closed-bet history with the same columns as Section 1's
    # bet history. Scoped to THIS bot.
    _render_bet_history_block(out, closed_history,
                              heading="Historical bets — closed (this bot)")
    out.append("</div></div>")


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
    """Kalshi-style hero block: current underlying value, % change, total
    Kalshi volume on the watchlist, time-to-close on the soonest market,
    and an SVG chart of the underlying. If there's an active position,
    a horizontal line on the chart marks the strike the user bought.
    """
    # Prefer Kalshi-derived underlying data for "current value" and
    # "% change". It's fresher (updates with every Kalshi tick) and
    # works even when the bot service isn't running locally. Falls back
    # to the bot's own model_snapshot only if Kalshi data is missing.
    current: float | None = None
    earliest_value: float | None = None
    if kalshi_history:
        # Take latest from Kalshi.
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
    out.append(
        f"<div class='wl-hero-price'>{html.escape(current_str)}"
        f"<span class='wl-hero-price-label'>forecast</span>"
        f"</div>"
    )
    arrow = ""
    if value_change is not None:
        arrow = "▲" if value_change >= 0 else "▼"
    change_display = (change_body if not arrow
                      else f"{arrow} {change_body}")
    out.append(
        f"<div class='wl-hero-change {change_cls}'>"
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

    # Chart: prefer Kalshi-derived underlying (matches Kalshi's market-
    # page chart axis). Falls back to the bot's own model_snapshots if
    # Kalshi creds aren't configured or the series has no candle history.
    if kalshi_history:
        out.append(svg_kalshi_chart(
            kalshi_history, display,
            reference_strike=reference_strike,
            strike_side=strike_side,
            strike_is_active_bet=strike_is_active,
            contract_open_ts=contract_open_ts,
            contract_close_ts=contract_close_ts,
            total_volume=total_volume,
        ))
    else:
        out.append(svg_underlying_chart(
            underlying_history, current, display,
            reference_strike=reference_strike,
            strike_side=strike_side,
            strike_is_active_bet=strike_is_active,
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
                      hedge_cfg: dict | None = None) -> None:
    accuracy = float(model["classifier_accuracy"]) if model and model.get("classifier_accuracy") else None
    accuracy_label = (f"{accuracy*100:.0f}%" if accuracy else "untrained")
    out.append(f"<div class='section'><h2>4 · Watchlist — model vs market "
               f"<span class='small gray'>(model historical accuracy {accuracy_label}; "
               f"confidence is scaled by it)</span></h2><div class='body'>")
    # Current prediction (moved here from the Model section per request).
    _render_current_prediction(out, model, display=display)

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
            ask_c = ya_c if best_side_v == "YES" else na_c
            ask_str = f" @ {ask_c}c" if ask_c is not None else ""
            cls = "badge-yes" if best_side_v == "YES" else "badge-no"
            badge = (f"<span class='badge {cls}'{tt}>"
                     f"BUY {best_side_v}{ask_str}</span>")
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
    # Buy-criteria reference sits directly under the watchlist table so
    # any WATCH/SKIP verdict can be cross-referenced against the rules
    # in one glance.
    _render_buy_criteria_table(out, edge_cfg, validator_cfg,
                                risk_caps, hedge_cfg)
    out.append("</div></div>")


def _render_buy_criteria(out: List[str],
                          edge_cfg: dict | None,
                          validator_cfg: dict | None,
                          risk_caps: dict | None) -> None:
    """Compact reference card listing every gate the bot enforces
    before opening a position. Lives under the watchlist so you can
    cross-reference rows against the rules in one glance."""
    if not edge_cfg or not validator_cfg or not risk_caps:
        return  # render_page wasn't called with cfg — skip cleanly

    # Helpers to keep formatting tight and consistent.
    def _row(label: str, rule: str) -> str:
        return (f"<tr><td>{html.escape(label)}</td>"
                f"<td><code>{html.escape(rule)}</code></td></tr>")

    out.append("<h3 class='subhead' style='margin-top:18px;'>"
               "Buy criteria — what the bot needs before placing a bet</h3>")
    out.append("<div class='small gray' style='margin-bottom:8px;'>"
               "Every gate below must clear. EV is the headline check; "
               "the rest are guardrails against trading into thin/noisy/"
               "stale-anchor markets.</div>")
    out.append("<table class='criteria'><thead><tr>"
               "<th>Gate</th><th>Rule</th></tr></thead><tbody>")

    # ── Edge / EV gates ────────────────────────────────────────────
    out.append("<tr class='criteria-group'><td colspan='2'>"
               "<b>Expected value</b></td></tr>")
    out.append(_row("Net EV per contract",
                    f"≥ ${edge_cfg.get('min_ev_per_contract', 0.03):.2f}"))
    out.append(_row("Model prob over break-even",
                    f"model_p − entry_price ≥ "
                    f"{edge_cfg.get('min_prob_edge_over_breakeven', 0.05)*100:.0f} pts"))
    out.append(_row("Confidence band (skip middle)",
                    f"model_p outside "
                    f"[{int(edge_cfg.get('min_model_confidence', 0.40)*100)}%, "
                    f"{int((1-edge_cfg.get('min_model_confidence', 0.40))*100)}%]"))
    out.append(_row("Per-market confidence",
                    f"|model_p − 0.5| × 2 × accuracy ≥ "
                    f"{edge_cfg.get('min_confidence', 0.30)*100:.0f}%"))
    out.append(_row("Global model accuracy",
                    f"directional accuracy ≥ "
                    f"{edge_cfg.get('min_model_accuracy', 0.55):.2f}"))

    # ── Liquidity gates ────────────────────────────────────────────
    out.append("<tr class='criteria-group'><td colspan='2'>"
               "<b>Liquidity</b></td></tr>")
    out.append(_row("Volume (lifetime trades)",
                    f"≥ {validator_cfg.get('min_volume', 50)} contracts"))
    out.append(_row("Open interest",
                    f"≥ {validator_cfg.get('min_open_interest', 50)} contracts"))
    out.append(_row("Book depth within 3¢",
                    f"≥ {validator_cfg.get('min_book_depth_contracts', 100)} contracts"))
    out.append(_row("Depth at exact best ask",
                    f"≥ {validator_cfg.get('min_depth_at_best_ask', 25)} contracts"))
    out.append(_row("Spread",
                    f"≤ {validator_cfg.get('max_spread_cents', 8)}¢"))
    pb = validator_cfg.get('prob_bounds_cents') or [5, 95]
    out.append(_row("YES ask price band",
                    f"in {pb[0]}¢ – {pb[1]}¢ (avoid deep tails)"))

    # ── Time gates ─────────────────────────────────────────────────
    out.append("<tr class='criteria-group'><td colspan='2'>"
               "<b>Time-to-close</b></td></tr>")
    mn = validator_cfg.get('min_minutes_to_close', 30)
    mx = validator_cfg.get('max_minutes_to_close', 10080)
    out.append(_row("Time-to-close window",
                    f"{mn} min ≤ TTC ≤ {mx//1440} days"))
    bk = validator_cfg.get('basis_risk_strike_window_dollars', 0.0)
    bh = validator_cfg.get('basis_risk_max_hours_to_close', 0.0)
    if bk > 0 and bh > 0:
        out.append(_row("Basis-risk zone (skip)",
                        f"strike within ±${bk:.2f} of model median "
                        f"AND TTC < {bh:.0f}h"))

    # ── Risk caps ──────────────────────────────────────────────────
    out.append("<tr class='criteria-group'><td colspan='2'>"
               "<b>Risk caps</b></td></tr>")
    out.append(_row("Open positions",
                    f"≤ {risk_caps.get('max_open_positions', 1)} at a time"))
    out.append(_row("Total exposure",
                    f"≤ ${risk_caps.get('max_total_exposure_cents', 200)/100:.2f}"))
    out.append(_row("Bets per day",
                    f"≤ {risk_caps.get('max_bets_per_day', 50)}"))
    cd_min = risk_caps.get('cooldown_seconds_same_market', 1800) // 60
    out.append(_row("Same-market cooldown",
                    f"{cd_min} min between bets on the same ticker"))

    out.append("</tbody></table>")


def _render_diagnostics(out: List[str], history: List[dict]) -> None:
    """Section 7 — decision-quality diagnostics.

    Four sub-panels:
      • Model → Trade Translation (entry stats vs realized)
      • Calibration table by predicted-probability bucket
      • Performance by time-to-close bucket
      • System Quality Score 0-100
      • Market Flow signals (placeholder when no signal source connected)

    Built entirely from closed-bet history. Sample sizes are shown
    honestly so the user can decide which numbers are statistically
    meaningful and which are still anecdotes.
    """
    out.append("<div class='section'><h2>5 · Decision diagnostics</h2>"
               "<div class='body'>")
    if not history:
        out.append("<div class='empty'>No closed bets yet — diagnostics "
                   "populate as the bot accumulates trade outcomes.</div>")
        # Even with no data, surface the Market Flow placeholder so the
        # section structure is visible.
        _render_market_flow_placeholder(out)
        out.append("</div></div>")
        return

    # ── Aggregate stats from closed bets ────────────────────────────
    n_total = len(history)
    n_with_entry_stats = sum(1 for b in history
                              if b.get("model_yes_prob_at_entry") is not None
                              and b.get("expected_ev_at_entry") is not None)

    avg_model_p = _mean([
        (float(b["model_yes_prob_at_entry"])
         if b.get("side") == "YES"
         else 1.0 - float(b["model_yes_prob_at_entry"]))
        for b in history if b.get("model_yes_prob_at_entry") is not None
    ])
    avg_kalshi_p = _mean([
        (float(b["kalshi_yes_prob_at_entry"])
         if b.get("side") == "YES"
         else 1.0 - float(b["kalshi_yes_prob_at_entry"]))
        for b in history if b.get("kalshi_yes_prob_at_entry") is not None
    ])
    avg_edge_pts = (None if avg_model_p is None or avg_kalshi_p is None
                    else (avg_model_p - avg_kalshi_p) * 100)
    avg_ev_per_contract = _mean([
        float(b["expected_ev_at_entry"]) for b in history
        if b.get("expected_ev_at_entry") is not None
    ])
    # Realized P&L per contract (cents -> dollars).
    avg_realized_per_contract_d = _mean([
        (b.get("realized_pnl_cents") or 0)
        / max(1, b.get("contracts") or 1) / 100.0
        for b in history if b.get("realized_pnl_cents") is not None
    ])

    out.append("<h3 class='subhead'>Model → Trade translation</h3>")
    out.append("<div class='small gray' style='margin-bottom:8px;'>"
               "Are the bot's profitable bets actually profitable? "
               f"Computed across {n_with_entry_stats}/{n_total} closed bets that have entry-time stats.</div>")
    out.append("<div class='row compact'>")
    out.append(_diag_card("Avg model p (selected)", avg_model_p, fmt="pct"))
    out.append(_diag_card("Avg Kalshi p (selected)", avg_kalshi_p, fmt="pct"))
    out.append(_diag_card("Avg entry edge", avg_edge_pts, fmt="pts"))
    out.append(_diag_card("Avg expected EV / contract",
                          avg_ev_per_contract, fmt="dollars",
                          color_by_sign=True))
    out.append(_diag_card("Avg realized $ / contract",
                          avg_realized_per_contract_d, fmt="dollars",
                          color_by_sign=True))
    # EV vs realized — the punchline number for "model right?" vs "execution right?".
    if avg_ev_per_contract is not None and avg_realized_per_contract_d is not None:
        gap = avg_realized_per_contract_d - avg_ev_per_contract
        out.append(_diag_card("Realized − Expected", gap, fmt="dollars",
                              color_by_sign=True,
                              tooltip="Negative => realized worse than EV "
                              "predicted (model overconfident or execution "
                              "leaking value). Positive => model conservative."))
    out.append(_diag_card("Trades in sample", n_with_entry_stats, fmt="int"))
    out.append("</div>")

    # ── Calibration ──────────────────────────────────────────────────
    out.append("<h3 class='subhead'>Calibration — predicted vs realized win rate</h3>")
    cal = _calibration_buckets(history)
    if not any(b["n"] > 0 for b in cal):
        out.append("<div class='empty small'>Need closed bets with model "
                   "probabilities to compute calibration.</div>")
    else:
        out.append("<table><thead><tr>"
                   "<th>Predicted prob bucket</th>"
                   "<th class='num'>Trades</th>"
                   "<th class='num'>Avg predicted</th>"
                   "<th class='num'>Actual win rate</th>"
                   "<th class='num' title='Actual − Predicted. "
                   "Negative = model overconfident in this bucket. "
                   "Positive = model under-confident.'>Calibration gap</th>"
                   "</tr></thead><tbody>")
        for b in cal:
            if b["n"] == 0:
                row = (f"<tr><td>{b['label']}</td>"
                       f"<td class='num gray'>0</td>"
                       f"<td class='num gray'>—</td>"
                       f"<td class='num gray'>—</td>"
                       f"<td class='num gray'>—</td></tr>")
            else:
                gap = b["actual"] - b["predicted"]
                gap_cls = ("red" if gap < -0.10 else
                           ("yellow" if abs(gap) >= 0.05 else "green"))
                row = (f"<tr><td>{b['label']}</td>"
                       f"<td class='num'>{b['n']}</td>"
                       f"<td class='num'>{b['predicted']*100:.0f}%</td>"
                       f"<td class='num'>{b['actual']*100:.0f}%</td>"
                       f"<td class='num {gap_cls}'>{gap*100:+.0f} pts</td></tr>")
            out.append(row)
        out.append("</tbody></table>")

    # ── Time-to-close performance ────────────────────────────────────
    out.append("<h3 class='subhead'>Performance by time-to-close at entry</h3>")
    ttc = _ttc_buckets(history)
    if not any(b["n"] > 0 for b in ttc):
        out.append("<div class='empty small'>Need closed bets with TTC at entry to compute.</div>")
    else:
        out.append("<table><thead><tr>"
                   "<th>TTC bucket</th>"
                   "<th class='num'>Trades</th>"
                   "<th class='num'>Win rate</th>"
                   "<th class='num'>Avg entry EV</th>"
                   "<th class='num'>Realized P&amp;L</th>"
                   "</tr></thead><tbody>")
        for b in ttc:
            if b["n"] == 0:
                row = (f"<tr><td>{b['label']}</td>"
                       f"<td class='num gray'>0</td><td class='num gray'>—</td>"
                       f"<td class='num gray'>—</td><td class='num gray'>—</td></tr>")
            else:
                pnl_str = fmt_signed_cents(b["pnl_cents"])
                pnl_cls = ("green" if b["pnl_cents"] > 0
                           else ("red" if b["pnl_cents"] < 0 else "gray"))
                ev_str = (f"${b['avg_ev']:+.3f}" if b["avg_ev"] is not None else "—")
                ev_cls = (_ev_status(b["avg_ev"])[0]
                          if b["avg_ev"] is not None else "gray")
                row = (f"<tr><td>{b['label']}</td>"
                       f"<td class='num'>{b['n']}</td>"
                       f"<td class='num'>{b['wr']*100:.0f}%</td>"
                       f"<td class='num {ev_cls}'>{ev_str}</td>"
                       f"<td class='num {pnl_cls}'>{pnl_str}</td></tr>")
            out.append(row)
        out.append("</tbody></table>")

    # ── System Quality Score ─────────────────────────────────────────
    score, score_cls, score_breakdown = _system_quality_score(history, cal)
    out.append("<h3 class='subhead'>System quality score</h3>")
    out.append("<div class='row compact'>")
    out.append(f"<div class='card'><div class='label' "
               f"title='Composite 0–100 of: realized P&amp;L sign, calibration "
               f"tightness, EV-vs-realized gap, sample size penalty, drawdown. "
               f"Diagnostic only — NOT a trading signal.'>System quality</div>"
               f"<div class='value {score_cls}'>{score}/100</div></div>")
    for label, val in score_breakdown:
        out.append(f"<div class='card'><div class='label'>{html.escape(label)}</div>"
                   f"<div class='value'>{html.escape(str(val))}</div></div>")
    out.append("</div>")

    # ── Market flow (placeholder until a signal source is wired up) ──
    _render_market_flow_placeholder(out)
    out.append("</div></div>")


def _render_market_flow_placeholder(out: List[str]) -> None:
    """Renders the Market Flow section, even when no upstream signal
    source is connected. Visible-but-empty per spec — the structure
    matters even before we have data."""
    out.append("<h3 class='subhead'>Market flow signals "
               "<span class='small gray'>(no signal source connected)</span></h3>")
    out.append("<div class='row compact'>")
    for label in ("Volume spike", "Order imbalance", "Large trade",
                  "Price acceleration"):
        out.append(f"<div class='card'><div class='label'>{label}</div>"
                   f"<div class='value gray'>—</div>"
                   f"<div class='small gray'>no data</div></div>")
    out.append(f"<div class='card'><div class='label'>Status</div>"
               f"<div class='value gray'>OFFLINE</div>"
               f"<div class='small gray'>placeholder — connect a "
               f"flow / whale-watcher feed</div></div>")
    out.append("</div>")


def _mean(values):
    """Return mean of a list of floats, ignoring Nones. None when empty."""
    xs = [float(v) for v in values if v is not None]
    return (sum(xs) / len(xs)) if xs else None


def _diag_card(label: str, value, fmt: str, color_by_sign: bool = False,
               tooltip: str = "") -> str:
    """One card in a diagnostics row — handles None, formatting, sign coloring."""
    if value is None:
        return (f"<div class='card'><div class='label'"
                f"{(' title=' + repr(tooltip)) if tooltip else ''}>"
                f"{html.escape(label)}</div>"
                f"<div class='value gray'>—</div></div>")
    if fmt == "pct":
        text = f"{value*100:.0f}%"
    elif fmt == "pts":
        text = f"{value:+.0f} pts"
    elif fmt == "dollars":
        text = f"${value:+.3f}"
    elif fmt == "int":
        text = f"{int(value)}"
    else:
        text = str(value)
    cls = ""
    if color_by_sign:
        cls = "green" if value > 0 else ("red" if value < 0 else "gray")
    title_attr = f" title={repr(tooltip)}" if tooltip else ""
    return (f"<div class='card'><div class='label'{title_attr}>"
            f"{html.escape(label)}</div>"
            f"<div class='value {cls}'>{html.escape(text)}</div></div>")


def _calibration_buckets(history: List[dict]) -> List[dict]:
    """Bucket closed bets by selected-side predicted probability and
    measure the empirical win rate inside each bucket."""
    bucket_defs = [
        (0.50, 0.60, "50–60%"),
        (0.60, 0.70, "60–70%"),
        (0.70, 0.80, "70–80%"),
        (0.80, 0.90, "80–90%"),
        (0.90, 1.01, "90–100%"),
    ]
    buckets = [{"label": lbl, "lo": lo, "hi": hi,
                "n": 0, "predicted": 0.0, "actual": 0.0}
               for lo, hi, lbl in bucket_defs]
    for b in history:
        m = b.get("model_yes_prob_at_entry")
        if m is None:
            continue
        side = (b.get("side") or "").upper()
        p_sel = float(m) if side == "YES" else (1.0 - float(m))
        won = (b.get("realized_pnl_cents") or 0) > 0
        for bk in buckets:
            if bk["lo"] <= p_sel < bk["hi"]:
                bk["n"] += 1
                bk["predicted"] += p_sel
                bk["actual"] += 1 if won else 0
                break
    for bk in buckets:
        if bk["n"] > 0:
            bk["predicted"] /= bk["n"]
            bk["actual"] /= bk["n"]
    return buckets


def _ttc_buckets(history: List[dict]) -> List[dict]:
    """Bucket closed bets by elapsed-at-entry time-to-close.

    We don't store TTC-at-entry directly, but opened_at + close_time
    can be derived from the position. Here we approximate using the
    ACTUAL bet duration (opened_at -> exited_at), which equals
    TTC-at-entry as long as the bet was held through resolution
    (true for the bot's flow). For hedged exits this is a slight
    underestimate — fine for first-cut diagnostics.
    """
    bucket_defs = [
        (0.5, 6, "30 min – 6 h"),
        (6, 24, "6 h – 1 d"),
        (24, 72, "1 – 3 d"),
        (72, 168, "3 – 7 d"),
        (168, 1e9, "> 7 d"),
    ]
    buckets = [{"label": lbl, "lo_h": lo, "hi_h": hi,
                "n": 0, "wins": 0, "ev_sum": 0.0, "ev_n": 0,
                "pnl_cents": 0}
               for lo, hi, lbl in bucket_defs]
    from datetime import datetime as _dt
    for b in history:
        opened = b.get("opened_at"); exited = b.get("exited_at")
        if not opened or not exited:
            continue
        try:
            duration_h = (
                (_dt.fromisoformat(exited) - _dt.fromisoformat(opened))
                .total_seconds() / 3600.0
            )
        except Exception:  # noqa: BLE001
            continue
        for bk in buckets:
            if bk["lo_h"] <= duration_h < bk["hi_h"]:
                bk["n"] += 1
                if (b.get("realized_pnl_cents") or 0) > 0:
                    bk["wins"] += 1
                bk["pnl_cents"] += b.get("realized_pnl_cents") or 0
                ev = b.get("expected_ev_at_entry")
                if ev is not None:
                    bk["ev_sum"] += float(ev)
                    bk["ev_n"] += 1
                break
    for bk in buckets:
        bk["wr"] = (bk["wins"] / bk["n"]) if bk["n"] > 0 else 0.0
        bk["avg_ev"] = (bk["ev_sum"] / bk["ev_n"]) if bk["ev_n"] > 0 else None
    return buckets


def _system_quality_score(history: List[dict], cal: List[dict]):
    """A 0–100 composite. Diagnostic only — should NOT drive trading.

    Components (each 0–20 points, summed):
      • realized P&L sign + magnitude
      • calibration tightness (largest |gap| across populated buckets)
      • EV-vs-realized convergence (smaller is better)
      • sample size (caps at 30 trades)
      • drawdown control (worst peak-to-trough fraction of total cost)
    """
    n = len(history)
    pnl = sum((b.get("realized_pnl_cents") or 0) for b in history)
    cost = sum((b.get("entry_price_cents") or 0)
               * (b.get("contracts") or 0) for b in history)
    pnl_dollars = pnl / 100.0
    pnl_pts = 20.0 if pnl > 0 else (10.0 if pnl == 0 else 0.0)
    # Sample-size points: linear up to 30 closed bets.
    sample_pts = min(20.0, n / 30.0 * 20.0)
    # Calibration: look at the worst-magnitude gap in any populated bucket.
    pop_cal = [bk for bk in cal if bk["n"] > 0]
    if pop_cal:
        worst = max(abs(bk["actual"] - bk["predicted"]) for bk in pop_cal)
        cal_pts = max(0.0, 20.0 - worst * 100.0)  # 0pp gap=20, 20pp gap=0
    else:
        cal_pts = 0.0
    # EV-vs-realized: small absolute gap = high score.
    realized = (pnl / 100.0)
    expected = sum(float(b.get("expected_ev_at_entry") or 0)
                   * (b.get("contracts") or 0)
                   for b in history if b.get("expected_ev_at_entry") is not None)
    if cost > 0:
        ev_gap = abs(realized - expected) / max(1.0, cost / 100.0)
        ev_pts = max(0.0, 20.0 - ev_gap * 50.0)
    else:
        ev_pts = 0.0
    # Drawdown: walk cumulative P&L, max peak-to-trough as fraction of total cost.
    cum = 0.0
    peak = 0.0
    max_dd = 0.0
    sorted_hist = sorted(history, key=lambda b: b.get("exited_at") or "")
    for b in sorted_hist:
        cum += (b.get("realized_pnl_cents") or 0) / 100.0
        peak = max(peak, cum)
        max_dd = max(max_dd, peak - cum)
    if cost > 0:
        dd_frac = max_dd / max(1.0, cost / 100.0)
        dd_pts = max(0.0, 20.0 - dd_frac * 100.0)
    else:
        dd_pts = 0.0
    score = int(round(pnl_pts + sample_pts + cal_pts + ev_pts + dd_pts))
    score = max(0, min(100, score))
    cls = ("green" if score >= 70 else
           ("yellow" if score >= 45 else "red"))
    breakdown = [
        ("Realized P&L", f"${pnl_dollars:+.2f}"),
        ("Sample size", f"{n} closed"),
        ("Calibration pts", f"{cal_pts:.0f}/20"),
        ("EV match pts", f"{ev_pts:.0f}/20"),
        ("Drawdown pts", f"{dd_pts:.0f}/20"),
    ]
    return score, cls, breakdown


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
               f"in this series, with strikes from <b>${lo:.3f}</b> to "
               f"<b>${hi:.3f}</b>.")
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
                # Local model_snapshots fallback for the chart only
                # when Kalshi creds aren't configured. 7 days covers
                # any weekly contract.
                underlying_history = fetch_underlying_history(
                    db_path, hours=7 * 24,
                )
                # Kalshi-derived underlying via strike-ladder interpolation
                # — same data source Kalshi uses for the chart at the top
                # of every market page. Lookback auto-sized to span the
                # full life of whichever event is currently open. When
                # that event resolves and Kalshi opens the next one, the
                # chart and ATM market roll over automatically.
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
                global_summary = fetch_global_summary(self.bots)
                global_active_bets: List[dict] = []
                global_history: List[dict] = []
                for b in self.bots:
                    if not b.get("available"):
                        continue
                    if b.get("dashboard_type") and b["dashboard_type"] != "standard":
                        continue
                    for ab in fetch_active_bets_with_marks(b["db_path"]):
                        ab["_bot_name"] = b["name"]
                        global_active_bets.append(ab)
                    for h in fetch_bet_history(b["db_path"], limit=50):
                        h["_bot_name"] = b["name"]
                        global_history.append(h)
                global_active_bets.sort(key=lambda x: x.get("opened_at", ""), reverse=True)
                global_history.sort(key=lambda x: x.get("exited_at", ""), reverse=True)

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
                if bot.get("dashboard_type") == "whale":
                    # Whale page uses meta-refresh, not the JS poller.
                    # Return a minimal stub so any client polling this
                    # endpoint gets a clean 200.
                    payload_dict = {"bot": bot["key"], "type": "whale"}
                else:
                    db_path = bot["db_path"]
                    payload_dict = build_snapshot(db_path, self.bots,
                                                   self.edge_cfg)
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
