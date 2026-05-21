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
import csv
import html
import json
import logging
import math
import re
import sqlite3
import sys
import time
from contextlib import closing
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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

    Also tolerates files that aren't SQLite at all (DatabaseError) —
    e.g. if a registry entry's db_path ever points at a JSONL.
    """
    if not Path(db_path).exists():
        return []
    try:
        with closing(_conn(db_path)) as c:
            return [dict(r) for r in c.execute(query, params).fetchall()]
    except (sqlite3.OperationalError, sqlite3.DatabaseError):
        return []


def fetch_bot_effective_config(db_path: str) -> dict | None:
    """Read ``<data_dir>/effective_config.json`` next to the bot's sim.db.

    Bots that use ``kalshi_sdk.write_effective_config`` at startup emit
    this file with their *resolved* gates (post env-overrides, post
    per-bot widens). When present, the dashboard's buy-criteria panel
    renders these instead of the dashboard YAML's display defaults —
    so what the panel claims is what the bot actually applies.

    Returns ``None`` when the file is missing or unreadable; the caller
    falls back to the dashboard YAML and marks the panel as "showing
    dashboard defaults — bot has not reported its live config".

    Tolerant of the bot writing JSON elsewhere — we also try the
    sibling ``effective_config.json`` alongside the watchlist.json that
    sport bots use as their data anchor.
    """
    if not db_path:
        return None
    candidates = [Path(db_path).parent / "effective_config.json"]
    try:
        for p in candidates:
            if p.exists():
                with open(p) as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    return data
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    return None


def resolve_bot_thresholds(
    bot: dict,
    fallback_edge: dict,
    fallback_validators: dict,
    fallback_risk: dict,
    fallback_hedge: dict,
) -> Tuple[dict, dict, dict, dict, dict]:
    """Build the per-bot edge/validators/risk/hedge dicts the renderer uses.

    Priority is the bot's live ``effective_config.json`` (written at
    startup by ``kalshi_sdk.write_effective_config``); missing fields
    fall back to the dashboard YAML so a partially-reported config
    still looks complete to the user.

    Returns ``(edge, validators, risk, hedge, source_meta)`` where
    ``source_meta`` carries provenance the rules modal renders:

        {"source": "live" | "fallback",
         "captured_at": "..." | None,
         "missing_keys": [...]}    # fields the bot didn't report
    """
    db_path = bot.get("db_path") or bot.get("watchlist_json_path") or ""
    live = fetch_bot_effective_config(db_path)
    if not live:
        return (
            dict(fallback_edge or {}),
            dict(fallback_validators or {}),
            dict(fallback_risk or {}),
            dict(fallback_hedge or {}),
            {"source": "fallback", "captured_at": None, "missing_keys": []},
        )
    missing: List[str] = []

    def _merge(live_section: Any, fallback: dict, section_label: str) -> dict:
        merged = dict(fallback or {})
        live_dict = live_section if isinstance(live_section, dict) else {}
        for k, v in live_dict.items():
            if v is not None:
                merged[k] = v
        # Track keys the dashboard expected but the bot didn't report —
        # surfaced in the modal so the user knows the panel is mixed.
        for k in (fallback or {}):
            if k not in live_dict or live_dict.get(k) is None:
                missing.append(f"{section_label}.{k}")
        return merged

    edge = _merge(live.get("edge"), fallback_edge, "edge")
    validators = _merge(live.get("validators"), fallback_validators, "validators")
    risk = _merge(live.get("risk"), fallback_risk, "risk")
    hedge = _merge(live.get("hedge"), fallback_hedge, "hedge")
    return (
        edge, validators, risk, hedge,
        {
            "source": "live",
            "captured_at": live.get("captured_at"),
            "missing_keys": missing,
        },
    )


def fetch_summary(db_path: str, period_days: int | None = None) -> dict:
    """Lifetime + recent stats used by the Summary section.

    ``period_days`` filters the period-scoped fields (period_bets_made,
    period_net_pnl_cents, period_wins, period_losses) to bets that
    opened (for bets_made) or closed (for P&L/wins/losses) within the
    last N days. None → lifetime.
    """
    empty = {
        "total_bets": 0, "open_count": 0, "exposure_cents": 0,
        "active_contracts": 0,
        "closed_count": 0, "realized_pnl_cents": 0,
        "wins_lifetime": 0, "losses_lifetime": 0,
        "avg_win_cents": 0, "avg_loss_cents": 0,
        "bets_today": 0, "this_week_pnl_cents": 0,
        "biggest_win_cents": 0, "biggest_loss_cents": 0,
        "period_bets_made": 0, "period_net_pnl_cents": 0,
        "period_wins": 0, "period_losses": 0,
        "period_money_spent_cents": 0,
        "period_money_gained_cents": 0,
        "period_contracts_bought": 0,
        "potential_gain_cents": 0,
        "active_money_spent_cents": 0,
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
            # Active-bets count + active-contracts sum: filter out
            # zombie open positions whose ticker has already settled
            # (>60 min past close). Mirrors the same guard the
            # active-bets table applies (``hide_settled``) so the
            # headline cards agree with what's rendered in the table.
            open_rows = c.execute(
                "SELECT p.ticker, p.contracts, p.entry_price_cents, "
                "  (SELECT mv.minutes_to_close FROM market_views mv "
                "     WHERE mv.ticker = p.ticker "
                "     ORDER BY mv.id DESC LIMIT 1) AS mtc "
                "FROM positions p WHERE p.status = 'open'"
            ).fetchall()
            active_count = 0
            active_contracts = 0
            active_money_spent_cents = 0
            active_potential_gain_cents = 0
            for r in open_rows:
                mtc = r["mtc"]
                if mtc is None:
                    mtc = minutes_to_close_from_ticker(r["ticker"])
                if (mtc if mtc is not None else 0) >= -60:
                    active_count += 1
                    ctr = r["contracts"] or 0
                    entry_c = r["entry_price_cents"] or 0
                    active_contracts += ctr
                    fee_c = kalshi_fee_cents(entry_c, ctr)
                    active_money_spent_cents += entry_c * ctr + fee_c
                    active_potential_gain_cents += (100 - entry_c) * ctr - fee_c
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
            # Potential gain + active money-spent are computed inline
            # from the open-rows loop above (matches the active-bets
            # table renderer cell-for-cell: subtracts Kalshi entry fee
            # on potential, includes entry fee in cost, and applies the
            # same hide-settled mtc≥−60 filter).
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
                contracts_row = c.execute(
                    "SELECT COALESCE(SUM(contracts), 0) v "
                    "FROM positions WHERE status = 'closed'"
                ).fetchone()
                period_money_spent = spent_row["v"] or 0
                period_money_gained = gained_row["v"] or 0
                period_contracts_bought = contracts_row["v"] or 0
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
                contracts_row = c.execute(
                    "SELECT COALESCE(SUM(contracts), 0) v "
                    "FROM positions WHERE status = 'closed' "
                    "  AND date(exited_at) >= date('now', ?)",
                    (period_window,),
                ).fetchone()
                period_money_spent = spent_row["v"] or 0
                period_money_gained = gained_row["v"] or 0
                period_contracts_bought = contracts_row["v"] or 0
    except (sqlite3.OperationalError, sqlite3.DatabaseError):
        return empty
    return {
        "total_bets": int(total["n"] or 0),
        "open_count": int(active_count),
        "exposure_cents": int(open_row["exp"] or 0),
        "active_contracts": int(active_contracts),
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
        "period_contracts_bought": int(period_contracts_bought or 0),
        "potential_gain_cents": int(active_potential_gain_cents),
        "active_money_spent_cents": int(active_money_spent_cents),
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
    on the gas-prices simulator schema; bots without it still appear in
    the cross-bot Summary with an empty Gas-at-close cell. floor_strike
    + cap_strike are pulled via subqueries on market_views so the bet-
    history view can render the human Question text per row.
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
                # decision_json is the catch-all entry-time payload
                # every bot writes. Natural-gas (and any older bot
                # schema that doesn't have dedicated model_yes_prob_
                # at_entry / kalshi_yes_prob_at_entry columns)
                # stashes ``model_prob`` + ``kalshi_implied_prob``
                # in here — we parse them out below so the History
                # tab's Model-p cell is populated regardless of which
                # schema the bot ships with.
                "decision_json",
                # Set to N > 1 by the same-ticker dedupe pass when
                # multiple flap-trades on the same (ticker, side) were
                # collapsed into this row. The History renderer surfaces
                # a "×N" badge so the user can tell merged rows apart.
                "merged_trade_count",
                "merged_position_ids",
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
    out = [dict(r) for r in rows]
    # Backfill model_yes_prob_at_entry / kalshi_yes_prob_at_entry from
    # decision_json for bots that don't have dedicated columns (the
    # natural-gas bot's older schema is the case in production).
    for h in out:
        if h.get("model_yes_prob_at_entry") is not None:
            continue
        raw = h.get("decision_json")
        if not raw:
            continue
        try:
            payload = json.loads(raw) if isinstance(raw, str) else raw
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        mp = payload.get("model_prob")
        kp = payload.get("kalshi_implied_prob")
        if mp is not None and h.get("model_yes_prob_at_entry") is None:
            try:
                h["model_yes_prob_at_entry"] = float(mp)
            except (TypeError, ValueError):
                pass
        if kp is not None and h.get("kalshi_yes_prob_at_entry") is None:
            try:
                h["kalshi_yes_prob_at_entry"] = float(kp)
            except (TypeError, ValueError):
                pass
    return out


def fetch_ev_realized_buckets(db_path: str) -> List[dict]:
    """Edge-vs-realized bucket analysis for the Models tab.

    For every closed position with both ``expected_ev_at_entry`` and
    ``realized_pnl_cents`` recorded, bucket by the predicted EV (in
    decimal $/contract) and compute count / mean predicted EV / mean
    realized ¢-per-contract / win rate / total P&L per bucket.

    Returns ``[]`` for bots whose schema doesn't carry
    ``expected_ev_at_entry`` (older / tennis-style bots). The caller
    decides whether to render the section.
    """
    if not Path(db_path).exists():
        return []
    try:
        with closing(_conn(db_path)) as c:
            cols = {r["name"] for r in
                    c.execute("PRAGMA table_info(positions)").fetchall()}
            if "expected_ev_at_entry" not in cols:
                return []
            rows = c.execute(
                "SELECT expected_ev_at_entry, realized_pnl_cents, contracts "
                "FROM positions "
                "WHERE status = 'closed' "
                "  AND expected_ev_at_entry IS NOT NULL "
                "  AND realized_pnl_cents IS NOT NULL "
                "  AND contracts > 0"
            ).fetchall()
    except (sqlite3.OperationalError, sqlite3.DatabaseError):
        return []
    # Bucket edges in decimal $ per contract. The first bucket catches
    # bets that slipped in below the EV gate (older data, or rounding);
    # the rest follow the 2¢ / 4¢ / 7¢ / 10¢ ladder we discussed.
    buckets: List[Tuple[str, float, float]] = [
        ("< 0¢",   -10.0,  0.0),
        ("0–2¢",    0.0,   0.02),
        ("2–4¢",    0.02,  0.04),
        ("4–7¢",    0.04,  0.07),
        ("7–10¢",   0.07,  0.10),
        ("10¢+",    0.10,  10.0),
    ]
    out: List[dict] = []
    for label, lo, hi in buckets:
        n = 0
        ev_sum = 0.0
        realized_per_sum = 0.0
        wins = 0
        total_pnl_cents = 0
        for r in rows:
            try:
                ev = float(r["expected_ev_at_entry"])
                pnl = int(r["realized_pnl_cents"])
                contracts = int(r["contracts"])
            except (TypeError, ValueError):
                continue
            if contracts <= 0:
                continue
            if ev < lo or ev >= hi:
                continue
            n += 1
            ev_sum += ev
            realized_per_sum += pnl / contracts
            total_pnl_cents += pnl
            if pnl > 0:
                wins += 1
        out.append({
            "label": label,
            "count": n,
            "predicted_ev_cents": (ev_sum / n) * 100.0 if n else None,
            "realized_per_contract_cents": (realized_per_sum / n
                                              if n else None),
            "win_rate": (wins / n) if n else None,
            "total_pnl_cents": total_pnl_cents,
        })
    return out


def fetch_hedge_audit(db_path: str) -> dict:
    """Was the hedge worth it?

    For every closed position whose ``error_type`` starts with
    ``hedge_`` (profit-lock or stop-loss), look up the same ticker's
    eventual settlement from ``market_views`` and compute the
    counterfactual P&L of holding to settlement. Aggregate across
    all such bets so the user can see whether hedges have net saved
    money or net cost money.

    Settlement is inferred from the latest market_views row for the
    ticker captured AFTER the hedge exit: YES settled when the final
    ``yes_ask_cents`` is ≥ 95, NO when ≤ 5. Anything else is treated
    as "unknown" and excluded from the counterfactual sums (still
    counted in totals).

    Returns a dict the renderer can shape into a small summary card.
    Empty (zero hedged) when no hedge-exited bets exist or the DB
    lacks the schema.
    """
    blank = {"n_hedged": 0, "n_with_settlement": 0,
             "actual_pnl_cents": 0, "counterfactual_pnl_cents": 0,
             "delta_cents": 0,
             "n_hedge_saved": 0, "n_hedge_cost": 0}
    if not Path(db_path).exists():
        return blank
    try:
        with closing(_conn(db_path)) as c:
            cols = {r["name"] for r in
                    c.execute("PRAGMA table_info(positions)").fetchall()}
            if "error_type" not in cols:
                return blank
            rows = c.execute(
                "SELECT id, ticker, side, entry_price_cents, contracts, "
                "       realized_pnl_cents, exited_at "
                "FROM positions "
                "WHERE status = 'closed' "
                "  AND error_type LIKE 'hedge_%' "
                "  AND realized_pnl_cents IS NOT NULL "
                "  AND entry_price_cents IS NOT NULL"
            ).fetchall()
            results: List[dict] = []
            for r in rows:
                # Settlement signal: latest market_views row for the
                # ticker captured AFTER the position exited. Extreme
                # yes_ask (≥95 or ≤5) ⇒ the contract resolved one way.
                settle = c.execute(
                    "SELECT yes_ask_cents FROM market_views "
                    "WHERE ticker = ? AND captured_at > ? "
                    "  AND yes_ask_cents IS NOT NULL "
                    "ORDER BY id DESC LIMIT 1",
                    (r["ticker"], r["exited_at"] or ""),
                ).fetchone()
                results.append({**dict(r),
                                "settle_yes": (dict(settle)["yes_ask_cents"]
                                                if settle else None)})
    except (sqlite3.OperationalError, sqlite3.DatabaseError):
        return blank
    if not results:
        return blank
    actual_total = 0
    counter_total = 0
    n_known = 0
    n_saved = 0
    n_cost = 0
    for r in results:
        actual_total += int(r["realized_pnl_cents"] or 0)
        settle_yes = r["settle_yes"]
        if settle_yes is None:
            continue
        if settle_yes >= 95:
            yes_won = True
        elif settle_yes <= 5:
            yes_won = False
        else:
            continue
        side = (r["side"] or "").upper()
        side_won = (yes_won if side == "YES" else not yes_won)
        entry = int(r["entry_price_cents"] or 0)
        contracts = int(r["contracts"] or 0)
        per_contract = (100 - entry) if side_won else (-entry)
        counter = per_contract * contracts
        counter_total += counter
        delta = int(r["realized_pnl_cents"] or 0) - counter
        # Positive delta ⇒ hedge made us money relative to holding
        # (we exited at a better price than the eventual settlement).
        if delta > 0:
            n_saved += 1
        elif delta < 0:
            n_cost += 1
        n_known += 1
    return {
        "n_hedged": len(results),
        "n_with_settlement": n_known,
        "actual_pnl_cents": actual_total,
        "counterfactual_pnl_cents": counter_total,
        "delta_cents": actual_total - counter_total,
        "n_hedge_saved": n_saved,
        "n_hedge_cost": n_cost,
    }


# Fallback bankroll used when neither display.bankroll_cents nor a
# live Kalshi balance is available. The Kelly sizing column on the
# watchlist multiplies half-Kelly fractions against this — change it
# via display.bankroll_cents in dashboard.yaml if a particular bot
# deserves a bigger or smaller stake size, or set the Kalshi API
# creds on the dashboard host to drive Size off the real balance.
DEFAULT_BANKROLL_CENTS = 100_000  # $1,000 fallback


def _resolve_bankroll(display: dict | None
                       ) -> tuple[int, str]:
    """Pick the Kelly-sizing bankroll for one watchlist render.

    Priority:
      1) ``display['bankroll_cents']`` — per-bot YAML override.
      2) Live Kalshi /portfolio/balance — cached 60s. Drives Size off
         the real account balance the user can actually deploy.
      3) ``DEFAULT_BANKROLL_CENTS`` — fallback when no creds or the
         Kalshi balance call fails.

    Returns ``(cents, human_readable_source_string)``. The source
    string is used inside Size-cell tooltips so the user can see at
    a glance *why* a particular Size value was suggested.
    """
    # Per-bot override wins unconditionally. Lets us pin a bot to a
    # smaller stake (or a paper amount) without touching the live
    # account balance for the rest of the dashboard.
    override = (display or {}).get("bankroll_cents")
    if override is not None:
        try:
            cents = int(override)
            return cents, f"${cents/100:,.0f} (bot override)"
        except (TypeError, ValueError):
            pass

    try:
        from . import kalshi_client
        live_cents, err = kalshi_client.get_balance_cents()
    except Exception as exc:  # noqa: BLE001
        log.warning("kalshi balance lookup raised: %s", exc)
        live_cents, err = None, str(exc)

    if live_cents is not None:
        return live_cents, f"${live_cents/100:,.2f} Kalshi balance"

    return DEFAULT_BANKROLL_CENTS, (
        f"${DEFAULT_BANKROLL_CENTS/100:,.0f} default "
        f"({err or 'no live balance'})"
    )


def kelly_contracts(price_cents: float | int | None,
                     win_prob: float | None,
                     bankroll_cents: int,
                     fraction: float = 0.5) -> int:
    """Half-Kelly suggested contract count for one side of a Kalshi
    binary contract.

    Kelly fraction f* = (b·p − q) / b   where
        b = decimal payout odds = (100 − price) / price
        p = win probability for the side we're buying
        q = 1 − p
    The dashboard plays it conservative: ``fraction`` defaults to 0.5
    so the suggestion is half-Kelly. Returns 0 when the bet has no
    positive Kelly fraction (no edge after fees model would be a
    further haircut on top — caller can layer that later).
    """
    if price_cents is None or win_prob is None:
        return 0
    try:
        price = float(price_cents)
        p = float(win_prob)
    except (TypeError, ValueError):
        return 0
    if price <= 0 or price >= 100:
        return 0
    q = 1.0 - p
    b = (100.0 - price) / price
    if b <= 0:
        return 0
    kf = (b * p - q) / b
    if kf <= 0:
        return 0
    capital = bankroll_cents * kf * fraction
    n = int(capital / price)  # price already in cents = cents per contract
    return max(0, n)


def bot_regime_status(db_path: str, days: int = 90,
                        min_bets: int = 10) -> dict:
    """Rolling edge-health check used by the Home-tab bot cards.

    Looks at the last ``days`` of closed bets with both
    ``expected_ev_at_entry`` and ``realized_pnl_cents`` recorded, then
    grades how well the realized cents-per-contract is tracking the
    predicted EV. Returns a small dict the renderer turns into a
    status pill.

    Statuses
    --------
    ``"green"``  — realized ≥ predicted × 0.5 (edge largely survived)
    ``"yellow"`` — realized > 0 but well below predicted (eroding)
    ``"red"``    — realized ≤ 0 where predicted was positive (anti-edge)
    ``"gray"``   — not enough data (or schema lacks the EV column)
    """
    empty = {"status": "gray", "label": "no data",
             "reason": "Needs closed bets with a recorded entry EV."}
    if not Path(db_path).exists():
        return empty
    try:
        with closing(_conn(db_path)) as c:
            cols = {r["name"] for r in
                    c.execute("PRAGMA table_info(positions)").fetchall()}
            if "expected_ev_at_entry" not in cols:
                return empty
            rows = c.execute(
                "SELECT expected_ev_at_entry, realized_pnl_cents, contracts "
                "FROM positions WHERE status = 'closed' "
                "  AND expected_ev_at_entry IS NOT NULL "
                "  AND realized_pnl_cents IS NOT NULL "
                "  AND contracts > 0 "
                "  AND date(exited_at) >= date('now', ?)",
                (f"-{int(days)} days",),
            ).fetchall()
    except (sqlite3.OperationalError, sqlite3.DatabaseError):
        return empty
    if len(rows) < min_bets:
        return {
            "status": "gray",
            "label": "warming up",
            "reason": (f"Only {len(rows)} closed bets in the last "
                       f"{days}d — need ≥ {min_bets} for a reading."),
        }
    pred_sum_cents = 0.0
    realized_per_sum = 0.0
    total_pnl_cents = 0
    for r in rows:
        try:
            ev = float(r["expected_ev_at_entry"])
            pnl = int(r["realized_pnl_cents"])
            contracts = int(r["contracts"])
        except (TypeError, ValueError):
            continue
        if contracts <= 0:
            continue
        pred_sum_cents += ev * 100.0
        realized_per_sum += pnl / contracts
        total_pnl_cents += pnl
    n = len(rows)
    pred = pred_sum_cents / n
    realized = realized_per_sum / n
    # Edge-survival rule mirrors the EV-vs-realized table: realized
    # within half of predicted (or within 1¢ at tiny predicted EV)
    # is "the edge held". Negative realized with positive predicted is
    # the anti-edge regime that warrants pausing the bot.
    if pred <= 0:
        status = "green" if realized > 0 else "red"
        label = "edge confirmed" if status == "green" else "anti-edge"
    elif realized >= max(pred * 0.5, pred - 1.0):
        status, label = "green", "edge confirmed"
    elif realized > 0:
        status, label = "yellow", "edge eroding"
    else:
        status, label = "red", "anti-edge"
    reason = (f"{n} bets · predicted {pred:+.1f}¢/contract · "
              f"realized {realized:+.1f}¢/contract · "
              f"net {'+' if total_pnl_cents > 0 else ('−' if total_pnl_cents < 0 else '')}"
              f"${abs(total_pnl_cents)/100:.2f} over {days}d")
    return {"status": status, "label": label, "reason": reason,
            "predicted_cents": pred, "realized_cents": realized,
            "total_pnl_cents": total_pnl_cents, "n_bets": n,
            "days": days}


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
        "active_contracts": 0,       # always current — sum of contracts open
        "active_bots": 0,            # distinct bots with at least one active bet
        "period_bets_made": 0,
        "period_net_pnl_cents": 0,
        "period_wins": 0,
        "period_losses": 0,
        "period_money_spent_cents": 0,
        "period_money_gained_cents": 0,
        "period_contracts_bought": 0,
        "potential_gain_cents": 0,    # always current — fee-subtracted
        "active_money_spent_cents": 0,  # entry cost of currently-open bets only
        "this_week_pnl_cents": 0,     # always lifetime-of-last-7-days
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
        # Tennis bot keeps its paper-trade ledger in a JSON file rather
        # than a sim.db; we map it onto the same dict shape via
        # ``tennis.summary_for_rollup`` so the cross-bot summary cards
        # at the top of the home page DO include tennis volume + P&L.
        if b.get("dashboard_type") == "tennis":
            from . import tennis as _tennis
            s = _tennis.summary_for_rollup(b.get("sim_state_path"))
        elif b.get("dashboard_type") == "survivor":
            from . import survivor as _survivor
            s = _survivor.summary_for_rollup(b.get("sim_state_path"))
        elif b.get("dashboard_type") == "billboard":
            from . import billboard as _billboard
            s = _billboard.summary_for_rollup(b.get("sim_state_path"))
        elif b.get("dashboard_type") and b["dashboard_type"] != "standard":
            continue
        else:
            s = fetch_summary(b["db_path"], period_days=period_days)
        rollup["active_bets"] += s.get("open_count", 0)
        rollup["active_contracts"] += s.get("active_contracts", 0)
        if s.get("open_count", 0) > 0:
            rollup["active_bots"] += 1
        rollup["period_bets_made"] += s.get("period_bets_made", 0)
        rollup["period_net_pnl_cents"] += s.get("period_net_pnl_cents", 0)
        rollup["period_wins"] += s.get("period_wins", 0)
        rollup["period_losses"] += s.get("period_losses", 0)
        rollup["period_money_spent_cents"] += s.get("period_money_spent_cents", 0)
        rollup["period_money_gained_cents"] += s.get("period_money_gained_cents", 0)
        rollup["period_contracts_bought"] += s.get("period_contracts_bought", 0)
        rollup["potential_gain_cents"] += s.get("potential_gain_cents", 0)
        rollup["active_money_spent_cents"] += s.get("active_money_spent_cents", 0)
        rollup["this_week_pnl_cents"] += s.get("this_week_pnl_cents", 0)
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


def _compute_active_bets_totals(active_bets: List[dict],
                                  hide_settled: bool = True) -> dict:
    """Compute the Active-bets headline values directly from the same
    list that feeds ``_render_active_bets_table`` — guarantees the
    Home-tab cards match the table column totals cell-for-cell
    regardless of how each bot's data is sourced.

    Returns dict with: ``open_count``, ``active_contracts``,
    ``active_bots`` (distinct bots in the table), ``active_money_spent_cents``
    (sum of Entry cost column, = entry × contracts + Kalshi fee per row),
    ``potential_gain_cents`` (sum of Potential gain column, = (100 −
    entry) × contracts − Kalshi fee per row).
    """
    if hide_settled:
        active_bets = [
            b for b in active_bets
            if (
                (b.get("minutes_to_close")
                 if b.get("minutes_to_close") is not None
                 else minutes_to_close_from_ticker(b.get("ticker"))) or 0
            ) >= -60
        ]
    contracts_total = 0
    money_spent_cents = 0
    potential_gain_cents = 0
    bots_in_table = set()
    for b in active_bets:
        entry = b.get("entry_price_cents") or 0
        ctr = b.get("contracts") or 0
        fee_c = kalshi_fee_cents(entry, ctr)
        contracts_total += ctr
        money_spent_cents += entry * ctr + fee_c
        potential_gain_cents += (100 - entry) * ctr - fee_c
        bk = b.get("_bot_key") or b.get("_bot_name") or ""
        if bk:
            bots_in_table.add(bk)
    return {
        "open_count": len(active_bets),
        # ``active_bets`` mirrors ``open_count`` so callers that
        # ``summary.update(...)`` this dict also overwrite the rollup's
        # ``active_bets`` field — the card value the Home tab renders
        # for "Active bets". Without this, the card kept the un-filtered
        # per-bot sum (which counts zombie positions whose ticker close
        # time is already past) and disagreed with the table beneath it.
        "active_bets": len(active_bets),
        "active_contracts": int(contracts_total),
        "active_bots": len(bots_in_table),
        "active_money_spent_cents": int(money_spent_cents),
        "potential_gain_cents": int(potential_gain_cents),
    }


def _build_global_active_bets(bots: List[dict]) -> List[dict]:
    """Cross-bot list of active-bet dicts in the shape the active-bets
    table renderer (and ``_compute_active_bets_totals``) expects.
    Tagged with ``_bot_key`` so the distinct-bots count works. Skips
    bots whose data source isn't available, matching the page-render
    bot iteration.
    """
    out: List[dict] = []
    for b in bots:
        if not b.get("available"):
            continue
        dt = b.get("dashboard_type") or "standard"
        if dt == "tennis":
            from . import tennis as _tennis
            rows = _tennis.active_bets_for_rollup(
                b.get("sim_state_path"),
                watchlist_path=b.get("watchlist_json_path"))
        elif dt in ("survivor", "billboard"):
            continue  # advisory bots — no positions
        elif dt != "standard":
            continue
        else:
            rows = fetch_active_bets_with_marks(b["db_path"])
        for ab in rows:
            ab["_bot_key"] = b["key"]
            ab["_bot_name"] = b["name"]
            out.append(ab)
    return out


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
        # Direction: prefer Kalshi's strike_type field — it's
        # authoritative ("greater" → above, "less" → below, "between"
        # → between). For above-style markets Kalshi sets BOTH
        # floor_strike and cap_strike to the same value (a convention,
        # not a range), so the old "both set ⇒ between" heuristic
        # mis-renders them as a zero-width band like "0.90pp – 0.90pp".
        # Fall back to that heuristic only when strike_type is absent.
        strike_type = (m.get("strike_type") or "").lower()
        if strike_type in ("greater", "above"):
            direction = "above"
            strike_high = None
        elif strike_type in ("less", "below"):
            direction = "below"
            strike_high = None
        elif strike_type == "between":
            direction = "between"
        elif strike_low is not None and strike_high is not None and strike_low != strike_high:
            direction = "between"
        else:
            direction = "above"
            strike_high = None
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


def pick_recent_market_view_ticker(db_path: str) -> str | None:
    """Most-recently-updated ticker in ``market_views`` with a non-null
    yes_ask_cents. Used as a robust chart-ticker fallback when neither
    the active bet nor Kalshi's ATM market produces a ticker that has
    fresh data locally (e.g. a bot mid-event-rollover).
    """
    if not Path(db_path).exists():
        return None
    try:
        with closing(_conn(db_path)) as c:
            row = c.execute(
                "SELECT ticker FROM market_views "
                "WHERE yes_ask_cents IS NOT NULL "
                "ORDER BY id DESC LIMIT 1"
            ).fetchone()
    except (sqlite3.OperationalError, sqlite3.DatabaseError):
        return None
    return (dict(row).get("ticker") if row else None)


def fetch_ticker_yes_prob_history(db_path: str, ticker: str | None,
                                    hours: int = 168) -> List[dict]:
    """Time-series of YES probability (in cents, 0-100) for one ticker.

    Reads ``market_views`` rows for the ticker over the last N hours.
    Each row is a snapshot the bot took at scoring time, so the chart
    point density mirrors how often the bot polls.

    Returns ``[{"ts": unix_seconds, "value": yes_ask_cents}, …]``.
    Empty when the DB / ticker has no data.
    """
    if not ticker or not Path(db_path).exists():
        return []
    try:
        with closing(_conn(db_path)) as c:
            rows = c.execute(
                "SELECT captured_at, yes_ask_cents AS value "
                "FROM market_views "
                "WHERE ticker = ? AND yes_ask_cents IS NOT NULL "
                "  AND captured_at >= datetime('now', ?) "
                "ORDER BY captured_at ASC",
                (ticker, f"-{int(hours)} hours"),
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
    # Override the active-bets headline fields with values computed
    # directly from the global active-bets list — guarantees the cards
    # equal the table column totals cell-for-cell, even when a bot's
    # ``summary_for_rollup`` shape drifts.
    summary.update(_compute_active_bets_totals(_build_global_active_bets(bots)))
    watchlist = fetch_watchlist(db_path)
    active_bets = fetch_active_bets_with_marks(db_path)

    def _ev_yes(p, ya_c, spread_c):
        if p is None or ya_c is None:
            return None
        fee_d = kalshi_fee_cents(ya_c, 1) / 100.0
        return (float(p) - (ya_c / 100.0)
                - ((spread_c or 0) / 200.0) - fee_d)
    def _ev_no(p, na_c, spread_c):
        if p is None or na_c is None:
            return None
        fee_d = kalshi_fee_cents(na_c, 1) / 100.0
        return ((1.0 - float(p)) - (na_c / 100.0)
                - ((spread_c or 0) / 200.0) - fee_d)

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
            "active_contracts": summary.get("active_contracts"),
            "active_bots": summary.get("active_bots"),
            "period_closed_bets": period_closed,
            "period_money_spent_cents": summary.get("period_money_spent_cents"),
            "period_money_gained_cents": summary.get("period_money_gained_cents"),
            "active_money_spent_cents": summary.get("active_money_spent_cents"),
            "potential_gain_cents": summary.get("potential_gain_cents"),
            "period_net_pnl_cents": summary.get("period_net_pnl_cents"),
            "period_win_pct": summary.get("period_win_pct"),
            "period_has_closed": period_closed > 0,
            "this_week_pnl_cents": summary.get("this_week_pnl_cents"),
            "net_pnl_cents": summary.get("net_pnl_cents"),
        },
        "watchlist": rows,
        "active_bets": actives,
        "min_ev": min_ev,
        "captured_at": datetime.now(timezone.utc).isoformat(),
    }


def _tennis_like_snapshot(
    watchlist_rows: List[dict], active_bets: List[dict],
    bots: List[dict], *, edge_cfg: dict,
    period_days: int | None,
) -> dict:
    """Shape a snapshot for tennis-shape (JSON-source) bots.

    The watchlist + active bets come pre-adapted via
    ``tennis.build_standard_watchlist_rows`` and
    ``tennis.active_bets_for_rollup`` — both already produce the
    standard row schema. Cross-bot summary fields come from the same
    rollup the sim.db bots use so the Home tab cards stay live.
    """
    summary = fetch_global_summary(bots, period_days=period_days)
    # Same override as the standard snapshot — see ``build_snapshot``.
    summary.update(_compute_active_bets_totals(_build_global_active_bets(bots)))
    rows = []
    for v in watchlist_rows:
        ya = v.get("yes_ask_cents")
        na = v.get("no_ask_cents")
        sp = v.get("spread_cents") or 0
        p = v.get("model_prob_yes")
        ev_yes = None
        ev_no = None
        if p is not None and ya is not None:
            fee_yes_d = kalshi_fee_cents(ya, 1) / 100.0
            ev_yes = (float(p) - (ya / 100.0)
                      - (sp / 200.0) - fee_yes_d)
        if p is not None and na is not None:
            fee_no_d = kalshi_fee_cents(na, 1) / 100.0
            ev_no = ((1.0 - float(p)) - (na / 100.0)
                     - (sp / 200.0) - fee_no_d)
        rows.append({
            "ticker": v.get("ticker"),
            "kalshi_yes": ya, "kalshi_no": na,
            "spread": v.get("spread_cents"),
            "volume": v.get("volume"),
            "open_interest": v.get("open_interest"),
            "minutes_to_close": v.get("minutes_to_close"),
            "model_prob_yes": p,
            "raw_model_prob_yes": v.get("raw_model_prob_yes"),
            "ev_yes": ev_yes, "ev_no": ev_no,
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
            "unreal_pnl_cents": unrealized_pnl_cents(ab) if
                ab.get("entry_price_cents") is not None
                and ab.get("contracts") is not None
                and ab.get("side") else None,
        })
    period_closed = (summary.get("period_wins", 0)
                     + summary.get("period_losses", 0))
    return {
        "summary": {
            "active_bets": summary.get("active_bets"),
            "active_contracts": summary.get("active_contracts"),
            "active_bots": summary.get("active_bots"),
            "period_closed_bets": period_closed,
            "period_money_spent_cents": summary.get("period_money_spent_cents"),
            "period_money_gained_cents": summary.get("period_money_gained_cents"),
            "active_money_spent_cents": summary.get("active_money_spent_cents"),
            "potential_gain_cents": summary.get("potential_gain_cents"),
            "period_net_pnl_cents": summary.get("period_net_pnl_cents"),
            "period_win_pct": summary.get("period_win_pct"),
            "period_has_closed": period_closed > 0,
            "this_week_pnl_cents": summary.get("this_week_pnl_cents"),
            "net_pnl_cents": summary.get("net_pnl_cents"),
        },
        "watchlist": rows,
        "active_bets": actives,
        "min_ev": edge_cfg.get("min_ev_per_contract", 0.03),
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
    sign = "−" if c < 0 else ""
    return f"{sign}${abs(c)/100:.2f}"


def _favicon_link() -> str:
    """Return a `<link rel="icon">` tag with an inline SVG data URI.

    The SVG is a stylized chameleon mark in the dashboard's orange +
    teal palette — same shape & colours as static/favicon.svg. Inline
    so the dashboard's BaseHTTPRequestHandler doesn't need a separate
    file-serving route. To swap for a different icon, edit this
    helper or replace static/favicon.svg and copy its contents here.
    """
    # `#` MUST be %23-escaped in data URIs (otherwise it's parsed as a
    # fragment marker). Spaces + < > render fine in modern browsers.
    # Keep this in sync with static/favicon.svg.
    svg = (
        "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'>"
        # Orange teardrop body with a small horn at the top.
        "<path d='M 28 4 L 25 1 L 23 6 C 12 9 4 20 4 32 C 4 46 "
        "14 56 30 56 C 42 56 50 50 52 42 C 56 28 50 12 36 6 "
        "C 33 5 30 4 28 4 Z' fill='%23F5A623'/>"
        # Teal eye dot.
        "<circle cx='46' cy='22' r='5' fill='%231F8B8B'/>"
        # Teal stroked spiral tail (curls back into itself, with
        # the round-cap thickness reading as a solid fill at
        # favicon resolution).
        "<path d='M 30 32 C 14 36 12 56 32 60 C 52 62 60 46 54 34 "
        "C 48 24 34 28 34 42 C 34 50 44 52 46 44' fill='none' "
        "stroke='%231F8B8B' stroke-width='8' stroke-linecap='round'/>"
        "</svg>"
    )
    return (f'<link rel="icon" type="image/svg+xml" '
            f'href="data:image/svg+xml,{svg}"/>')


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
    # Horizontal gridlines removed — the only horizontal line on the
    # chart is the dashed Entry reference (drawn elsewhere when an
    # active bet sets a strike side).
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
                      y_min: float | None = None,
                      y_max: float | None = None,
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

    # Y-axis: callers can pin the range (``y_min`` / ``y_max``) when
    # the chart represents a value with inherent bounds — e.g. a
    # probability series capped at 0..100¢. When unpinned, auto-scale
    # to the actual data range with 8% padding (default behaviour).
    if y_min is not None and y_max is not None:
        y_lo = float(y_min)
        y_hi = float(y_max)
    else:
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
    # ``data-y-range`` exposes the chart's plotted Y range + padding to
    # the row-click JS hook so it can draw a horizontal threshold line
    # at the clicked row's strike value. Format:
    #   y_min, y_max, pad_b, pad_t, pad_l, pad_r
    y_range_attr = f"{y_lo:.6f},{y_hi:.6f},{pad_b},{pad_t},{pad_l},{pad_r}"
    out: List[str] = [
        f"<div class='wl-chart-wrap' "
        f"data-tmin='{t_min:.0f}' data-tmax='{t_max:.0f}' "
        f"data-padl='{pad_l}' data-innerw='{inner_w}' "
        f"data-padt='{pad_t}' data-padb='{pad_b}' data-h='{height}' "
        f"data-vbw='{width}' "
        f"data-points='{html.escape(points_payload)}' "
        f"data-fmt='{html.escape(fmt_payload)}'>",
        f"<svg data-chart='wl-hero' data-y-range='{y_range_attr}' "
        f"width='100%' height='{height}' viewBox='0 0 {width} {height}' "
        f"preserveAspectRatio='none' style='display:block'>"
    ]

    # 5 evenly-spaced y-axis labels, no horizontal gridlines — the
    # only horizontal line on the chart is the dashed Entry reference
    # drawn below (when there's an active bet). Keeps the visual
    # weight on the plotted polyline.
    for i in range(5):
        v = y_lo + (i / 4.0) * (y_hi - y_lo)
        y = y_at(v)
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
    # NO  position → red dotted line, label reads "Below $X"
    # The colour communicates "your winning territory": YES bets win
    # when the underlying ends up above the line (green = win), NO
    # bets win when it stays below (red = the threshold you don't
    # want to be above).
    if strike_is_active_bet and strike_in_range and reference_strike is not None:
        ys = y_at(float(reference_strike))
        is_no = (side == "NO")
        line_color = "#f85149" if is_no else "#3fb950"
        label = "Entry"
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
    "above $175000.00".

    When ``display['question_format']`` is set, an alternate idiom
    is used. Supported values:
      * ``"at_least_full"`` — "at least 200,000" (raw value, comma-
        separated, no divisor / unit). Used by the unemployment-claims
        bot to surface the full strike count in plain English instead
        of the "above 200K" shorthand fmt_underlying produces.
    """
    if display and display.get("question_format") == "at_least_full":
        if direction == "between" and low is not None and high is not None:
            return f"{int(round(float(low))):,} – {int(round(float(high))):,}"
        if low is not None and direction in ("above", "greater"):
            return f"at least {int(round(float(low))):,}"
        if low is not None and direction in ("below", "less"):
            return f"below {int(round(float(low))):,}"
        if low is not None:
            return f"{direction} {int(round(float(low))):,}"
        return direction or "—"
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
    """Compact "1.7d / 9.2h / 45m" rendering for the Closes-in cell.

    Negative inputs mean the contract's published close time is
    already in the past — typical for tennis paper bets whose match
    settled days ago but whose simulator hasn't received a settle
    signal yet. We surface those as "settled" + how long ago, so
    the user can see the bet is stuck open rather than just a dash.

    Special case: the ±2-minute window around the close time
    renders as "closing" instead of "0m" / "settled 1m ago".
    Avoids the misleading "0m" reading on rows that are actively
    resolving — the hedge daemon will sweep them shortly via the
    ``settled_auto`` path.
    """
    if minutes is None:
        return "—"
    if -2 < minutes < 2:
        return "closing..."
    if minutes < 0:
        ago = -minutes
        if ago > 1440:
            return f"settled {ago/1440:.0f}d ago"
        if ago > 60:
            return f"settled {ago/60:.0f}h ago"
        return f"settled {int(ago)}m ago"
    if minutes > 1440:
        return f"{minutes/1440:.1f}d"
    if minutes > 60:
        return f"{minutes/60:.1f}h"
    return f"{int(minutes)}m"


# Kalshi tickers encode the settlement date as ``YYMMMDD`` after the
# series prefix. ``KXATPMATCH-26MAY12TIRMED`` → 2026-05-12. Tennis
# sim positions don't carry an explicit expected_expiration_time so
# this regex is the universal fallback for the "Closes in" column.
_TICKER_DATE_RE = re.compile(
    r"-(?P<yy>\d{2})(?P<mon>[A-Z]{3})(?P<dd>\d{2})", re.IGNORECASE,
)
_MONTH_MAP = {m: i + 1 for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
     "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
)}


def minutes_to_close_from_ticker(ticker: str | None,
                                    assumed_close_hour_utc: int = 23,
                                    ) -> float | None:
    """Parse the settlement date out of a Kalshi ticker and return the
    signed minutes from now until that day's close window. Settlement
    happens after the event ends, so we anchor at the LAST hour of
    the encoded date (23:59 UTC by default).

    Positive return = minutes remaining until close.
    Negative return = minutes since the contract already settled.
    None = ticker doesn't match the ``-YYMMMDD`` pattern at all.

    The caller's display ``time_to_close_str`` knows how to format
    negative values as "settled Nd ago" so stuck-open paper positions
    on long-finished matches show meaningful state instead of "—".
    """
    if not ticker:
        return None
    m = _TICKER_DATE_RE.search(ticker)
    if not m:
        return None
    mon = _MONTH_MAP.get(m.group("mon").upper())
    if mon is None:
        return None
    try:
        year = 2000 + int(m.group("yy"))
        day = int(m.group("dd"))
        ts = datetime(year, mon, day, assumed_close_hour_utc, 59,
                       tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None
    return (ts - datetime.now(timezone.utc)).total_seconds() / 60.0


def ticker_cell_html(ticker: str | None) -> str:
    """Render a ticker as a Kalshi market-page link.

    Output mirrors the convention already used in the Watchlist table
    (``class='ticker-link'``): the visible text is the full market
    ticker, but the href targets ``kalshi.com/markets/<series>``
    where series is everything before the first hyphen, lowercased.
    Linking to the series page lands on the same market group the
    row is describing; Kalshi resolves it to whichever event is
    currently live.

    Returns "—" for None / empty input so callers can drop it into a
    `<td>` directly.
    """
    if not ticker:
        return "—"
    tt_esc = html.escape(ticker)
    series_lower = ticker.split("-", 1)[0].lower()
    if not series_lower:
        return tt_esc
    url = f"https://kalshi.com/markets/{series_lower}"
    return (f"<a href='{html.escape(url)}' target='_blank' "
            f"rel='noopener noreferrer' class='ticker-link'>{tt_esc}</a>")


def _match_text_from_ticker(ticker: str | None) -> str:
    """Parse the matchup string out of a Kalshi NBA ticker.

    Format: ``KXNBAGAME-{YY}{MMM}{DD}{AWAY}{HOME}-{TEAM}``
    Example: ``KXNBAGAME-26MAY08SASMIN-MIN`` → ``"MIN vs SAS"``.

    Returns ``""`` when the ticker doesn't fit the NBA pattern (gas /
    CPI / jobless tickers); the caller renders a ``—`` placeholder.
    """
    if not ticker:
        return ""
    parts = ticker.split("-")
    if len(parts) < 3 or not parts[0].startswith("KXNBAGAME"):
        return ""
    # Middle chunk: 7-char date prefix (YYMMMDD = e.g. ``26MAY08``)
    # then two 3-char tricodes for away + home.
    body = parts[1]
    if len(body) < 13:
        return ""
    away_tri = body[7:10]
    home_tri = body[10:13]
    if not (away_tri.isalpha() and home_tri.isalpha()):
        return ""
    return f"{home_tri.upper()} vs {away_tri.upper()}"


def _side_tricode_from_ticker(ticker: str | None, side: str) -> str:
    """Return the team tricode the bet is on. The third hyphen-segment
    of an NBA ticker carries the team for which YES = "this team wins".
    On a NO bet, we want the *other* team. Returns "" for non-NBA tickers.
    """
    if not ticker:
        return ""
    parts = ticker.split("-")
    if len(parts) < 3 or not parts[0].startswith("KXNBAGAME"):
        return ""
    yes_team = parts[2].upper()
    if (side or "").upper() == "YES":
        return yes_team
    # NO side → return the other team from the matchup chunk.
    body = parts[1]
    if len(body) < 13:
        return yes_team
    away_tri = body[7:10].upper()
    home_tri = body[10:13].upper()
    return away_tri if yes_team == home_tri else home_tri


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
.card { background: #1d232c; border: 1px solid #30363d; border-radius: 8px; padding: 14px 18px; flex: 1; min-width: 180px; box-shadow: 0 1px 2px rgba(0,0,0,0.35); text-align: center; }
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
/* Sport-bot active-bet side cell: same team-tricode / "vs opp"
   layout the watchlist uses underneath. Player / team names render
   in the default cell colour (no YES/NO accent) so the column reads
   as identity, not direction. */
.active-side-team strong { font-weight: 700; }
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
/* Bot-name link in the active-bets / bet-history tables — same
   restraint as the ticker links so the table stays readable. */
a.bot-link { color: inherit; text-decoration: none; }
a.bot-link:hover { color: #58a6ff; text-decoration: underline; }
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
/* Seasons tab — one card per league. Fixed-width slots (auto-fill
   so a single card never stretches to fill its row) keep the grid
   uniform regardless of how many cards are on the page. */
.season-grid { display: grid; gap: 14px;
   grid-template-columns: repeat(auto-fill, minmax(280px, 320px));
   justify-content: start; }
.season-card { background: #1d232c; border: 1px solid #30363d;
   border-radius: 8px; padding: 14px 16px;
   box-shadow: 0 1px 2px rgba(0,0,0,0.35); display: flex;
   flex-direction: column; gap: 10px; }
.season-card-head { display: flex; align-items: center;
   justify-content: space-between; gap: 8px; }
.season-bot { color: #f0f6fc; font-weight: 600; font-size: 14px;
   text-decoration: none; }
.season-bot:hover { color: #58a6ff; text-decoration: underline; }
.season-name { color: #c9d1d9; font-size: 12px; }
.season-countdown { display: flex; align-items: baseline; gap: 8px;
   margin-top: 4px; flex-wrap: wrap; }
.season-countdown-label { font-size: 11px; text-transform: uppercase;
   letter-spacing: 0.05em; color: #8b949e; }
.season-countdown-value { font-size: 18px; font-weight: 600;
   font-variant-numeric: tabular-nums; }
/* Progress bar from start → end. Empty before start, fills as time
   passes, stays full once the season is over. */
.season-progress { background: #161b22; border: 1px solid #30363d;
   border-radius: 999px; height: 6px; overflow: hidden; }
.season-progress-fill { background: #58a6ff; height: 100%;
   transition: width 1s linear; }
.season-meta { display: grid; grid-template-columns: repeat(3, 1fr);
   gap: 6px; margin-top: 4px; }
.season-meta > div { display: flex; flex-direction: column; gap: 2px; }
.season-meta-label { font-size: 10px; text-transform: uppercase;
   letter-spacing: 0.05em; color: #8b949e; }
.season-meta-value { font-size: 13px; color: #f0f6fc;
   font-variant-numeric: tabular-nums; }
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
/* Same circle-i affordance for the Kalshi-rules section header.
   Clicking opens the shared modal with the extended contract rules
   (primary + secondary paragraphs from Kalshi). */
.contract-rules-btn {
    background: #21262d; color: #8b949e; border: 1px solid #30363d;
    border-radius: 50%; width: 22px; height: 22px; padding: 0;
    font-family: Georgia, "Times New Roman", serif;
    font-style: italic; font-weight: 700;
    font-size: 13px; line-height: 1; cursor: pointer;
    display: inline-flex; align-items: center; justify-content: center;
    transition: background 120ms, border-color 120ms, color 120ms;
    vertical-align: 4px; }
.contract-rules-btn:hover { background: #2d333b; border-color: #1f6feb;
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
/* Slash separator inside the combined Kalshi/My/Edge/EV cells —
   muted so the per-side numbers (which keep their own colour
   spans) stay the visual focus, with the "/" reading as a divider. */
.cell-sep { color: #6e7681; padding: 0 2px; }
/* Align the YES | NO split inside watchlist .num cells AND the
   matching header sub-row so the pipe character lands on the
   same x-coordinate across header + every row. Each side becomes
   a fixed-width inline-block; YES right-aligned, NO left-aligned,
   separator fixed in the middle. */
td.num [data-side='yes'],
th.num [data-side='yes'] {
    display: inline-block; min-width: 3.5em;
    text-align: right; font-variant-numeric: tabular-nums; }
td.num [data-side='no'],
th.num [data-side='no'] {
    display: inline-block; min-width: 3.5em;
    text-align: left; font-variant-numeric: tabular-nums; }
/* Stack the header label on top and the "yes | no" sub-row
   beneath so the sub-row's pipe column-aligns with the data
   pipes in the same column. Lowercase + small + gray so the
   header label visually dominates. */
th.num .th-side-row { display: block; line-height: 1.3;
    margin-top: 2px; font-weight: 400; text-transform: none;
    letter-spacing: 0; }
/* Vertical YES-on-top / NO-on-bottom layout for the side-paired
   columns (My %, Kalshi %, Edge, EV). YES always renders green,
   NO always renders red — the side is conveyed by colour AND
   position, replacing the old horizontal "yes | no" rendering.
   ``.side-yes`` / ``.side-no`` use !important to override the
   per-row tinting rules (.row-bought etc.) that previously
   dimmed cells inside acted-on rows — the side colour should
   stay legible regardless of row state. */
td.num.cell-stack { padding-top: 2px; padding-bottom: 2px;
    line-height: 1.2; }
td.num.cell-stack .side-yes,
td.num.cell-stack .side-no {
    display: block; text-align: right;
    font-variant-numeric: tabular-nums;
    /* drop the inline-block min-width set by the [data-side]
       rules above — vertical cells don't need horizontal
       alignment between YES and NO. */
    min-width: 0; }
td.num.cell-stack .side-yes { color: #3fb950 !important; }  /* green */
td.num.cell-stack .side-no  { color: #f85149 !important; }  /* red   */
/* Bot card drift badge — amber pill that lights up when the model's
   training accuracy and live actual-win-% diverge by >10pp on n≥10
   closed bets. Surfaces "this model may have drifted" as a one-look
   signal without forcing users to compare two cells. */
/* Models panel header — the section title sits on a flex row
   that also accommodates the Pre-game / In-game toggle for sport
   bots. ALL model pages use this header so the title + body sit
   at the same vertical position regardless of whether the toggle
   is present. min-height matches the toggle's natural height so
   the row is the same size with or without it. */
.section .section-header { display: flex; align-items: center;
    justify-content: space-between; gap: 12px;
    padding: 14px 22px 10px; min-height: 32px; }
.section .section-header h2 { padding: 0; margin: 0; }
/* Pre-game / In-game toggle that lives in the Models panel header
   for sport bots. Pills mimic the existing tab-pill idiom but
   live inside one section instead of the page-level tab bar. */
.model-view-toggle { display: inline-flex; gap: 4px;
    padding: 4px; background: #0d1117;
    border: 1px solid #21262d; border-radius: 8px; }
.model-view-toggle .model-view-pill {
    text-decoration: none; color: #8b949e; font-size: 12px;
    font-weight: 600; padding: 6px 14px; border-radius: 5px;
    text-transform: none; letter-spacing: 0.02em; }
.model-view-toggle .model-view-pill:hover { color: #c9d1d9; }
.model-view-toggle .model-view-pill.model-view-active {
    background: #1d232c; color: #f0f6fc;
    box-shadow: 0 1px 2px rgba(0,0,0,0.4); }
/* In-game model pill — appears in the active-bets table next to the
   YES/NO badge when the live model has a confident view of the
   position. EXIT = red (model expects loss), RUN = green (model
   expects win and you should hold), HOLD = yellow (market may be
   overreacting; defer to thresholds). */
.in-game-pill { display: inline-block; margin-left: 6px;
    padding: 1px 6px; border-radius: 4px;
    font-size: 9px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.04em; line-height: 1.5;
    vertical-align: 2px; cursor: help; }
.in-game-pill.ig-green { background: rgba(63, 185, 80, 0.18);
    color: #3fb950; border: 1px solid rgba(63, 185, 80, 0.35); }
.in-game-pill.ig-red { background: rgba(248, 81, 73, 0.18);
    color: #f85149; border: 1px solid rgba(248, 81, 73, 0.35); }
.in-game-pill.ig-yellow { background: rgba(227, 179, 65, 0.18);
    color: #e3b341; border: 1px solid rgba(227, 179, 65, 0.35); }
.in-game-pill.ig-gray { background: rgba(139, 148, 158, 0.15);
    color: #8b949e; border: 1px solid rgba(139, 148, 158, 0.30); }
.drift-badge { display: inline-block; margin-left: 6px;
    padding: 1px 6px; border-radius: 4px;
    background: rgba(212, 153, 0, 0.18);
    color: #d49900; border: 1px solid rgba(212, 153, 0, 0.35);
    font-size: 9px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.04em; line-height: 1.5;
    vertical-align: 2px; }
/* Forecast-staleness badge — fires when the bot's stored
   current_gas_price has drifted away from the live Kalshi-implied
   spot by more than $0.20, which usually means the bot is reading
   a stale upstream data feed (EIA publishing lag, missed retrain,
   etc.). Shares the drift-badge typography so the two pills sit
   visually consistent next to the bot name. */
.stale-badge { display: inline-block; margin-left: 6px;
    padding: 1px 6px; border-radius: 4px;
    background: rgba(227, 179, 65, 0.18);
    color: #e3b341; border: 1px solid rgba(227, 179, 65, 0.35);
    font-size: 9px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.04em; line-height: 1.5;
    vertical-align: 2px; }
/* "×N" badge on history rows where the same ticker was traded
   multiple times (flap-trades collapsed into one row). Small,
   muted-gray so it doesn't compete with WON/LOST coloring. */
.merged-badge { display: inline-block; margin-left: 6px;
    padding: 0 5px; border-radius: 3px;
    background: rgba(139, 148, 158, 0.18);
    color: #8b949e; border: 1px solid rgba(139, 148, 158, 0.3);
    font-size: 9px; font-weight: 700; line-height: 1.4;
    vertical-align: 1px; cursor: help; }
/* Auto-pause notifications panel — surfaced above the bot-card
   grid on Home when the regime monitor has flipped a bot off in the
   recent past. Silent (no DOM) when the audit log is empty so the
   page stays calm on the happy path. */
.notifications-panel { margin-bottom: 14px;
    background: #1d1f24; border: 1px solid #3d342a;
    border-left: 3px solid #e3934d; border-radius: 6px;
    padding: 10px 14px; }
.notifications-head { display: flex; gap: 10px;
    flex-wrap: wrap; align-items: baseline; margin-bottom: 6px; }
.notifications-title { color: #e3934d; font-weight: 700;
    font-size: 12px; text-transform: uppercase;
    letter-spacing: 0.06em; }
.notifications-list { margin: 0; padding: 0; list-style: none;
    display: flex; flex-direction: column; gap: 4px; }
.notifications-list li { display: grid;
    grid-template-columns: 130px 160px 1fr;
    gap: 10px; font-size: 12px; color: #c9d1d9; }
.notification-ts { color: #8b949e; font-family: monospace;
    font-size: 11px; }
.notification-bot { color: #f0f6fc; font-weight: 600; }
.notification-reason { color: #8b949e; }
/* Regime-status pill — sits inline with the bot name on the Home
   tab cards. Three states map to existing summary colours so the
   palette stays consistent: green = edge confirmed, yellow = edge
   eroding, red = anti-edge. */
.regime-pill { display: inline-block; margin-left: 6px;
    padding: 1px 6px; border-radius: 4px;
    font-size: 9px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.04em; line-height: 1.5;
    vertical-align: 2px; }
.regime-pill.regime-green { background: rgba(63, 185, 80, 0.18);
    color: #3fb950; border: 1px solid rgba(63, 185, 80, 0.35); }
.regime-pill.regime-yellow { background: rgba(227, 179, 65, 0.18);
    color: #e3b341; border: 1px solid rgba(227, 179, 65, 0.35); }
.regime-pill.regime-red { background: rgba(248, 81, 73, 0.18);
    color: #f85149; border: 1px solid rgba(248, 81, 73, 0.35); }
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
.bot-card-head { display: flex; align-items: flex-start;
    justify-content: space-between; gap: 12px;
    border-bottom: 1px solid #21262d; padding-bottom: 10px;
    margin-bottom: 10px;
    /* Reserve a fixed height for the name + ticker block. Without
       this, adding / removing the PAUSED badge bumps the card by
       ~18px when the toggle flips. */
    min-height: 48px; }
.bot-card-head-left { display: flex; flex-direction: column;
    gap: 2px; min-width: 0; }
.bot-card-head .bot-name { font-size: 14px; font-weight: 700;
    color: #f0f6fc; letter-spacing: -0.2px;
    display: inline-flex; align-items: center; gap: 6px;
    /* PAUSED badge inserts/removes inline next to the name — line
       height clamp keeps the row a constant height regardless. */
    line-height: 22px; }
.bot-card-head .bot-meta { font-size: 10px; color: #8b949e;
    text-transform: uppercase; letter-spacing: 0.04em;
    margin-top: 2px; }
/* On/off toggle in the card header. The track + knob is pure CSS
   styled to feel like the iOS-style switches the rest of the
   industry uses — green when on, gray when off, with a 200ms slide
   on the knob so the click feels responsive. */
.bot-toggle { all: unset; cursor: pointer; display: inline-flex;
    align-items: center; gap: 6px; padding: 4px 8px;
    border-radius: 999px; background: transparent;
    border: 1px solid transparent; }
.bot-toggle:hover { background: #1d232c; border-color: #30363d; }
.bot-toggle .bot-toggle-track { position: relative;
    width: 32px; height: 18px; border-radius: 999px;
    background: #30363d; transition: background 160ms; }
.bot-toggle .bot-toggle-knob { position: absolute;
    top: 2px; left: 2px; width: 14px; height: 14px;
    border-radius: 50%; background: #f0f6fc;
    transition: transform 200ms cubic-bezier(0.4, 0, 0.2, 1); }
.bot-toggle[data-enabled='1'] .bot-toggle-track { background: #2da44e; }
.bot-toggle[data-enabled='1'] .bot-toggle-knob { transform: translateX(14px); }
.bot-toggle .bot-toggle-label { font-size: 10px; font-weight: 700;
    color: #8b949e; letter-spacing: 0.05em;
    font-family: ui-monospace, SFMono-Regular, Consolas, monospace; }
.bot-toggle[data-enabled='1'] .bot-toggle-label { color: #2da44e; }
/* Paused card — dim the content + drop the hover lift so the bot
   reads as "off" at a glance without disappearing entirely. */
.bot-card-paused { opacity: 0.55; border-style: dashed; }
.bot-card-paused:hover { transform: none; border-color: #30363d;
    background: #0d1117; }
.paused-badge { display: inline-block; padding: 1px 6px;
    border-radius: 4px; background: rgba(139, 148, 158, 0.2);
    color: #c9d1d9; font-size: 9px; font-weight: 600;
    letter-spacing: 0.08em; text-transform: uppercase; }
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
   position on. No green/red tint — the row reads in plain white
   instead, with a subtle left rail + bold ticker to stay
   distinguishable. The Verdict column's HOLDING YES / HOLDING NO
   badge is what conveys the bet direction. Wins specificity over
   row-suspect so a held position is never dimmed. */
tr.row-bought td { opacity: 1 !important; color: #c9d1d9 !important; }
tr.row-bought td:first-child { border-left: 3px solid #8b949e; }
tr.row-bought td.mono a.ticker-link,
tr.row-bought td.mono { color: #f0f6fc; font-weight: 600; }
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
/* Watchlist chart hero top-left — the forecast price + change
   indicator that lived here previously were removed per user
   request. The replacement is the volume of the contract the
   chart line represents (atm market), matching the same large-
   number + small-label visual rhythm. */
.wl-hero-volume { font-size: 24px; font-weight: 700; color: #f0f6fc;
    letter-spacing: -0.3px; }
.wl-hero-volume-label { font-size: 12px; font-weight: 500; color: #8b949e;
    text-transform: lowercase; margin-left: 4px; letter-spacing: 0.02em; }
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
/* Scroll container around the Summary's "Active bets" table. The
   per-bot active-bets list lower on the Watchlist tab keeps its
   natural height — only the global aggregate at the top of Home
   gets clamped. Matches the .watchlist-scroll idiom used for the
   strike-ladder table. */
.summary-active-scroll { max-height: 280px; overflow-y: auto;
    border: 1px solid #30363d; border-radius: 6px;
    background: #0d1117; }
.summary-active-scroll table { margin: 0; }
.summary-active-scroll thead th { position: sticky; top: 0;
    background: #161b22; z-index: 1; }
/* Watchlist-tab Active bets scroller. Mirrors .watchlist-scroll
   (the strike-ladder table below) so the two read as a stacked
   pair, but capped at a smaller height since it's the bet list
   not the full ladder. Section-grey background contrasts the
   near-black chart panel directly above it. */
.watchlist-active-scroll { max-height: 220px; overflow-y: auto;
    border: 1px solid #21262d; border-radius: 6px;
    background: #161b22; margin-top: 4px;
    margin-bottom: 14px; }
.watchlist-active-scroll table { margin: 0; border: none; }
.watchlist-active-scroll thead th { position: sticky; top: 0;
    z-index: 1; background: #1d232c;
    box-shadow: 0 1px 0 #30363d; }
/* History tab scroll container — taller than the Summary's active
   bets scroll since the History tab is dedicated to this table.
   ~14 rows visible before the user scrolls. */
.history-scroll { max-height: 640px; overflow-y: auto;
    border: 1px solid #30363d; border-radius: 6px;
    background: #0d1117; margin-top: 10px; }
.history-scroll table { margin: 0; }
.history-scroll thead th { position: sticky; top: 0;
    background: #161b22; z-index: 1; }
/* HTML <details> wrappers inside the scroll container — the
   collapsed "show more" rows are invisible until expanded; keep the
   summary line sticky so it stays accessible when scrolled. */
.history-scroll details > summary { position: sticky; bottom: 0;
    background: #161b22; padding: 6px 10px; cursor: pointer;
    border-top: 1px solid #30363d; }
/* History tab P&L attribution — small breakdown tables in a two-up
   grid. Each panel has its own h3 subhead and a compact table. */
.attribution-grid { display: grid; gap: 14px;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    margin-bottom: 14px; }
.attribution-panel { background: #0d1117;
    border: 1px solid #21262d; border-radius: 6px;
    padding: 10px 14px; }
.attribution-panel h3.subhead { margin-top: 0; margin-bottom: 6px; }
.attribution-panel table { font-size: 12px; }
.attribution-panel th, .attribution-panel td { padding: 6px 8px; }
/* History tab P&L line chart — sits between the headline cards and
   the ledger table. The wrap is `position: relative` so the empty-
   state overlay can be absolute-positioned over the SVG frame. */
.history-chart-section { margin-top: 14px; }
/* Inline toolbar above the chart: chart title on the left, period
   selector on the right. Suppress the .bot-filter-bar divider so
   the filter reads as a chart control, not a section break. */
.history-chart-toolbar { display: flex; align-items: center;
    justify-content: space-between; gap: 12px;
    margin: 0 0 8px 0; flex-wrap: wrap; }
.history-chart-toolbar .history-chart-title {
    color: #c9d1d9; font-size: 14px; font-weight: 600; }
.history-chart-toolbar .bot-filter-bar { padding: 0; margin: 0;
    border-bottom: none; }
.history-chart-wrap { position: relative;
    border: 1px solid #30363d; border-radius: 6px;
    background: #0d1117; padding: 8px 4px 4px 4px; }
"""


def render_page(
    model: dict | None,
    global_summary: dict,
    global_active_bets: List[dict],
    global_history: List[dict],
    latest_active: dict | None,
    bot_active_bets: List[dict] | None,
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
    prob_history: List[dict] | None = None,
    model_view: str = "pregame",
    threshold_source: dict | None = None,
) -> str:
    out: List[str] = []
    out.append("<!doctype html><html><head>")
    out.append("<meta charset='utf-8'>")
    # No meta-refresh — JS at the bottom of the page polls /api/snapshot
    # every 5s and patches live cells in place. The page never reloads.
    out.append(f"<title>Kalshi simulation dashboard</title>")
    out.append(_favicon_link())
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
        ("models", "Models"),
        ("history", "History"),
        ("seasons", "Seasons"),
    ]
    valid_tabs = {k for k, _ in tabs}
    active_tab = tab_key if tab_key in valid_tabs else "home"

    # Bot filter sits above the tab bar (per user request) so it
    # applies uniformly across every tab and doesn't reflow when
    # panels swap. Selecting a bot navigates to that bot's URL on
    # the current tab — the per-tab filters that previously lived
    # inside Summary / Watchlist / Models sections were removed to
    # avoid duplication.
    if available_bots:
        _render_bot_filter(out, available_bots,
                            current_bot=current_bot,
                            period_key=period_key,
                            select_id="bot-select-top",
                            include_all_option=True,
                            tab_key=active_tab)

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
                     period_key=period_key, current_bot=current_bot,
                     available_bots=available_bots,
                     hedge_cfg=hedge_cfg)
    out.append("<div class='section'><h2>Model performance</h2>"
               "<div class='body'>")
    _render_notifications_panel(out)
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
                          bot_active_bets=bot_active_bets or [],
                          kalshi_history=kalshi_history,
                          prob_history=prob_history or [],
                          atm_market=atm_market,
                          contract_open_ts=contract_open_ts,
                          contract_close_ts=contract_close_ts,
                          event_title=event_title,
                          edge_cfg=edge_cfg,
                          validator_cfg=validator_cfg,
                          risk_caps=risk_caps,
                          hedge_cfg=hedge_cfg,
                          threshold_source=threshold_source,
                          available_bots=available_bots,
                          current_bot=current_bot,
                          period_key=period_key)
        _render_contract_rules(
            out, watchlist, current_bot,
            contract_close_ts=contract_close_ts,
        )
    out.append("</div>")  # /watchlist panel

    # ── MODELS tab — per-bot model deep-dive ─────────────────────────
    _open_panel("models")
    current_bot_dict = next(
        (b for b in available_bots if b.get("key") == current_bot),
        None,
    )
    _render_models_panel(
        out,
        bot=current_bot_dict or {},
        model=model,
        display=display,
        available_bots=available_bots,
        current_bot=current_bot,
        model_view=model_view,
        bot_active_bets=bot_active_bets,
        bot_closed_positions=bot_closed_positions,
    )
    out.append("</div>")  # /models panel

    # ── HISTORY tab — closed-bet history across all bots ──────────────
    _open_panel("history")
    out.append(
        f"<div class='section'><h2>Contract history "
        f"<span class='small gray'>({html.escape(period_label)})"
        f"</span></h2>"
        f"<div class='body'>"
    )
    # Headline cards at the top of the panel. id_suffix='-history'
    # lets the snapshot poller (which targets Home-tab ids) skip them.
    _render_summary_cards(out, global_summary, id_suffix="-history",
                           show_closed_contracts=True)
    # Daily P&L chart with the Day/Week/Month/Year/All-time period
    # selector rendered as the chart's toolbar — period scopes the
    # chart, cards, and table below via a full-page reload on the
    # ``?period=X`` query param.
    _render_history_chart(out, global_history,
                            period_key=period_key,
                            current_bot=current_bot)
    # P&L attribution panels — small breakdown tables that answer
    # "where did this P&L come from?" by bot / month / side / EV
    # bucket. Sits between the chart and the ledger so the user sees
    # the shape of the P&L before reading individual bet rows.
    _render_history_attribution(out, global_history)
    # Pass heading="" so the table renders without a duplicate
    # subhead — the section title above already carries the period.
    # Scroll container clamps the table to a sensible viewport
    # height so long histories don't push the rest of the page off
    # screen — same idiom as the Summary's active-bets scroll.
    out.append("<div class='history-scroll'>")
    _render_bet_history_block(
        out, global_history,
        heading="",
        shown_initially=20,
    )
    out.append("</div>")
    out.append("</div></div>")
    out.append("</div>")  # /history panel

    # ── SEASONS tab — one card per bot with a real-world season ──────
    _open_panel("seasons")
    _render_seasons_panel(out, available_bots)
    out.append("</div>")  # /seasons panel

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
        "_source": threshold_source or {"source": "fallback",
                                          "captured_at": None,
                                          "missing_keys": []},
    }, separators=(",", ":"), default=str)
    out.append(
        f"<script>window.__BUY_CRITERIA__ = {buy_criteria_payload};</script>"
    )
    out.append(_BOT_TOGGLE_JS)
    out.append(_HISTORY_CHART_JS)
    out.append(_SEASON_COUNTDOWN_JS)
    out.append(_live_update_script(current_bot, period_key=period_key))
    out.append("</body></html>")
    return "".join(out)


# Click handler for the homepage bot-card toggle. Hits the
# /api/bot/toggle POST endpoint with the bot key, then updates the
# card's data-enabled state + label / PAUSED badge without a reload.
# event.preventDefault stops the parent <a class='bot-card'> from
# following its href when the user clicks the toggle itself.
_BOT_TOGGLE_JS = """<script>
function toggleBotState(ev, btn) {
  ev.preventDefault();
  ev.stopPropagation();
  const key = btn.dataset.botKey;
  if (!key) return;
  btn.disabled = true;
  fetch('/api/bot/toggle?bot=' + encodeURIComponent(key), {method: 'POST'})
    .then(function (r) { return r.json(); })
    .then(function (data) {
      const enabled = !!data.enabled;
      btn.dataset.enabled = enabled ? '1' : '0';
      btn.setAttribute('aria-pressed', enabled ? 'true' : 'false');
      const card = btn.closest('.bot-card');
      if (card) {
        card.classList.toggle('bot-card-paused', !enabled);
        // Add or remove the PAUSED badge to match the new state.
        const nameEl = card.querySelector('.bot-name');
        if (nameEl) {
          const existing = nameEl.querySelector('.paused-badge');
          if (!enabled && !existing) {
            const pill = document.createElement('span');
            pill.className = 'paused-badge';
            pill.title = 'Bot is paused — toggle on to resume taking bets.';
            pill.textContent = 'PAUSED';
            nameEl.appendChild(pill);
          } else if (enabled && existing) {
            existing.remove();
          }
        }
      }
    })
    .catch(function () { /* swallow — keep current visual state */ })
    .finally(function () { btn.disabled = false; });
}
</script>"""


# Seasons-tab live countdown. Each card carries data-start / data-end
# (ISO datetimes) — we tick once a second and update the headline +
# remaining-time fields. Only two states are surfaced:
#   • Before start  → "Starts in …" (yellow)
#   • Between       → "Ends in …"   (green)
# A card whose season has already ended is hidden server-side (the
# renderer doesn't emit it), so the JS doesn't need an "over" branch.
_SEASON_COUNTDOWN_JS = """<script>
(function () {
  function fmt(ms) {
    if (ms <= 0) return "0d 0h 0m 0s";
    const s = Math.floor(ms / 1000);
    const days = Math.floor(s / 86400);
    const hours = Math.floor((s % 86400) / 3600);
    const mins = Math.floor((s % 3600) / 60);
    const secs = s % 60;
    return days + "d " + hours + "h " + mins + "m " + secs + "s";
  }
  function pct(now, start, end) {
    if (now <= start) return 0;
    if (now >= end) return 100;
    const span = end - start;
    if (span <= 0) return 100;
    return Math.max(0, Math.min(100, ((now - start) / span) * 100));
  }
  function tick() {
    const now = Date.now();
    document.querySelectorAll('[data-season-card]').forEach(function (card) {
      const start = parseInt(card.dataset.start, 10);
      const end = parseInt(card.dataset.end, 10);
      if (!isFinite(start) || !isFinite(end)) return;
      const statusEl = card.querySelector('[data-season-status]');
      const labelEl = card.querySelector('[data-season-countdown-label]');
      const valueEl = card.querySelector('[data-season-countdown-value]');
      const fillEl = card.querySelector('[data-season-progress-fill]');
      let status, label, value, color;
      if (now < start) {
        status = 'Upcoming';
        label = 'Starts in';
        value = fmt(start - now);
        color = 'yellow';
      } else if (now < end) {
        status = 'In season';
        label = 'Ends in';
        value = fmt(end - now);
        color = 'green';
      } else {
        // Season just ticked past its end window while the page was
        // open. Hide the card rather than flashing "season over" —
        // matches the server-side behaviour of dropping ended cards.
        card.style.display = 'none';
        return;
      }
      if (statusEl) {
        statusEl.textContent = status;
        statusEl.classList.remove('green', 'yellow', 'gray');
        statusEl.classList.add(color);
      }
      if (labelEl) labelEl.textContent = label;
      if (valueEl) {
        valueEl.textContent = value;
        valueEl.classList.remove('green', 'yellow', 'gray');
        valueEl.classList.add(color);
      }
      if (fillEl) fillEl.style.width = pct(now, start, end).toFixed(2) + '%';
    });
  }
  tick();
  setInterval(tick, 1000);
})();
</script>"""

# History-tab daily P&L chart renderer. Reads the closed-bet ledger
# embedded as JSON on the SVG node, filters by selected bot + date
# range, buckets bets by UTC day, then plots each day's net realized
# P&L in cents. The line crosses zero naturally when winning vs. losing
# days alternate; a dashed baseline at $0 makes the sign obvious.
_HISTORY_CHART_JS = """<script>
(function () {
  const svg = document.querySelector('[data-history-chart]');
  if (!svg) return;
  let raw = [];
  try { raw = JSON.parse(svg.dataset.points || '[]'); } catch (e) {}
  const W = 800, H = 260;
  const PAD_L = 56, PAD_R = 14, PAD_T = 14, PAD_B = 28;
  const INNER_W = W - PAD_L - PAD_R;
  const INNER_H = H - PAD_T - PAD_B;
  function fmtSignedDollars(cents) {
    const v = cents / 100;
    const sign = v > 0 ? '+' : (v < 0 ? '-' : '');
    return sign + '$' + Math.abs(v).toFixed(2);
  }
  function fmtDate(epoch) {
    const d = new Date(epoch * 1000);
    return d.toLocaleDateString(undefined,
      {month: 'short', day: 'numeric'});
  }
  function render() {
    const now = Math.floor(Date.now() / 1000);
    // Bucket every closed bet by UTC day, summing realized P&L in
    // cents per bucket. Each series point is (UTC-midnight epoch,
    // day net).
    const daily = new Map();
    raw.forEach(function (p) {
      const d = new Date(p[0] * 1000);
      const dayEpoch = Math.floor(Date.UTC(
        d.getUTCFullYear(), d.getUTCMonth(), d.getUTCDate()) / 1000);
      daily.set(dayEpoch, (daily.get(dayEpoch) || 0) + p[1]);
    });
    const series = Array.from(daily.entries())
      .sort(function (a, b) { return a[0] - b[0]; });
    svg.innerHTML = '';
    if (series.length === 0) return;
    // X range: first closed-bet day → now.
    const tMin = series[0][0];
    const tMax = now;
    const tSpan = Math.max(1, tMax - tMin);
    // Y range: include 0 so the zero baseline always shows. 8% pad.
    let vals = series.map(function (s) { return s[1]; });
    vals.push(0);
    let yMin = Math.min.apply(null, vals);
    let yMax = Math.max.apply(null, vals);
    if (yMin === yMax) { yMin -= 1; yMax += 1; }
    const yPad = (yMax - yMin) * 0.08;
    yMin -= yPad; yMax += yPad;
    function x(t) {
      return PAD_L + ((t - tMin) / tSpan) * INNER_W;
    }
    function y(v) {
      return PAD_T + (1 - (v - yMin) / (yMax - yMin)) * INNER_H;
    }
    const NS = 'http://www.w3.org/2000/svg';
    function el(name, attrs, text) {
      const n = document.createElementNS(NS, name);
      for (const k in attrs) n.setAttribute(k, attrs[k]);
      if (text != null) n.textContent = text;
      return n;
    }
    // Horizontal gridlines + Y labels (5 ticks).
    for (let i = 0; i <= 4; i++) {
      const yv = yMin + (i / 4) * (yMax - yMin);
      const py = y(yv);
      svg.appendChild(el('line', {
        x1: PAD_L, y1: py, x2: W - PAD_R, y2: py,
        stroke: '#1f2530', 'stroke-width': '1'
      }));
      svg.appendChild(el('text', {
        x: PAD_L - 6, y: py + 4, fill: '#8b949e',
        'font-size': '10', 'text-anchor': 'end'
      }, fmtSignedDollars(yv)));
    }
    // Zero baseline — always drawn so the user can see at a glance
    // when the daily line is above (gains) vs. below (losses).
    const py0 = y(0);
    svg.appendChild(el('line', {
      x1: PAD_L, y1: py0, x2: W - PAD_R, y2: py0,
      stroke: '#6e7681', 'stroke-width': '1'
    }));
    // X-axis date ticks — up to 6 evenly spaced labels across the
    // visible time span.
    const N_TICKS = 5;
    for (let i = 0; i <= N_TICKS; i++) {
      const t = tMin + (i / N_TICKS) * tSpan;
      const px = x(t);
      svg.appendChild(el('text', {
        x: px, y: H - 8, fill: '#8b949e',
        'font-size': '10',
        'text-anchor': i === 0 ? 'start' :
          (i === N_TICKS ? 'end' : 'middle')
      }, fmtDate(t)));
    }
    // Split the line into colored segments — green when the value is
    // >= 0 (gains), red when < 0 (losses). When two consecutive points
    // straddle zero, interpolate the crossing so the color flips
    // exactly at the baseline.
    const GREEN = '#3fb950', RED = '#f85149';
    const colorOf = function (v) { return v >= 0 ? GREEN : RED; };
    for (let i = 1; i < series.length; i++) {
      const a = series[i - 1], b = series[i];
      const sameSide = (a[1] >= 0) === (b[1] >= 0);
      if (sameSide) {
        svg.appendChild(el('polyline', {
          points: x(a[0]).toFixed(1) + ',' + y(a[1]).toFixed(1) + ' ' +
            x(b[0]).toFixed(1) + ',' + y(b[1]).toFixed(1),
          fill: 'none', stroke: colorOf(a[1]),
          'stroke-width': '2',
          'stroke-linejoin': 'round', 'stroke-linecap': 'round'
        }));
      } else {
        // Interpolate t where the line crosses zero.
        const t = a[1] / (a[1] - b[1]);
        const xc = a[0] + t * (b[0] - a[0]);
        svg.appendChild(el('polyline', {
          points: x(a[0]).toFixed(1) + ',' + y(a[1]).toFixed(1) + ' ' +
            x(xc).toFixed(1) + ',' + y(0).toFixed(1),
          fill: 'none', stroke: colorOf(a[1]),
          'stroke-width': '2',
          'stroke-linejoin': 'round', 'stroke-linecap': 'round'
        }));
        svg.appendChild(el('polyline', {
          points: x(xc).toFixed(1) + ',' + y(0).toFixed(1) + ' ' +
            x(b[0]).toFixed(1) + ',' + y(b[1]).toFixed(1),
          fill: 'none', stroke: colorOf(b[1]),
          'stroke-width': '2',
          'stroke-linejoin': 'round', 'stroke-linecap': 'round'
        }));
      }
    }
    // Dots on each daily point, colored by sign of that day's net.
    series.forEach(function (s) {
      svg.appendChild(el('circle', {
        cx: x(s[0]), cy: y(s[1]), r: '2.5',
        fill: colorOf(s[1])
      }));
    });
  }
  render();
})();
</script>"""


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
    if (ev === null || ev === undefined) return "0";
    const rounded = Math.round(ev * 100) / 100;
    if (rounded === 0) return "0";
    const sign = rounded >= 0 ? "+" : "−";
    return sign + "$" + Math.abs(rounded).toFixed(2);
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
    // Home tab: Active bets | Active contracts | Active bots
    // | Money spent | Potential gain | Week change %.
    // All values are live, never period-scoped.
    const s = snap.summary || {{}};
    patch("card-active-bets", String(s.active_bets ?? 0));
    patch("card-active-contracts", String(s.active_contracts ?? 0));
    patch("card-active-bots", String(s.active_bots ?? 0));
    patch("card-money-spent",
          (s.active_money_spent_cents ?? 0) === 0
            ? "$0.00"
            : fmtSignedCents(-(s.active_money_spent_cents ?? 0)));
    patch("card-potential-earnings",
          "+" + fmtSignedCents(s.potential_gain_cents).replace(/^[+−-]/, ""),
          "green");
    // Week change %: (this_week / |net - this_week|) * 100. Mirrors
    // the Python _week_change_pct so the polled value matches the
    // server-rendered first paint.
    {{
      const tw = s.this_week_pnl_cents ?? 0;
      const lt = s.net_pnl_cents ?? 0;
      const wa = lt - tw;
      let text, cls;
      if (wa === 0) {{ text = "—"; cls = "gray"; }}
      else {{
        const pct = (tw / Math.abs(wa)) * 100;
        const sign = pct > 0 ? "+" : (pct < 0 ? "−" : "");
        text = sign + Math.abs(pct).toFixed(1) + "%";
        cls = pct > 0 ? "green" : (pct < 0 ? "red" : "gray");
      }}
      patch("card-week-change", text, cls);
    }}

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
        }} else {{
          // Per user request: only HOLDING rows render full-bright
          // white. When a position closes mid-poll, re-apply the
          // greyed style so the row drops back to dimmed without
          // a full page reload.
          tr.classList.add("row-suspect");
        }}
      }});
      // Patch a single side-span inside one of the combined cells
      // (Kalshi / My / Edge / EV). Each cell has two spans flanking
      // a "/" separator; we update them in place so the polled
      // refresh keeps the per-side colour without re-rendering.
      function patchSide(cell, side, text, cls) {{
        if (!cell) return;
        const span = cell.querySelector("span[data-side='" + side + "']");
        if (!span) return;
        if (span.textContent !== text) {{
          span.textContent = text;
          flash(cell);
        }}
        if (cls !== undefined) {{
          span.classList.remove("green", "red", "yellow", "gray");
          if (cls) span.classList.add(cls);
        }}
      }}
      function fmtPctEdge(e) {{
        if (e === null || e === undefined) return "0";
        const pp = Math.round(e * 100);
        if (pp === 0) return "0";
        return (pp >= 0 ? "+" : "") + pp + "%";
      }}
      function edgeClass(e) {{
        if (e === null || e === undefined) return "gray";
        if (e >= 0.05) return "green";
        if (e > 0) return "yellow";
        if (e <= -0.02) return "red";
        return "gray";
      }}
      snap.watchlist.forEach(function (r) {{
        const tr = rowsByTicker[r.ticker];
        if (!tr) return;  // server added a new row — page reload would catch
        const ya = r.kalshi_yes, na = r.kalshi_no;
        const kyes = (ya !== null && ya !== undefined) ? (ya + "%")
                   : (na !== null && na !== undefined) ? ((100 - na) + "%")
                   : "—";
        const kno  = (na !== null && na !== undefined) ? (na + "%")
                   : (ya !== null && ya !== undefined) ? ((100 - ya) + "%")
                   : "—";
        const myYes = (r.model_prob_yes !== null && r.model_prob_yes !== undefined)
          ? (Math.round(r.model_prob_yes * 100) + "%") : "—";
        const myNo = (r.model_prob_yes !== null && r.model_prob_yes !== undefined)
          ? (Math.round((1 - r.model_prob_yes) * 100) + "%") : "—";
        // Edge (raw model − Kalshi ask, no half-spread). Computed
        // here client-side so the snapshot endpoint doesn't need to
        // ship two extra fields per row.
        const edgeYes = (r.model_prob_yes !== null && r.model_prob_yes !== undefined
                          && ya !== null && ya !== undefined)
          ? (r.model_prob_yes - ya / 100) : null;
        const edgeNo = (r.model_prob_yes !== null && r.model_prob_yes !== undefined
                         && na !== null && na !== undefined)
          ? ((1 - r.model_prob_yes) - na / 100) : null;
        patchCell(tr.querySelector("[data-field='oi']"),
                  r.open_interest !== null && r.open_interest !== undefined
                    ? Number(r.open_interest).toLocaleString() : "—");
        const kalshiCell = tr.querySelector("[data-field='kalshi']");
        patchSide(kalshiCell, 'yes', kyes);
        patchSide(kalshiCell, 'no',  kno);
        const myCell = tr.querySelector("[data-field='my']");
        patchSide(myCell, 'yes', myYes);
        patchSide(myCell, 'no',  myNo);
        const edgeCell = tr.querySelector("[data-field='edge']");
        patchSide(edgeCell, 'yes', fmtPctEdge(edgeYes), edgeClass(edgeYes));
        patchSide(edgeCell, 'no',  fmtPctEdge(edgeNo),  edgeClass(edgeNo));
        const evCell = tr.querySelector("[data-field='ev']");
        patchSide(evCell, 'yes', fmtEv(r.ev_yes), evClass(r.ev_yes, minEv));
        patchSide(evCell, 'no',  fmtEv(r.ev_no),  evClass(r.ev_no, minEv));
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
  // Bot-selectors (one on Home, one on each per-bot Watchlist tab).
  // All marked with [data-bot-select]; on change we navigate to the
  // option's value. The option values bake in the SERVER-rendered tab
  // (from ?tab= at page load), but tab pills swap panels client-side
  // via history.replaceState — so the option's URL goes stale once
  // the user changes tabs. Re-read the current tab from the URL bar
  // (or fall back to the active tab pill) and override the option's
  // ?tab= so the bot switch keeps the user on whichever tab is
  // currently visible.
  // Browsers default to "auto" scroll restoration on history navigation
  // but they do NOT restore scroll on cross-URL navigation (which is
  // what a bot-select change triggers). Stash the current scrollY in
  // sessionStorage so the next page load can re-apply it.
  if ("scrollRestoration" in history) {{
    history.scrollRestoration = "manual";
  }}
  const _SCROLL_KEY = "dashboardBotSwitchScrollY";
  try {{
    const saved = sessionStorage.getItem(_SCROLL_KEY);
    if (saved !== null) {{
      sessionStorage.removeItem(_SCROLL_KEY);
      // Defer one frame so the layout settles before scrolling — the
      // body keeps growing as deferred SVG charts paint in.
      requestAnimationFrame(function () {{
        window.scrollTo(0, parseInt(saved, 10) || 0);
      }});
    }}
  }} catch (err) {{ /* sessionStorage disabled — ignore */ }}

  document.querySelectorAll("[data-bot-select]").forEach(function (sel) {{
    sel.addEventListener("change", function () {{
      let target = sel.value;
      if (!target) return;
      // The visible tab pill is the authoritative source of truth for
      // what panel the user is looking at — tab clicks swap panels
      // client-side and then call history.replaceState, but that
      // replaceState can lag or skip in edge cases (modal interaction,
      // browser quirks). The pill's .tab-pill-active class is always
      // in sync with the visible panel.
      let currentTab = null;
      const activePill = document.querySelector(".tab-pill-active");
      if (activePill) currentTab = activePill.getAttribute("data-tab");
      if (!currentTab) {{
        try {{
          currentTab = new URL(window.location.href)
            .searchParams.get("tab");
        }} catch (err) {{ /* old browser */ }}
      }}
      try {{
        // Resolve against the current origin so we can use
        // URLSearchParams regardless of whether the option value
        // starts with "?" or "/". Per-bot options carry ?bot=X — for
        // those we always inject the active tab (overwriting whatever
        // tab the server rendered into the option) so the user stays
        // on the same tab. The "All bots" entry has no ?bot= and is
        // intentionally left alone — it lands on the cross-bot Home.
        const u = new URL(target, window.location.origin);
        if (currentTab && u.searchParams.has("bot")) {{
          u.searchParams.set("tab", currentTab);
          target = u.pathname + u.search;
        }}
      }} catch (err) {{ /* fall through to raw target */ }}
      try {{
        sessionStorage.setItem(_SCROLL_KEY, String(window.scrollY));
      }} catch (err) {{ /* ignore */ }}
      window.location.href = target;
    }});
  }});
  // Period-selectors (Home + History tabs).
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
    return sign + "$" + Math.abs(v).toFixed(2);
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
    // Bullet-point overview of every gate the bot runs before buying
    // and exiting. Each bullet renders the gate description AND the
    // actual numeric value the bot uses (pulled from the bot's
    // ``data/effective_config.json`` when present, else the dashboard
    // YAML — the source banner at the top tells the user which).
    // Use the per-position Why? button to see the actual values that
    // cleared each gate at entry-time for a specific bet.
    const ed  = (r && r.edge)       || {{}};
    const va  = (r && r.validators) || {{}};
    const rk  = (r && r.risk)       || {{}};
    const hg  = (r && r.hedge)      || {{}};
    const src = (r && r._source)    || {{}};

    function fmtNum(v, suffix) {{
      if (v === null || v === undefined || (typeof v === "number" && !isFinite(v))) {{
        return "—";
      }}
      if (typeof v === "boolean") return v ? "on" : "off";
      if (Array.isArray(v))      return v.join("–");
      return v + (suffix || "");
    }}
    function fmtCash(c) {{
      if (c === null || c === undefined || !isFinite(c)) return "—";
      return "$" + (c / 100).toFixed(2);
    }}
    function fmtPctF(v) {{
      if (v === null || v === undefined || !isFinite(v)) return "—";
      return (v * 100).toFixed(0) + "%";
    }}
    function fmtMinH(m) {{
      if (m === null || m === undefined || !isFinite(m)) return "—";
      if (m >= 1440) return (m / 1440).toFixed(0) + "d";
      if (m >= 60)   return (m / 60).toFixed(0) + "h";
      return m + "min";
    }}
    function fmtSec(s) {{
      if (s === null || s === undefined || !isFinite(s)) return "—";
      if (s >= 3600) return (s / 3600).toFixed(1) + "h";
      if (s >= 60)   return (s / 60).toFixed(0) + "min";
      return s + "s";
    }}
    function valSpan(s) {{
      return "<span style='font-variant-numeric:tabular-nums;"
           + "color:#f0f6fc;font-weight:600;'>" + s + "</span>";
    }}

    function bullets(items) {{
      let h = "<ul style='margin:0 0 0 18px;padding:0;line-height:1.55;"
            + "font-size:13px;color:#c9d1d9;'>";
      for (const it of items) {{
        // it = [label, description, value-html-or-null]
        const label = it[0];
        const desc  = it[1];
        const val   = it[2];
        h += "<li style='margin:0 0 6px 0;'>"
           + "<b>" + label + "</b>"
           + (val ? " " + valSpan(val) : "")
           + (desc ? " — <span class='gray'>" + desc + "</span>" : "")
           + "</li>";
      }}
      h += "</ul>";
      return h;
    }}
    let html = "";

    // ── Source banner. Tells the user whether the values below come
    //    from the bot's live config or the dashboard YAML defaults.
    if (src.source === "live") {{
      const ts = src.captured_at
        ? " <span class='gray'>(reported " + src.captured_at + ")</span>"
        : "";
      html += "<div class='crit-section' style='font-size:11px;"
           + "color:#3fb950;margin-bottom:10px;border:1px solid "
           + "rgba(63,185,80,0.30);background:rgba(63,185,80,0.08);"
           + "border-radius:4px;padding:6px 10px;'>"
           + "● Live config — these are the gates the bot is currently "
           + "applying" + ts
           + (src.missing_keys && src.missing_keys.length
              ? " · <span class='gray'>"
                + src.missing_keys.length
                + " field(s) fell back to dashboard defaults</span>"
              : "")
           + "</div>";
    }} else {{
      html += "<div class='crit-section' style='font-size:11px;"
           + "color:#e3b341;margin-bottom:10px;border:1px solid "
           + "rgba(227,179,65,0.30);background:rgba(227,179,65,0.08);"
           + "border-radius:4px;padding:6px 10px;'>"
           + "● Dashboard defaults — this bot hasn't reported its live "
           + "config, so values below may not match what the bot is "
           + "actually applying."
           + "</div>";
    }}

    html += "<div class='crit-section' style='font-size:13px;"
         + "line-height:1.55;color:#c9d1d9;margin-bottom:14px;'>"
         + "Before this bot opens a position it runs every contract "
         + "through four gates: <b>(1) does the model have an edge "
         + "worth taking</b>, <b>(2) is the market healthy enough to "
         + "fill at a fair price</b>, <b>(3) does the trade fit inside "
         + "today's risk budget</b>, and <b>(4) is the auto-hedge "
         + "armed to close the position</b>. Every check below must "
         + "pass on the chosen side (YES or NO); a single failure "
         + "drops the bet."
         + "</div>";

    // Probability-bounds is stored as [low, high] in cents → render
    // the two extremes as a price band the bot will trade in.
    const pb = va.prob_bounds_cents;
    const pbStr = (Array.isArray(pb) && pb.length === 2)
      ? pb[0] + "¢–" + pb[1] + "¢" : "—";

    html += "<div class='crit-section'>"
         + "<h4>1 · Edge &amp; EV — does the model think the price is wrong?</h4>"
         + bullets([
           ["Min model confidence",
            "skip-band around 50/50; the model's blended probability has to land outside this band before the bot considers either side.",
            "skip if p ∈ [" + fmtPctF(ed.min_model_confidence) + ", "
              + fmtPctF(1 - (ed.min_model_confidence || 0)) + "]"],
           ["Min expected value per contract",
            "expected $ return on a $1 contract after subtracting half the spread. Filters thin-margin trades where slippage eats the edge.",
            "≥ $" + fmtNum(ed.min_ev_per_contract)],
           ["Min edge over break-even",
            "buffer above the price-implied break-even probability. The model has to win meaningfully more often than the price says it has to, not just barely more often.",
            "≥ " + fmtPctF(ed.min_prob_edge_over_breakeven)],
           ["Min raw model edge",
            "raw (un-blended) model probability has to clear the ask by this much, so a market-dominated blend can't mask a thin underlying edge.",
            "≥ " + fmtPctF(ed.min_raw_model_edge)],
           ["Max entry price",
            "hard cap on the per-contract price the bot will pay. Above this, the loss-vs-gain ratio is too punishing even on a positive-EV call.",
            "≤ " + fmtCash(ed.max_entry_price_cents)],
         ])
         + "</div>";

    html += "<div class='crit-section'>"
         + "<h4>2 · Market health — is the book good enough to trade?</h4>"
         + bullets([
           ["Min book depth",
            "total contracts resting across the YES + NO order book within 3¢ of the touch. Avoids markets where the bot's own order would move the price.",
            "≥ " + fmtNum(va.min_book_depth_contracts) + " contracts"],
           ["Max spread",
            "ceiling on YES-ask minus NO-ask. Wide spreads mean Kalshi can't even tell us a real price — the bot won't bet into them.",
            "≤ " + fmtNum(va.max_spread_cents, "¢")],
           ["Time-to-close window",
            "trade only when the contract has enough time left to play out but not so much that the edge has time to erode before settle.",
            fmtMinH(va.min_minutes_to_close) + " – " + fmtMinH(va.max_minutes_to_close)],
           ["Probability bounds",
            "skip contracts already priced as near-certain or near-impossible. They pay too little to be worth the tail risk even when the edge is real.",
            pbStr],
           ["Min volume",
            "minimum contracts traded so far. Brand-new markets with zero volume have unreliable mid prices.",
            "≥ " + fmtNum(va.min_volume)],
           ["Min open interest",
            "real positions held by other traders. Confirms there are counterparties on this contract, not just the bot's own echo on a thin book.",
            "≥ " + fmtNum(va.min_open_interest)],
           ["Min depth at best ask",
            "size resting at the exact ask the bot would lift. Ensures the bot can fill its bet size without walking up the book.",
            "≥ " + fmtNum(va.min_depth_at_best_ask) + " contracts"],
           ["Basis-risk strike window",
            "skip trades whose underlying is too close to the contract's strike — those settle on noise instead of the model's view.",
            "±" + fmtNum(va.basis_risk_strike_window_dollars)],
           ["Basis-risk time window",
            "the basis-risk filter only applies inside the final few hours before settlement; farther out the underlying has room to move.",
            "< " + fmtNum(va.basis_risk_max_hours_to_close, "h") + " to close"],
         ])
         + "</div>";

    html += "<div class='crit-section'>"
         + "<h4>3 · Risk caps — does this trade fit in the budget?</h4>"
         + bullets([
           ["Fixed bet size",
            "every position the bot opens is the same $ size, not scaled by edge magnitude.",
            fmtCash(rk.bet_size_cents)],
           ["Max concurrent open positions",
            "ceiling on the number of simultaneous open contracts. Prevents racking up correlated exposure across the strike ladder.",
            "≤ " + fmtNum(rk.max_open_positions)],
           ["Max total exposure",
            "$ ceiling on the combined entry cost of all open positions. New trades skip when adding them would breach this cap.",
            "≤ " + fmtCash(rk.max_total_exposure_cents)],
           ["Max bets per day",
            "throttle on how many fresh positions the bot can open in 24h. Brakes against runaway loops if the model gets stuck endorsing the same contract.",
            "≤ " + fmtNum(rk.max_bets_per_day)],
           ["Cooldown on same market",
            "minimum wait before re-entering a contract after closing a position on it. Stops flap-trades when the price moves through break-even repeatedly.",
            "≥ " + fmtSec(rk.cooldown_seconds_same_market)],
         ])
         + "</div>";

    html += "<div class='crit-section'>"
         + "<h4>4 · Auto-hedge / exit rules — when does the bot leave?</h4>"
         + bullets([
           ["Auto-hedger on/off",
            "kill switch for the exit monitor. When off, positions ride to settlement and the bot accepts the binary outcome.",
            fmtNum(hg.enabled)],
           ["Profit-lock",
            "close the position once the mark has gained enough cents above entry. Locks in realised profit instead of giving it back if the edge fades.",
            "+" + fmtNum(hg.profit_lock_cents, "¢")],
           ["Stop-loss",
            "close the position once the mark has dropped enough cents below entry. Caps the per-trade downside if the model turns out wrong.",
            "−" + fmtNum(hg.stop_loss_cents, "¢")],
           ["Hedge size fraction",
            "fraction of the original position to close when a trigger fires. Full-exit on the whole bet, or scale half off and let the rest run.",
            fmtPctF(hg.hedge_size_fraction)],
         ])
         + "</div>";

    html += "<div class='crit-section' style='font-size:11px;color:#8b949e;'>"
         + "Every contract the bot considers must clear sections 1, 2, "
         + "and 3 before a bet is placed; once open, section 4 decides "
         + "when the bot exits. Click <b>Why?</b> on any open position "
         + "to see the actual values that cleared each gate at "
         + "entry-time for that specific bet."
         + "</div>";
    return html;
  }}
  function showRules(btn) {{
    if (!critOverlay || !critModal) return;
    // Prefer per-button data-rules payload; fall back to the global
    // window.__BUY_CRITERIA__ stash so callers without explicit
    // configs still open the same modal.
    let data = (window.__BUY_CRITERIA__ || {{}});
    try {{
      const local = btn.dataset && btn.dataset.rules
        ? JSON.parse(btn.dataset.rules) : null;
      if (local && Object.keys(local).length) data = local;
    }} catch (e) {{}}
    const h3 = critModal.querySelector("h3");
    if (h3) h3.textContent = "Buy criteria & validators";
    if (critTicker) critTicker.textContent = "";
    if (critBody)   critBody.innerHTML = buildRulesHTML(data);
    critOverlay.hidden = false;
    critModal.hidden   = false;
  }}

  // Contract-rules popup body. Kalshi's API only exposes a short
  // rules_primary string (and sometimes a rules_secondary block);
  // the *full* rules live in Kalshi's web UI. Render a clear link
  // to the market's "View full rules" page on Kalshi as the primary
  // affordance, with the cached short text shown as quick context.
  function buildContractRulesHTML(d) {{
    let html = "";
    if (d.kalshi_url) {{
      html += "<div class='crit-section'>"
           + "<a href='" + d.kalshi_url + "' target='_blank' "
           + "rel='noopener noreferrer' "
           + "style='display:inline-block;padding:8px 14px;"
           + "background:#1f6feb;color:#fff;text-decoration:none;"
           + "border-radius:6px;font-weight:600;font-size:13px;'>"
           + "View full rules on Kalshi ↗</a>"
           + "<div class='gray' style='font-size:11px;margin-top:6px;'>"
           + "Kalshi's web UI shows the complete rules text, including "
           + "settlement sources and edge cases.</div></div>";
    }}
    if (d.primary) {{
      html += "<div class='crit-section'><h4>Quick summary</h4>"
           + "<div style='font-size:13px;line-height:1.6;color:#c9d1d9;'>"
           + d.primary.split(/\\n/).map(function (p) {{
               return "<p style='margin:0 0 10px 0;'>" + p + "</p>";
             }}).join("")
           + "</div></div>";
    }}
    if (d.secondary) {{
      html += "<div class='crit-section'><h4>Additional details</h4>"
           + "<div style='font-size:13px;line-height:1.6;color:#c9d1d9;'>"
           + d.secondary.split(/\\n/).map(function (p) {{
               return "<p style='margin:0 0 10px 0;'>" + p + "</p>";
             }}).join("")
           + "</div></div>";
    }}
    if (!html) {{
      html = "<div class='crit-section'>"
           + "<span class='gray'>No rules cached yet.</span></div>";
    }}
    return html;
  }}
  function showContractRules(btn) {{
    if (!critOverlay || !critModal) return;
    let data = {{}};
    try {{
      data = JSON.parse(btn.dataset.contractRules || "{{}}");
    }} catch (e) {{}}
    const h3 = critModal.querySelector("h3");
    if (h3) h3.textContent = "Contract rules";
    if (critTicker) critTicker.textContent = "";
    if (critBody)   critBody.innerHTML = buildContractRulesHTML(data);
    critOverlay.hidden = false;
    critModal.hidden   = false;
  }}

  document.addEventListener("click", function (e) {{
    const contractBtn = e.target.closest(".contract-rules-btn");
    if (contractBtn) {{
      e.preventDefault();
      showContractRules(contractBtn);
      return;
    }}
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
        // Seasons is also cross-bot — no per-bot view, so navigate to
        // a clean ?tab=seasons URL (drops ?bot= and ?period=). Avoids
        // showing an empty panel when the user clicks Seasons from a
        // bot-scoped page that doesn't render the panel.
        if (key === "seasons") {{
          window.location.href = "?tab=seasons";
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


def _week_change_pct(rollup: dict) -> Tuple[str, str]:
    """Return (text, css_class) for the Home tab's "Week change" card.

    Compares lifetime net P&L now to where it stood seven days ago
    (now − this_week_pnl). Positive = account is up vs. last week.
    Returns ('—', 'gray') when there's no baseline to compare against
    (week-ago P&L was zero) so the card doesn't show a misleading ∞%.
    """
    this_week = rollup.get("this_week_pnl_cents", 0) or 0
    lifetime = rollup.get("net_pnl_cents", 0) or 0
    week_ago = lifetime - this_week
    if week_ago == 0:
        return ("—", "gray")
    pct = (this_week / abs(week_ago)) * 100.0
    cls = "green" if pct > 0 else ("red" if pct < 0 else "gray")
    sign = "+" if pct > 0 else ("−" if pct < 0 else "")
    return (f"{sign}{abs(pct):.1f}%", cls)


def _render_home_summary_cards(out: List[str], rollup: dict) -> None:
    """Emit the Home tab's 6 headline cards: Active bets, Active
    contracts, Active bots, Money spent (active bets only), Potential
    earnings, Week change %. All values are always-current — the Home
    tab has no period filter, and every dollar/contract card mirrors
    a column total in the Active bets table directly below.
    """
    active_bets = rollup.get("active_bets", 0)
    active_contracts = rollup.get("active_contracts", 0)
    active_bots = rollup.get("active_bots", 0)
    money_spent = rollup.get("active_money_spent_cents", 0)
    potential = rollup.get("potential_gain_cents", 0)
    week_text, week_cls = _week_change_pct(rollup)
    out.append("<div class='row'>")
    out.append(f"<div class='card'><div class='label' "
               f"title='Live count of currently-open positions across "
               f"all bots.'>"
               f"Active bets</div>"
               f"<div class='value' id='card-active-bets'>"
               f"{active_bets}</div></div>")
    out.append(f"<div class='card'><div class='label' "
               f"title='Total number of contracts held across "
               f"currently-open positions.'>"
               f"Active contracts</div>"
               f"<div class='value' id='card-active-contracts'>"
               f"{active_contracts}</div></div>")
    out.append(f"<div class='card'><div class='label' "
               f"title='Distinct bots that have at least one bet in "
               f"the Active bets table below.'>"
               f"Active bots</div>"
               f"<div class='value' id='card-active-bots'>"
               f"{active_bots}</div></div>")
    out.append(f"<div class='card'><div class='label' "
               f"title='Sum of the Entry cost column across the "
               f"Active bets table (entry × contracts + Kalshi entry "
               f"fee).'>"
               f"Money spent</div>"
               f"<div class='value' id='card-money-spent'>"
               f"{fmt_signed_cents(-money_spent)}</div></div>")
    out.append(f"<div class='card'><div class='label' "
               f"title='Sum of the Potential gain column across the "
               f"Active bets table ((100 − entry) × contracts − "
               f"entry fee).'>"
               f"Potential gain</div>"
               f"<div class='value green' id='card-potential-earnings'>"
               f"+{fmt_signed_cents(potential).lstrip('+')}</div></div>")
    out.append(f"<div class='card'><div class='label' "
               f"title='Change in lifetime net P&amp;L over the last "
               f"7 days, expressed as a percent of what it was a "
               f"week ago.'>"
               f"Week change</div>"
               f"<div class='value {week_cls}' id='card-week-change'>"
               f"{week_text}</div></div>")
    out.append("</div>")


def _render_summary_cards(out: List[str], rollup: dict,
                           id_suffix: str = "",
                           show_closed_contracts: bool = False) -> None:
    """Emit the 6-card headline row used on Home and History tabs.

    All values come from ``rollup`` (period-scoped) except
    ``active_bets``, which is always the live cross-bot open count.

    ``id_suffix`` is appended to each card's DOM id so the same row
    can render twice on one page without colliding. The empty default
    matches the existing snapshot-poller selectors on Home; the
    History instance uses ``"-history"`` so the poller skips it and
    the cards refresh only on full-page reload (which is fine since
    closed-bet rollups change rarely).

    ``show_closed_contracts`` swaps the first card from the live
    "Active bets" count to "Closed contracts" — total contracts
    bought across positions closed in the period. Used on the History
    tab where past activity matters more than the current open count.
    """
    net = rollup.get("period_net_pnl_cents", 0)
    pnl_cls = "green" if net > 0 else ("red" if net < 0 else "gray")
    win_pct = rollup.get("period_win_pct", 0.0)
    has_closed = (rollup.get("period_wins", 0)
                  + rollup.get("period_losses", 0)) > 0
    win_cls = ("green" if win_pct > 0.5
               else ("red" if has_closed and win_pct < 0.5 else "gray"))
    win_pct_str = f"{win_pct*100:.0f}%" if has_closed else "—"
    closed_bets = (rollup.get("period_wins", 0)
                   + rollup.get("period_losses", 0))
    money_spent = rollup.get("period_money_spent_cents", 0)
    money_gained = rollup.get("period_money_gained_cents", 0)
    closed_contracts = rollup.get("period_contracts_bought", 0)
    out.append("<div class='row compact'>")
    if show_closed_contracts:
        out.append(f"<div class='card'><div class='label' "
                   f"title='Total number of contracts bought across "
                   f"positions closed in the selected period.'>"
                   f"Closed contracts</div>"
                   f"<div class='value' "
                   f"id='card-closed-contracts{id_suffix}'>"
                   f"{closed_contracts}</div></div>")
    else:
        out.append(f"<div class='card'><div class='label' "
                   f"title='Live count of currently-open positions across "
                   f"all bots. Not affected by the period filter.'>"
                   f"Active bets</div>"
                   f"<div class='value' id='card-active-bets{id_suffix}'>"
                   f"{rollup['active_bets']}</div></div>")
    out.append(f"<div class='card'><div class='label'>"
               f"Closed bets</div>"
               f"<div class='value' id='card-closed-bets{id_suffix}'>"
               f"{closed_bets}</div></div>")
    out.append(f"<div class='card'><div class='label' "
               f"title='Total cost basis of every position opened in "
               f"the period (entry × contracts).'>"
               f"Money spent</div>"
               f"<div class='value' id='card-money-spent{id_suffix}'>"
               f"{fmt_signed_cents(-money_spent)}</div></div>")
    out.append(f"<div class='card'><div class='label' "
               f"title='Total payout received from positions closed in "
               f"the period (entry × contracts + realized P&amp;L).'>"
               f"Money gained</div>"
               f"<div class='value green' id='card-money-gained{id_suffix}'>"
               f"+{fmt_signed_cents(money_gained).lstrip('+')}</div></div>")
    out.append(f"<div class='card'><div class='label'>"
               f"P&amp;L</div>"
               f"<div class='value {pnl_cls}' id='card-net-pnl{id_suffix}'>"
               f"{fmt_signed_cents(net)}</div></div>")
    out.append(f"<div class='card'><div class='label' "
               f"title='Wins divided by closed bets in the selected "
               f"period. 0-100%; above 50% means winning more than "
               f"losing.'>"
               f"Total win %</div>"
               f"<div class='value {win_cls}' id='card-win-pct{id_suffix}'>"
               f"{win_pct_str}</div></div>")
    out.append("</div>")


def _render_summary(out: List[str], rollup: dict, active_bets: List[dict],
                    history: List[dict],
                    period_key: str = "all",
                    current_bot: str = "",
                    available_bots: List[dict] | None = None,
                    hedge_cfg: dict | None = None) -> None:
    """Section 1 — global cross-bot summary. The dropdown above the
    headline cards is a Bot navigator: selecting any bot jumps to its
    Watchlist tab so the user can dive into per-bot detail without
    hunting through the bot-card grid below. The summary cards
    themselves stay scoped to All-time totals (the period filter
    moved off Home — the History tab still has its period filter).
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

    out.append("<div class='section'><h2>Summary — across all bots</h2>"
               "<div class='body summary-body'>")

    # Bot-jump dropdown moved above the tab bar (per user request) so
    # it applies to every tab in one place.

    # ── Headline cards ────────────────────────────────────────────────
    _render_home_summary_cards(out, rollup)

    # Active bets list — same table used in the per-bot view below.
    # Same circle-i info button as on the Watchlist tab — opens the
    # shared rules popup ("What does the bot need before it'll buy?")
    # using window.__BUY_CRITERIA__ as the data source.
    out.append(
        "<h3 class='subhead' "
        "style='display:flex;align-items:center;gap:8px;'>"
        "Active bets "
        "<button type='button' class='criteria-rules-btn' "
        "title=\"What does the bot need before it'll buy?\">i</button>"
        "</h3>"
    )
    # Scroll container — keeps the Summary's active-bets table from
    # pushing the bot-card grid off-screen when many bots have
    # positions open at once. Max-height was picked so ~6 rows are
    # visible before the user has to scroll; matches the watchlist
    # scroll idiom used elsewhere on the page.
    out.append("<div class='summary-active-scroll'>")
    _render_active_bets_table(out, active_bets,
                                empty_msg="No active bets right now.",
                                hedge_cfg=hedge_cfg)
    out.append("</div>")

    out.append("</div></div>")


def _render_notifications_panel(out: List[str],
                                  limit: int = 5) -> None:
    """Recent auto-pause notifications from the regime monitor.

    Reads the tail of ``data/regime_notifications.jsonl`` and renders
    a compact panel above the bot card grid when there's at least one
    entry. Silent (no rendering) when the file is empty or missing —
    a no-news-is-good-news posture so the Home tab stays calm.
    """
    from . import regime_monitor
    notes = regime_monitor.read_notifications(limit=limit)
    if not notes:
        return
    out.append("<div class='notifications-panel'>")
    out.append(
        "<div class='notifications-head'>"
        "<span class='notifications-title'>Recent auto-pauses</span>"
        "<span class='small gray'>The regime monitor auto-disabled "
        "these bots after 3 consecutive 30-day windows of negative "
        "P&amp;L. Use the bot card toggle to resume.</span></div>"
    )
    out.append("<ul class='notifications-list'>")
    for n in notes:
        ts = (n.get("ts") or "")[:19].replace("T", " ")
        bot_name = n.get("bot_name") or n.get("bot_key") or "—"
        reason = n.get("reason") or ""
        out.append(
            f"<li><span class='notification-ts'>{html.escape(ts)}</span>"
            f"<span class='notification-bot'>"
            f"{html.escape(str(bot_name))}</span>"
            f"<span class='notification-reason'>"
            f"{html.escape(reason)}</span></li>"
        )
    out.append("</ul></div>")


def _render_bot_cards(out: List[str], rollup: dict,
                        bot_models: List[dict] | None,
                        period_label: str) -> None:
    """Per-bot card grid for the Performance tab. Compact, clickable —
    each card is an anchor to the bot's Watchlist tab. Cards align on
    a fixed grid (auto-fit minmax 280px) so they share row + column
    edges. Contract rules live on the Watchlist tab to keep these
    cards skimmable.

    Top-right slot carries an on/off toggle (active / paused state).
    Bot name + current underlying value + series_ticker stack down
    the left so the most-glanceable info (which bot, what's its
    current quote, which Kalshi series) sits together. Paused cards
    get a dimmed style + PAUSED badge so disabled bots are visually
    distinct from the running ones.
    """
    if not bot_models:
        out.append("<div class='empty'>No bot data yet.</div>")
        return

    # Pull the latest per-bot enable state so the toggles render in
    # their current position. Defaults to enabled = True for bots
    # without a stored state.
    from . import bot_state
    bot_states = bot_state.get_all_states()

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
        # Upper-right meta slot. For Kalshi sim.db bots this is the
        # series_ticker prefix (e.g. "KXNBAGAME"). JSON-source sport bots
        # (tennis, table-tennis) read the same series_ticker field from
        # config — set it to the bot's label of choice (e.g.
        # "BASELINEBREAK" for tennis, "TABLETENNIS" for table-tennis).
        series_ticker = b.get("series_ticker") or "—"
        # Period-scoped net P&L from this bot's per-bot summary row.
        perf = perf_by_name.get(name, {})
        gain_loss = perf.get("period_net_pnl_cents", 0) or 0
        gl_cls = ("green" if gain_loss > 0
                   else ("red" if gain_loss < 0 else "gray"))
        gl_str = fmt_signed_cents(gain_loss)
        # Each card is a link to the bot's Models tab — the deeper
        # per-bot view (feature importance, calibration, confusion
        # matrix, all features used to make decisions). Tennis routes
        # through its own page since its model has its own renderer.
        if not bot_key:
            href = "#"
        elif b.get("dashboard_type") in ("tennis", "survivor", "billboard"):
            href = f"?bot={html.escape(bot_key)}&tab=models"
        else:
            href = f"?tab=models&bot={html.escape(bot_key)}"

        # Compute drift-badge HTML (if any) up-front so it can be
        # rendered inline with the bot name in the card header.
        # Drift = |training accuracy − live actual-win-%| > 10pp on
        # n ≥ 10 closed bets.
        # Tennis is exempt from the drift badge — its paper-trade
        # ledger settles probabilistically and the user explicitly
        # asked not to surface drift on the tennis card.
        ACTUAL_WIN_MIN_N = 10
        DRIFT_PP_THRESHOLD = 0.10
        drift_html = ""
        # Natural-gas is also exempt per user request — the bot's
        # "training accuracy" is a per-strike grid average that doesn't
        # line up apples-to-apples with the live actual-win-%, so the
        # drift badge fires spuriously on every load.
        _drift_exempt = b.get("dashboard_type") in ("tennis", "survivor", "billboard") \
            or bot_key == "natural-gas"
        if m and not _drift_exempt:
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
        # Regime status pill — rolling edge-health check sourced from
        # the bot's last 90 days of closed bets. Sits inline with the
        # bot name so the user can scan the grid for which bots are
        # currently making money. Tennis-style adapters don't have
        # the schema we need, so they get no pill (rather than a
        # misleading "no data" badge on every load).
        regime_html = ""
        if b.get("dashboard_type") not in ("tennis", "survivor", "billboard"):
            regime = bot_regime_status(b.get("db_path") or "")
            if regime.get("status") and regime["status"] != "gray":
                regime_html = (
                    f"<span class='regime-pill regime-{regime['status']}' "
                    f"title='{html.escape(regime.get('reason') or '')}'>"
                    f"{html.escape(regime.get('label') or '')}</span>"
                )

        # Forecast-staleness badge — flags when the bot's stored
        # ``current_gas_price`` model snapshot has drifted away from
        # the live Kalshi-implied spot (50¢-crossover strike on the
        # series' most-imminent event). Triggered above a $0.20
        # gap, which is roughly 1σ for the natgas residual and well
        # outside normal noise. Catches the "EIA feed lag" failure
        # mode where the bot's price-input is days behind reality
        # and its forecast (and every model_prob_yes) is anchored
        # to a stale level.
        #
        # Only applies to sim.db-style bots that record a scalar
        # underlying — same exclusion as the regime pill above.
        staleness_html = ""
        if (b.get("dashboard_type") not in
                ("tennis", "survivor", "billboard", "whale", "rules-parser")
                and m and m.get("current_gas_price") is not None
                and b.get("series_ticker")):
            try:
                bot_price = float(m["current_gas_price"])
                divisor = float((b.get("display") or {}).get("divisor", 1.0)) or 1.0
                bot_price_in_market_units = bot_price / divisor
                from . import kalshi_client as _kc
                implied, _err = _kc.get_implied_spot(b["series_ticker"])
            except Exception:  # noqa: BLE001
                bot_price_in_market_units = None
                implied = None
            if implied is not None and bot_price_in_market_units is not None:
                # Use $0.20 absolute threshold for natgas-shape series;
                # this is roughly 1σ of the bot's residual and well
                # outside normal intra-day noise. For markets where
                # the scalar isn't a $/MMBTU price (e.g. jobless claims
                # in thousands), the same absolute gap reads as a
                # different number of "units" — acceptable as a v1
                # heuristic, can be refined per-bot later.
                gap = abs(implied - bot_price_in_market_units)
                if gap >= 0.20:
                    fmt = lambda v: f"{v:.2f}"  # noqa: E731
                    tip = (
                        f"Bot's stored current value: "
                        f"{fmt(bot_price_in_market_units)} · "
                        f"Live Kalshi-implied: {fmt(implied)} · "
                        f"Gap: {gap:+.2f} — model may be reading a "
                        f"stale upstream data feed."
                    )
                    staleness_html = (
                        f"<span class='stale-badge' "
                        f"title='{html.escape(tip)}'>⚠ stale</span>"
                    )
        # Toggle state for this bot — defaults to enabled = True.
        bot_state_entry = bot_states.get(bot_key) or {}
        bot_enabled = bool(bot_state_entry.get("enabled", True))
        card_classes = ["bot-card"]
        if not bot_enabled:
            card_classes.append("bot-card-paused")
        # Card header layout per user spec: bot name on top, ticker
        # directly below. The previous underlying-value "price" line
        # was dropped — surfacing $4.50 / 211K / 0.38pp on a model-
        # performance card was confusing (it's not the bot's *score*,
        # it's the upstream market value), and the bots that don't
        # track a scalar (tennis / survivor) were forced to render
        # "—" there anyway.
        paused_badge = (
            "<span class='paused-badge' title='Bot is paused — toggle "
            "on to resume taking bets.'>PAUSED</span>"
            if not bot_enabled else ""
        )
        out.append(
            f"<a class='{' '.join(card_classes)}' href='{href}' "
            f"data-bot-key='{html.escape(bot_key)}'>"
        )
        out.append("<div class='bot-card-head'>")
        out.append("<div class='bot-card-head-left'>")
        out.append(
            f"<div class='bot-name'>{html.escape(name)}{regime_html}"
            f"{drift_html}{staleness_html}{paused_badge}</div>"
        )
        out.append(
            f"<div class='bot-meta'>{html.escape(series_ticker)}</div>"
        )
        out.append("</div>")
        # Toggle switch — click is intercepted by JS so the parent
        # anchor's navigation doesn't fire. data-* attributes hold the
        # mutable state the JS flips.
        toggle_attrs = (
            f"data-bot-key='{html.escape(bot_key)}' "
            f"data-enabled='{'1' if bot_enabled else '0'}' "
            f"aria-pressed='{'true' if bot_enabled else 'false'}' "
            "type='button' onclick='toggleBotState(event, this)'"
        )
        out.append(
            f"<button class='bot-toggle' {toggle_attrs}>"
            f"<span class='bot-toggle-track'>"
            f"<span class='bot-toggle-knob'></span>"
            f"</span>"
            f"</button>"
        )
        out.append("</div>")

        if not m:
            out.append("<dl><dt class='gray'>Model</dt>"
                       "<dd class='gray' style='grid-column:span 3;text-align:left;'>"
                       "no snapshot yet</dd></dl>")
        else:
            a_wins = int(m.get("actual_wins") or 0)
            a_losses = int(m.get("actual_losses") or 0)
            a_total = a_wins + a_losses
            # Show the real percentage at any sample size (per user
            # request). n=0 still shows "—" to distinguish "no data
            # yet" from "0%". The drift-badge logic above keeps its
            # n ≥ 10 guard since drift needs a meaningful sample.
            a_pct = a_wins / a_total if a_total > 0 else None
            if a_total > 0:
                a_str = f"{a_pct*100:.0f}%"
                a_cls = ("green" if a_pct > 0.55
                         else ("red" if a_pct < 0.45 else ""))
            else:
                a_str = "—"
                a_cls = "gray"
            features = int(m.get("feature_count") or 0)
            # Sample sizes: training-set rows the model fit on, and the
            # held-out test rows the headline metrics were measured on.
            # Both come from model_snapshots (sqlite bots) or metrics.json
            # (tennis-style adapters via fetch_latest_model). Cell reads
            # "—" when a bot hasn't been retrained since the schema added
            # the column.
            def _fmt_n(v):
                try:
                    return f"{int(v):,}" if v else "—"
                except (TypeError, ValueError):
                    return "—"
            train_str = _fmt_n(m.get("rows_train"))
            test_str = _fmt_n(m.get("rows_test"))
            out.append("<dl>")
            out.append(f"<dt>Accuracy</dt><dd>{_fmt_pct(m.get('classifier_accuracy'), 1)}</dd>"
                        f"<dt>F1</dt><dd>{_fmt_pct(m.get('training_f1'))}</dd>")
            out.append(f"<dt>Precision</dt><dd>{_fmt_pct(m.get('training_precision'))}</dd>"
                        f"<dt>ROC AUC</dt><dd>{_fmt_pct(m.get('training_roc_auc'))}</dd>")
            out.append(f"<dt>Recall</dt><dd>{_fmt_pct(m.get('training_recall'))}</dd>"
                        f"<dt>Features</dt><dd>{features}</dd>")
            out.append(f"<dt title='Training-set size — number of historical observations the model fit on. More rows = more market regimes covered.'>Train rows</dt>"
                        f"<dd>{train_str}</dd>"
                        f"<dt title='Held-out test-set size — observations the headline metrics were measured on.'>Test rows</dt>"
                        f"<dd>{test_str}</dd>")
            out.append(f"<dt>Actual win %</dt><dd class='{a_cls}'>{a_str}</dd>"
                        f"<dt>P&amp;L</dt><dd class='{gl_cls}'>{gl_str}</dd>")
            out.append("</dl>")

        # Footer hints at the click affordance — same idiom as the
        # ticker cells in the watchlist (subtle "go here" signal).
        out.append("<div class='bot-card-foot'>"
                   "<span>View model</span>"
                   "<span class='arrow'>›</span>"
                   "</div>")
        out.append("</a>")  # /bot-card
    out.append("</div>")  # /bot-cards-grid


def _render_active_bets_table(out: List[str], bets: List[dict],
                              empty_msg: str = "No active bets.",
                              show_bot: bool = True,
                              chart_link: bool = False,
                              hedge_cfg: dict | None = None,
                              hide_settled: bool = True,
                              watchlist: List[dict] | None = None,
                              event_title: str | None = None,
                              is_sport_bot: bool = False,
                              display: dict | None = None) -> None:
    """Shared renderer used by both Section 1 (cross-bot summary) and
    the per-bot view inside the Watchlist tab. Columns:
        Opened | [Bot] | Ticker | Question | Contracts | Side
        | Entry cost | Current | Potential gain | Closes in
    The Bot column is skipped when ``show_bot`` is False (per-bot view
    where the bot is implied by the surrounding section). Entry cost /
    Current / Potential gain are in dollars (per-position totals).

    ``hedge_cfg`` is accepted for parity with callers but the table no
    longer renders a per-row hedge column — the actual hedge
    execution lives in ``hedge_monitor.py`` which closes any position
    that crosses the configured profit-lock / stop-loss thresholds.
    Once closed, the position drops out of this table and shows up
    on the History tab with ``exit_reason='hedge'``.

    ``hide_settled=True`` (default) filters out positions whose
    Kalshi-ticker-encoded settlement date is more than 1 hour in
    the past — zombie open positions whose bot didn't record the
    settle event. The hedge daemon closes them on its next tick
    so they appear on History; this renderer hides them in the
    interim so the active-bets table reflects only positions the
    bot is actually exposed on.

    ``chart_link=True`` makes each row clickable and stamps the
    chart-overlay attributes (``data-ticker``, ``data-strike``,
    ``data-yes-prob``) so the watchlist hero chart can draw a
    threshold line at the bet's strike (or entry probability for
    sport bots) — the same affordance the strike-ladder rows have.
    """
    # Drop already-settled positions when requested (default). The
    # Summary already pre-filters before calling us; per-bot
    # Watchlist views and the standard renderer rely on this guard.
    if hide_settled:
        bets = [
            b for b in bets
            if (
                (b.get("minutes_to_close")
                 if b.get("minutes_to_close") is not None
                 else minutes_to_close_from_ticker(b.get("ticker"))) or 0
            ) >= -60
        ]
    if not bets:
        out.append(f"<div class='empty'>{html.escape(empty_msg)}</div>")
        return
    # Look-up of the ticker table underneath so the active-bets row
    # mirrors the same title + side text the watchlist shows for the
    # same ticker. Per-bot views pass `watchlist`; the cross-bot
    # Summary tab leaves it None and falls back to the legacy fields
    # stored on each position.
    wl_by_ticker: dict = {
        (w.get("ticker") or ""): w for w in (watchlist or [])
    }
    use_event_title = bool((display or {}).get("watchlist_title_use_event")
                           and event_title)
    bot_th = "<th>Bot</th>" if show_bot else ""
    # Column layout: ``Title`` carries Kalshi's published contract title
    # (the YES question text). ``Side`` carries the YES / NO badge.
    # No separate ``Question`` column — Title already names the
    # contract, and on sport rows it would just restate the matchup.
    tbody_attrs = " id='wl-active-tbody' data-chart-link='1'" if chart_link else ""
    out.append("<table><thead><tr>"
               f"<th>Opened</th>{bot_th}<th>Ticker</th>"
               "<th>Title</th>"
               "<th>Side</th>"
               "<th class='num' title='Number of contracts in this position — the size of your bet.'>My contracts</th>"
               "<th class='num' title='Model probability for our side at entry — what the model thought before we bet.'>Model prob</th>"
               "<th class='num' title='Implied probability of our side at entry (= entry price in ¢).'>Entry prob</th>"
               "<th class='num' title='Implied probability of our side right now, taken from the market mid.'>Current prob</th>"
               "<th class='num' title='Entry prob × contracts + Kalshi entry fee — total cash out at open'>Entry cost</th>"
               "<th class='num' title='(100¢ − entry) × contracts − entry fee — gross profit if our side wins'>Potential gain</th>"
               "<th class='num' title='Time until the contract resolves'>Closes in</th>"
               "<th></th>"
               f"</tr></thead><tbody{tbody_attrs}>")
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
        # Entry-cost cell shows base − fee inline so the user reads
        # both pieces as cash outflows (a positive fee is still cash
        # leaving the account at open). Tooltip explains.
        entry_cost_cell = (
            f"<td class='num red' title='Entry prob × contracts + "
            f"Kalshi entry fee — total cash out at open'>"
            f"−${entry_cost_base:.2f}"
            f"<span class='entry-fee'> − ${entry_fee_dollars:.2f}</span>"
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
        # Universal fallback: parse the settlement date out of the
        # ticker (Kalshi encodes ``YYMMMDD`` after the series prefix).
        # Catches tennis paper bets — those sim positions don't record
        # an expected_expiration_time, so the previous tennis-adapter
        # lookup against live_state.json missed them once the match
        # rolled off the live state.
        if mtc is None:
            mtc = minutes_to_close_from_ticker(b.get("ticker"))
        # Sign / color logic for potential gain — usually positive
        # (winning side pays $1 minus entry minus fees), but very
        # high entry prices on extreme strikes can flip negative.
        pg_sign = "+" if potential_gain >= 0 else "−"
        pg_cls  = "green" if potential_gain >= 0 else "red"
        # Bot cell: link to that bot's watchlist tab so the user can
        # jump from a row in the cross-bot active-bets summary into
        # the per-bot detail view in one click. Tennis routes through
        # its own page; the rest land on the standard watchlist tab.
        bot_key = b.get("_bot_key") or ""
        bot_dt = b.get("_dashboard_type") or "standard"
        if bot_key:
            if bot_dt == "tennis":
                href = f"?bot={html.escape(bot_key)}&tab=watchlist"
            else:
                href = f"?tab=watchlist&bot={html.escape(bot_key)}"
            bot_link = (f"<a href='{href}' class='bot-link'>"
                        f"{html.escape(bot_name)}</a>")
        else:
            bot_link = html.escape(bot_name)
        bot_td = (f"<td>{bot_link}</td>" if show_bot else "")
        # Build the "why was this bet chosen" payload from entry-time
        # snapshot fields recorded on the position. JS reads this from
        # data-criteria on click and populates the shared modal.
        m_yes = b.get("model_yes_prob_at_entry")
        k_yes = b.get("kalshi_yes_prob_at_entry")
        # Backfill from decision_json for bots whose schema doesn't
        # have dedicated columns (natural-gas stashes both probs
        # inside the JSON payload). Same fallback fetch_bet_history
        # applies to closed rows.
        if (m_yes is None or k_yes is None) and b.get("decision_json"):
            try:
                _dj = json.loads(b["decision_json"]) if isinstance(
                    b["decision_json"], str) else b["decision_json"]
                if isinstance(_dj, dict):
                    if m_yes is None and _dj.get("model_prob") is not None:
                        m_yes = _dj["model_prob"]
                    if k_yes is None and _dj.get("kalshi_implied_prob") is not None:
                        k_yes = _dj["kalshi_implied_prob"]
            except (TypeError, ValueError, json.JSONDecodeError):
                pass
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
        # Title cell: mirror the watchlist row underneath for the same
        # ticker so the two tables always show identical text. Falls
        # through to the position's stored title and finally to a
        # derived question when no watchlist match is available (the
        # cross-bot Summary tab doesn't supply a watchlist).
        wl_row = wl_by_ticker.get(b.get("ticker") or "")
        if use_event_title:
            title_text = event_title
        elif wl_row and wl_row.get("title"):
            title_text = wl_row.get("title")
        else:
            title_text = b.get("_title") or b.get("title") or ""
        if not title_text:
            match_text = b.get("_match") or _match_text_from_ticker(b.get("ticker"))
            side_player = b.get("_side_player")
            if side_player:
                title_text = (f"{match_text} — bet on {side_player}"
                               if match_text else side_player)
            elif match_text:
                tri = _side_tricode_from_ticker(b.get("ticker"), side)
                title_text = (f"{match_text} — bet on {tri}"
                               if tri else match_text)
            else:
                title_text = question
        # Closes in: for tennis paper bets, _minutes_to_close is provided
        # by the tennis adapter (derived from expected_expiration_time);
        # for Kalshi bots the simulator already supplies minutes_to_close.
        # Chart-overlay data attrs (only emitted when ``chart_link``) —
        # the watchlist hero chart's row-click hook reads these and
        # draws a horizontal threshold line at the bet's strike (non-
        # sport) or entry probability (sport).
        if chart_link:
            tr_attrs = f" data-ticker='{html.escape(b.get('ticker') or '')}'"
            try:
                if strike_low is not None:
                    tr_attrs += f" data-strike='{float(strike_low):.6f}'"
            except (TypeError, ValueError):
                pass
            try:
                # YES bet's entry price = implied YES probability; NO
                # bet's entry price implies (100 - entry)% YES.
                yes_prob = (entry / 100.0 if side == "YES"
                             else (100 - entry) / 100.0)
                tr_attrs += f" data-yes-prob='{yes_prob:.4f}'"
            except (TypeError, ValueError):
                pass
        else:
            tr_attrs = ""
        # Model prob cell — renders the side-adjusted model probability
        # from the criteria computation above. Tooltip surfaces the
        # implied edge (model − Kalshi) when both are available.
        if model_p is None:
            model_prob_cell = "<td class='num gray'>—</td>"
        else:
            tip = ""
            if edge_pts is not None:
                tip = (f" title='Model edge {edge_pts:+.1f}pp vs entry "
                       f"price'")
            model_prob_cell = (
                f"<td class='num'{tip}>{model_p*100:.0f}%</td>"
            )
        # Side cell: for sport bots, mirror the watchlist row underneath
        # (team tricode on top, "vs opponent" beneath). The team we're
        # actually rooting for sits on top — on a NO bet that's the
        # team the YES side is *against* — so the user reads the bet
        # the same way Kalshi's market page reads it. The badge color
        # (green YES / red NO) is preserved as a left-edge accent so
        # the bet direction stays visible at a glance. Non-sport bots
        # keep the legacy YES/NO badge — the watchlist's third column
        # is a different field (Question) over there.
        if is_sport_bot:
            # Sport rows prefer pre-supplied labels (tennis carries
            # _yes_label / _no_label = player names); fall back to the
            # NBA tricode parser for KXNBAGAME tickers.
            yes_label = b.get("_yes_label") or _side_tricode_from_ticker(
                b.get("ticker"), "YES")
            no_label = b.get("_no_label") or _side_tricode_from_ticker(
                b.get("ticker"), "NO")
            if side == "YES":
                side_team, opp_team = yes_label, no_label
            else:
                side_team, opp_team = no_label, yes_label
            if side_team:
                # No badge_cls colour on the player name — Side reads
                # as identity, not direction.
                side_cell = (
                    f"<td class='active-side-team'>"
                    f"<strong>{html.escape(str(side_team))}</strong>"
                    f"<br><span class='small gray'>vs "
                    f"{html.escape(str(opp_team))}</span></td>"
                )
            else:
                side_cell = (
                    f"<td><span class='badge {badge_cls}'>{side}</span></td>"
                )
        else:
            side_cell = (
                f"<td><span class='badge {badge_cls}'>{side}</span></td>"
            )
        # In-game model pill — only renders when the live model has a
        # confident view. Sits inline with the Side cell so it reads
        # as additional context on what we're holding.
        in_game = b.get("_in_game") or {}
        ig_action = (in_game.get("action") or "").lower()
        ig_pill = ""
        if in_game and ig_action in {"exit_now", "let_run", "hold"}:
            cls_map = {
                "exit_now": "ig-red",
                "let_run": "ig-green",
                "hold": "ig-yellow",
            }
            label_map = {
                "exit_now": "EXIT",
                "let_run": "RUN",
                "hold": "HOLD",
            }
            ig_pill = (
                f"<span class='in-game-pill {cls_map[ig_action]}' "
                f"title='{html.escape(in_game.get('reason') or '')}'>"
                f"{label_map[ig_action]}</span>"
            )
        # Inject the pill into the side cell so the table doesn't
        # gain a column.
        if ig_pill:
            side_cell = side_cell.replace("</td>", f" {ig_pill}</td>", 1)

        out.append(
            f"<tr{tr_attrs}><td>{html.escape(opened)}</td>"
            f"{bot_td}"
            f"<td class='mono'>{ticker_cell_html(b.get('ticker'))}</td>"
            f"<td>{html.escape(title_text)}</td>"
            f"{side_cell}"
            f"<td class='num'>{contracts}</td>"
            f"{model_prob_cell}"
            f"{entry_prob_cell}"
            f"{current_prob_cell}"
            f"{entry_cost_cell}"
            f"<td class='num {pg_cls}' title='"
            f"(100¢ − {entry}¢) × {contracts} contracts − ${entry_fee_dollars:.2f} fee = "
            f"${(100 - entry) * contracts / 100.0:.2f} − ${entry_fee_dollars:.2f} = "
            f"${potential_gain:.2f}. Entry fee already paid; settlement "
            f"at 100¢ or 0¢ has zero exit fee.'>"
            f"{pg_sign}${abs(potential_gain):.2f}</td>"
            f"<td class='num'>{time_to_close_str(mtc)}</td>"
            f"<td><button type='button' class='criteria-btn' "
            f"title='Why was this bet chosen?' "
            f"data-criteria='{criteria_json}'>i</button></td>"
            f"</tr>"
        )
    out.append("</tbody></table>")


def _parse_season_dt(value: str | None) -> datetime | None:
    """Best-effort ISO-8601 → aware datetime. Accepts the ``Z`` suffix
    that PyYAML / config files commonly use; returns None on parse
    failure so a broken season block just hides the affected card
    rather than 500-ing the page."""
    if not value:
        return None
    s = str(value).strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _humanize_countdown(delta_seconds: float) -> str:
    """Server-side initial value for the countdown cells, in the same
    ``Xd Xh Xm Xs`` shape the JS tick() function renders. Floors to 0
    on negatives so the placeholder never reads as a negative duration
    if the JS hasn't run yet."""
    s = max(0, int(delta_seconds))
    days, rem = divmod(s, 86400)
    hours, rem = divmod(rem, 3600)
    mins, secs = divmod(rem, 60)
    return f"{days}d {hours}h {mins}m {secs}s"


def _humanize_duration(delta_seconds: float) -> str:
    """Render a duration as ``Xd`` or ``Xw Yd`` for season-length cells.
    Used as the static "Length" value on the Seasons tab; the live
    countdown values are rendered client-side in JS."""
    if delta_seconds <= 0:
        return "—"
    days = int(delta_seconds // 86400)
    if days < 14:
        return f"{days}d"
    weeks, rem_days = divmod(days, 7)
    if rem_days == 0:
        return f"{weeks}w"
    return f"{weeks}w {rem_days}d"


def _render_seasons_panel(out: List[str], available_bots: List[dict]) -> None:
    """One card per league. A bot can declare multiple ``seasons:``
    entries when its bot trades multiple Kalshi competitions
    (tennis = ATP + WTA tours, darts = Premier League + PDC World
    Championship, etc.) — each entry renders as its own card.
    Cards whose end time has already passed are hidden so the tab
    stays focused on what's actually trading; the countdown only
    flips between "Starts in …" and "Ends in …". Live leagues sort
    above upcoming ones."""
    now = datetime.now(timezone.utc)

    cards: List[tuple[dict, dict, datetime, datetime]] = []
    for bot in available_bots:
        for season in (bot.get("seasons") or []):
            start = _parse_season_dt(season.get("start"))
            end = _parse_season_dt(season.get("end"))
            if not start or not end:
                continue
            if end < now:
                # Season already wrapped up; update the YAML to the
                # next iteration to bring this league back to the tab.
                continue
            cards.append((bot, season, start, end))

    # Live leagues first (soonest end), then upcoming (soonest start).
    def _sort_key(item):
        _, _, s, e = item
        return (0, e) if s <= now <= e else (1, s)
    cards.sort(key=_sort_key)

    out.append(
        "<div class='section'><h2>Seasons</h2><div class='body'>"
    )
    if not cards:
        out.append(
            "<div class='empty'>No leagues have an active or "
            "upcoming season configured. Update a "
            "<code>seasons:</code> block in "
            "<code>config/dashboard.yaml</code> to bring a card "
            "back.</div>"
        )
        out.append("</div></div>")
        return

    out.append(
        "<p class='small gray' style='margin: 0 0 14px 0;'>"
        "One card per league. Live leagues "
        "(<span class='green'>Ends in</span>) sit above upcoming "
        "ones (<span class='yellow'>Starts in</span>)."
        "</p>"
    )
    out.append("<div class='season-grid'>")
    for bot, season, start, end in cards:
        bot_label = bot.get("name") or bot.get("key")
        season_name = season.get("name") or bot_label
        length = _humanize_duration((end - start).total_seconds())
        start_ms = int(start.timestamp() * 1000)
        end_ms = int(end.timestamp() * 1000)
        start_str = start.strftime("%b %-d, %Y")
        end_str = end.strftime("%b %-d, %Y")
        if now < start:
            init_status, init_color = "Upcoming", "yellow"
            init_label, init_value = "Starts in", _humanize_countdown(
                (start - now).total_seconds())
        else:
            init_status, init_color = "In season", "green"
            init_label, init_value = "Ends in", _humanize_countdown(
                (end - now).total_seconds())
        total = (end - start).total_seconds()
        if now <= start or total <= 0:
            init_pct = 0.0
        else:
            init_pct = max(0.0, min(100.0,
                ((now - start).total_seconds() / total) * 100.0))
        bot_href = f"?bot={html.escape(bot['key'])}&tab=watchlist"
        out.append(
            f"<div class='season-card' data-season-card "
            f"data-start='{start_ms}' data-end='{end_ms}'>"
            f"<div class='season-card-head'>"
            f"<a class='season-bot' href='{bot_href}'>"
            f"{html.escape(bot_label)}</a>"
            f"<span class='status-pill {init_color}' "
            f"data-season-status>{html.escape(init_status)}</span>"
            f"</div>"
            f"<div class='season-name'>{html.escape(season_name)}</div>"
            f"<div class='season-countdown'>"
            f"<div class='season-countdown-label' "
            f"data-season-countdown-label>"
            f"{html.escape(init_label)}</div>"
            f"<div class='season-countdown-value {init_color}' "
            f"data-season-countdown-value>"
            f"{html.escape(init_value)}</div>"
            f"</div>"
            f"<div class='season-progress'>"
            f"<div class='season-progress-fill' "
            f"data-season-progress-fill "
            f"style='width: {init_pct:.2f}%;'></div>"
            f"</div>"
            f"<div class='season-meta'>"
            f"<div><span class='season-meta-label'>Start</span>"
            f"<span class='season-meta-value'>"
            f"{html.escape(start_str)}</span></div>"
            f"<div><span class='season-meta-label'>End</span>"
            f"<span class='season-meta-value'>"
            f"{html.escape(end_str)}</span></div>"
            f"<div><span class='season-meta-label'>Length</span>"
            f"<span class='season-meta-value'>"
            f"{html.escape(length)}</span></div>"
            f"</div>"
            f"</div>"
        )
    out.append("</div>")  # /season-grid
    out.append("</div></div>")  # /body /section


def _render_history_chart(out: List[str], history: List[dict],
                            period_key: str = "all",
                            current_bot: str = "") -> None:
    """Daily net P&L line chart for the History tab. The closed-bet
    ledger is embedded as JSON on the SVG node; the JS buckets bets
    by UTC day and draws the line client-side. The period dropdown
    sits inline above the chart as its toolbar.
    """
    points: List[list] = []
    for h in history:
        ts_str = h.get("exited_at")
        pnl = h.get("realized_pnl_cents")
        if not ts_str or pnl is None:
            continue
        # Same idiom as the bet-history rows: chop to 19 chars to drop
        # fractional seconds + tz suffix, then parse as UTC. Timestamps
        # in sim.db are recorded in UTC.
        try:
            dt = datetime.fromisoformat(ts_str[:19])
            epoch = int(dt.replace(tzinfo=timezone.utc).timestamp())
        except (TypeError, ValueError):
            continue
        try:
            pnl_int = int(pnl)
        except (TypeError, ValueError):
            continue
        points.append([epoch, pnl_int])
    points_payload = html.escape(
        json.dumps(points, separators=(",", ":")),
        quote=True,
    )
    out.append("<div class='history-chart-section'>")
    # Toolbar row: chart title on the left, period selector on the
    # right — visually anchors the filter to the chart it controls.
    out.append("<div class='history-chart-toolbar'>")
    out.append("<div class='history-chart-title'>Daily net P&amp;L</div>")
    _render_period_filter(out, period_key, current_bot=current_bot,
                            tab_key="history")
    out.append("</div>")
    out.append(
        "<div class='history-chart-wrap'>"
        f"<svg data-history-chart data-points='{points_payload}' "
        "width='100%' height='260' viewBox='0 0 800 260' "
        "preserveAspectRatio='none' style='display:block'></svg>"
        "</div>"
    )
    out.append("</div>")  # /history-chart-section


def _render_history_attribution(out: List[str],
                                  history: List[dict]) -> None:
    """P&L attribution panels for the History tab — small breakdown
    tables that slice the closed-bet ledger four ways: by bot, by
    month, by side (YES/NO), and by predicted-EV bucket. Each panel
    tries to answer "where is the P&L coming from?" so the user can
    spot whether profit is broad (likely real edge) or concentrated
    in one slice (likely a quirk).

    Respects whatever period filter the History tab is currently on
    — ``history`` is already the period-scoped list the caller passes
    into the chart and ledger renderers below.
    """
    if not history:
        return  # Empty state already covered by the ledger block.

    def _row(label: str, bets: List[dict]) -> dict:
        n = len(bets)
        total = sum((b.get("realized_pnl_cents") or 0) for b in bets)
        wins = sum(1 for b in bets if (b.get("realized_pnl_cents") or 0) > 0)
        return {"label": label, "n": n, "total_cents": total,
                "win_pct": (wins / n) if n else 0.0}

    def _emit_table(title: str, hint: str, rows: List[dict]) -> None:
        out.append(
            f"<div class='attribution-panel'>"
            f"<h3 class='subhead'>{html.escape(title)} "
            f"<span class='small gray'>{html.escape(hint)}</span></h3>"
        )
        if not rows:
            out.append("<div class='empty'>No data in this slice.</div>"
                       "</div>")
            return
        out.append(
            "<table><thead><tr>"
            "<th>Bucket</th>"
            "<th class='num'>Bets</th>"
            "<th class='num'>P&amp;L</th>"
            "<th class='num'>Win %</th>"
            "</tr></thead><tbody>"
        )
        for r in rows:
            pnl_cls = ("green" if r["total_cents"] > 0
                        else ("red" if r["total_cents"] < 0 else "gray"))
            win = r["win_pct"]
            win_cls = ("green" if win > 0.5
                        else ("red" if r["n"] > 0 and win < 0.5 else "gray"))
            win_str = f"{win*100:.0f}%" if r["n"] > 0 else "—"
            dollars = r["total_cents"] / 100.0
            sign = "+" if r["total_cents"] > 0 else (
                "−" if r["total_cents"] < 0 else "")
            out.append(
                f"<tr><td>{html.escape(r['label'])}</td>"
                f"<td class='num'>{r['n']}</td>"
                f"<td class='num {pnl_cls}'>{sign}${abs(dollars):.2f}</td>"
                f"<td class='num {win_cls}'>{win_str}</td></tr>"
            )
        out.append("</tbody></table></div>")

    # ── Slice: by bot ───────────────────────────────────────────────
    by_bot: dict[str, List[dict]] = {}
    for h in history:
        by_bot.setdefault(h.get("_bot_name") or "—", []).append(h)
    bot_rows = sorted(
        (_row(name, bets) for name, bets in by_bot.items()),
        key=lambda r: r["total_cents"], reverse=True,
    )

    # ── Slice: by month (YYYY-MM) ───────────────────────────────────
    by_month: dict[str, List[dict]] = {}
    for h in history:
        ts = (h.get("exited_at") or "")[:7]  # YYYY-MM
        if ts:
            by_month.setdefault(ts, []).append(h)
    month_rows = [_row(m, bets) for m, bets in
                  sorted(by_month.items(), reverse=True)]

    # ── Slice: by side (YES vs NO) ──────────────────────────────────
    by_side: dict[str, List[dict]] = {}
    for h in history:
        side = (h.get("side") or "").upper() or "—"
        by_side.setdefault(side, []).append(h)
    side_rows = [_row(s, bets) for s, bets in sorted(by_side.items())]

    # ── Slice: by predicted EV bucket (decimal $/contract) ──────────
    ev_buckets = [
        ("< 0¢",   -10.0,  0.0),
        ("0–2¢",    0.0,   0.02),
        ("2–4¢",    0.02,  0.04),
        ("4–7¢",    0.04,  0.07),
        ("7–10¢",   0.07,  0.10),
        ("10¢+",    0.10,  10.0),
        ("untagged", None, None),  # bets without recorded EV
    ]
    ev_rows: List[dict] = []
    for label, lo, hi in ev_buckets:
        bucket_bets: List[dict] = []
        for h in history:
            ev = h.get("expected_ev_at_entry")
            if lo is None:
                if ev is None:
                    bucket_bets.append(h)
                continue
            if ev is None:
                continue
            try:
                ev_f = float(ev)
            except (TypeError, ValueError):
                continue
            if ev_f < lo or ev_f >= hi:
                continue
            bucket_bets.append(h)
        if bucket_bets:
            ev_rows.append(_row(label, bucket_bets))

    out.append(
        "<h3 class='subhead' style='margin-top:14px;'>"
        "P&amp;L attribution "
        "<span class='small gray'>(where the closed-bet P&amp;L came "
        "from in the selected period)</span></h3>"
    )
    out.append("<div class='attribution-grid'>")
    _emit_table("By bot", "which bots carried this period",
                 bot_rows)
    _emit_table("By month", "calendar month of exit",
                 month_rows)
    _emit_table("By side", "YES vs NO",
                 side_rows)
    _emit_table("By predicted EV", "entry-EV bucket vs realized P&L",
                 ev_rows)
    out.append("</div>")


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
        "<th title='Date the contract was opened (UTC).'>Opened</th>"
        "<th title='Date the contract was closed (UTC).'>Closed</th>"
        "<th>Bot</th><th>Ticker</th>"
        "<th>Title</th>"
        "<th>Side</th>"
        "<th class='num' title='Model probability for the side we bet on, recorded at entry.'>Model p</th>"
        "<th class='num'>Entry</th><th class='num'>Exit</th>"
        "<th class='num'>Contracts</th>"
        "<th class='num' title='Net EV per contract at entry: (model_p − entry_price) − half-spread. "
        "Positive = +EV trade.'>Entry EV</th>"
        "<th class='num'>P&amp;L</th>"
        "<th>Outcome</th>"
        "</tr></thead><tbody>"
    )

    def render_row(b):
        # Both timestamps to ISO YYYY-MM-DD HH:MM:SS (UTC). Slicing at
        # 19 chops the fractional seconds + tz offset for compactness.
        opened = (b.get("opened_at") or "")[:19].replace("T", " ")
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
        if ev is None or round(float(ev), 2) == 0:
            ev_str = "0"
            ev_cls = "gray"
        else:
            ev_sign = "+" if ev > 0 else "−"
            ev_str = f"{ev_sign}${abs(ev):.2f}"
            ev_cls = _ev_status(ev)[0]
        # Title cell: Kalshi-published contract title; falls back to a
        # derived "matchup — bet on X" or the strike question text
        # when no Kalshi title is recorded on this row.
        title_text = b.get("_title") or b.get("title") or ""
        if not title_text:
            match_text = b.get("_match") or _match_text_from_ticker(b.get("ticker"))
            side_player = b.get("_side_player")
            if side_player:
                title_text = (f"{match_text} — bet on {side_player}"
                               if match_text else side_player)
            elif match_text:
                tri = _side_tricode_from_ticker(b.get("ticker"), side)
                title_text = (f"{match_text} — bet on {tri}"
                               if tri else match_text)
            else:
                title_text = question
        # Bot cell: link to the bot's watchlist (same idiom as the
        # active-bets table).
        bot_key = b.get("_bot_key") or ""
        bot_dt = b.get("_dashboard_type") or "standard"
        if bot_key:
            if bot_dt == "tennis":
                bot_href = f"?bot={html.escape(bot_key)}&tab=watchlist"
            else:
                bot_href = f"?tab=watchlist&bot={html.escape(bot_key)}"
            bot_cell = (f"<td><a href='{bot_href}' class='bot-link'>"
                         f"{html.escape(bot_name)}</a></td>")
        else:
            bot_cell = f"<td>{html.escape(bot_name)}</td>"
        # ``merged_trade_count > 1`` flags a history row that collapses
        # multiple flap-trades (open + close + re-open on the same
        # match/strike) into one. Surface a small "×N" badge next to
        # the ticker so the user can tell a merged row from a single
        # trade — and the P&L column makes sense as the *net* across
        # those N trades.
        merged_n = int(b.get("merged_trade_count") or 1)
        if merged_n > 1:
            merged_badge = (
                f"<span class='merged-badge' "
                f"title='Net P&L across {merged_n} trades on this same "
                f"ticker (bot re-opened the position after each close). "
                f"Click for the raw trade IDs.'>×{merged_n}</span>"
            )
        else:
            merged_badge = ""
        return (f"<tr><td>{html.escape(opened)}</td>"
                f"<td>{html.escape(closed)}</td>"
                f"{bot_cell}"
                f"<td class='mono'>{ticker_cell_html(b.get('ticker'))}"
                f"{merged_badge}</td>"
                f"<td>{html.escape(title_text)}</td>"
                f"<td><span class='badge {badge_cls}'>{side}</span></td>"
                f"<td class='num'>{mp_str}</td>"
                f"<td class='num'>{entry}c</td>"
                f"<td class='num'>{cents_or_dash(exit_c)}</td>"
                f"<td class='num'>{contracts}</td>"
                f"<td class='num {ev_cls}'>{ev_str}</td>"
                f"<td class='num {pnl_cls_}'>{fmt_signed_cents(pnl)}</td>"
                f"<td class='{pnl_cls_}'>{outcome}</td></tr>")

    # All rows go into a single table — the History tab's
    # max-height: 640px scroll container handles overflow. The old
    # ``Show N more`` collapsible details was confusing when the
    # scroll already implies "everything's in here".
    # ``shown_initially`` is retained on the function signature for
    # back-compat with callers (Section 5 / Home use a smaller
    # window) but the History tab passes the full list through.
    out.append(head)
    for b in history:
        out.append(render_row(b))
    out.append("</tbody></table>")


def _render_bot_filter(out: List[str], available_bots: List[dict],
                       current_bot: str,
                       period_key: str = "all",
                       select_id: str = "bot-select",
                       include_all_option: bool = False,
                       tab_key: str = "watchlist") -> None:
    """Bot selector dropdown — used on both the Home tab (as the
    "jump to a bot's watchlist" navigator) and on the per-bot
    Watchlist tab. Native <select> for keyboard-friendliness.

    Each instance gets its own ``select_id`` so multiple dropdowns
    coexist on the same DOM (the page's tab panels all live in one
    document). The page's onchange JS finds them via
    ``[data-bot-select]`` so it doesn't need to know the id.

    ``include_all_option`` adds a leading "All bots" entry that lands
    on ``/`` — used on Home so the dropdown can return the user to
    the cross-bot summary view after browsing into a watchlist.
    """
    period_qs = (f"&period={html.escape(period_key)}"
                 if period_key and period_key != "all" else "")
    out.append("<div class='bot-filter-bar'>")
    out.append(f"<label for='{html.escape(select_id)}' "
               f"class='filter-label'>Bot</label>")
    out.append(
        f"<select id='{html.escape(select_id)}' class='bot-select' "
        f"data-bot-select>"
    )
    if include_all_option:
        # "All bots" returns to the cross-bot home page.
        all_url = f"/?period={html.escape(period_key)}" if period_key and period_key != "all" else "/"
        sel = " selected" if not current_bot else ""
        out.append(
            f"<option value='{html.escape(all_url)}'{sel}>All bots</option>"
        )
    for b in available_bots:
        avail = b.get("available", True)
        suffix = "" if avail else " (no data)"
        sel = " selected" if b["key"] == current_bot else ""
        # Tennis routes through its own renderer; both branches build
        # the same URL shape so the dropdown jumps the user into the
        # right per-bot tab regardless of dashboard_type.
        href = (f"?bot={html.escape(b['key'])}"
                f"&tab={html.escape(tab_key)}{period_qs}")
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


# ── Models page (per-bot deep-dive) ──────────────────────────────────────
def _find_training_artifact(db_path: str, *names: str) -> Path:
    """Locate a trainer-written file for a bot. Different bots stage
    their training artifacts in different siblings of ``data/sim.db``:

    - newer bots (nba, cpi, claims) land them next to sim.db (``data/``)
    - retail-gas writes into ``artifacts/`` instead
    - natural-gas / peak-load writes models into ``models/``

    Returns the first existing path; falls back to the ``data/`` location
    if nothing is found so that callers can pass it to readers that
    handle missing files (they'll just return empty results).
    """
    base = Path(db_path).parent
    candidates: List[Path] = []
    for name in names:
        candidates.extend([
            base / name,
            base.parent / "artifacts" / name,
            base.parent / "models" / name,
        ])
    for p in candidates:
        if p.exists():
            return p
    return Path(db_path).parent / names[0]


def _read_feature_importance(csv_path: str) -> List[dict]:
    """Parse the bot's feature_importance.csv into a list of feature
    dicts: {feature, mean_importance, positive_folds, selected}.

    Tolerates missing/empty files by returning an empty list — the
    renderer shows an empty-state in that case rather than crashing.

    Two on-disk schemas exist in the wild:
      • Newer bots (nba, cpi, claims, natural-gas): a clean
        ``feature, mean_importance, positive_folds, selected`` header.
      • Retail-gas (legacy): per-fold columns + a ``mean`` /
        ``positive_folds`` / ``eligible`` triple, with an unnamed
        first column carrying the feature name.

    The reader picks whichever set of columns exists so every bot's
    feature_importance.csv renders the same way on the dashboard.
    """
    p = Path(csv_path)
    if not p.exists():
        return []
    out: List[dict] = []
    try:
        with p.open("r") as f:
            rd = csv.DictReader(f)
            fields = list(rd.fieldnames or [])
            name_key = ("feature" if "feature" in fields
                        else ("" if "" in fields
                              else (fields[0] if fields else None)))
            imp_key = ("mean_importance" if "mean_importance" in fields
                       else ("mean" if "mean" in fields else None))
            sel_key = ("selected" if "selected" in fields
                       else ("eligible" if "eligible" in fields else None))
            for row in rd:
                feat_name = (row.get(name_key) or "") if name_key is not None else ""
                try:
                    imp = float(row.get(imp_key) or 0.0) if imp_key else 0.0
                except (TypeError, ValueError):
                    imp = 0.0
                try:
                    pf = int(float(row.get("positive_folds") or 0))
                except (TypeError, ValueError):
                    pf = 0
                sel_raw = row.get(sel_key) if sel_key else None
                sel = str(sel_raw or "").strip().lower() in (
                    "true", "1", "yes",
                )
                out.append({
                    "feature": feat_name,
                    "mean_importance": imp,
                    "positive_folds": pf,
                    "selected": sel,
                })
    except (OSError, csv.Error):
        return []
    return out


def _holdout_confidence(pairs: List[Tuple[float, int]]) -> dict:
    """Translate the trainer's held-out predictions into a
    sample-size-driven confidence tier for the metrics on the model
    page. Returns a dict with ``tier`` (none/low/moderate/good/high),
    a CSS colour, a one-word label, and a sentence-long ``reason``
    that explains *why* the user should (or shouldn't) trust the
    headline numbers.

    The thresholds borrow standard rules-of-thumb for binary-classifier
    holdout sample sizes:
      • <30 predictions or minority class <5 → noisy, can flip 5+ pts
      • <100 / minority <20 → directionally meaningful, ±2-3 pts
      • <500 → stable to ~1 pt
      • ≥500 → calibration deciles each carry enough data
    """
    n = len(pairs)
    n_pos = sum(1 for _, y in pairs if y == 1)
    n_neg = n - n_pos
    minority = min(n_pos, n_neg) if n else 0
    if n == 0:
        return {
            "tier": "none", "color": "#8b949e",
            "label": "No held-out data",
            "reason": ("This bot's trainer hasn't written a "
                       "holdout_predictions.csv yet — the metrics on "
                       "this page can't be confidence-graded."),
            "n": 0, "n_pos": 0, "n_neg": 0,
        }
    if n < 30 or minority < 5:
        return {
            "tier": "low", "color": "#f85149",
            "label": "Low confidence",
            "reason": (f"Only {n} held-out predictions"
                       + (f" (minority class = {minority})"
                          if minority < 5 else "")
                       + " — the accuracy / ROC / calibration figures "
                       "below are noisy at this sample size and can "
                       "swing 5+ percentage points across retrains."),
            "n": n, "n_pos": n_pos, "n_neg": n_neg,
        }
    if n < 100 or minority < 20:
        return {
            "tier": "moderate", "color": "#d29922",
            "label": "Moderate confidence",
            "reason": (f"{n} held-out predictions ({n_pos} positives / "
                       f"{n_neg} negatives) — directionally meaningful "
                       "but the per-decile calibration bins still carry "
                       "wide error bars. Treat headline metrics as "
                       "±2-3 pts."),
            "n": n, "n_pos": n_pos, "n_neg": n_neg,
        }
    if n < 500:
        return {
            "tier": "good", "color": "#3fb950",
            "label": "Good confidence",
            "reason": (f"{n} held-out predictions ({n_pos} positives / "
                       f"{n_neg} negatives) — sample size is large "
                       "enough that the headline accuracy / ROC AUC "
                       "are stable to within ~1 pt across retrains."),
            "n": n, "n_pos": n_pos, "n_neg": n_neg,
        }
    return {
        "tier": "high", "color": "#3fb950",
        "label": "High confidence",
        "reason": (f"{n:,} held-out predictions ({n_pos:,} positives / "
                   f"{n_neg:,} negatives) — enough data per "
                   "calibration decile to read at face value."),
        "n": n, "n_pos": n_pos, "n_neg": n_neg,
    }


def _render_confidence_card(out: List[str], conf: dict,
                             extra_lines: List[str] | None = None) -> None:
    """Render the held-out-data trust indicator at the top of the
    Model panel. Compact one-liner: row count + confidence tier in
    muted grey, with the tier label tinted by tier colour. The
    underlying ``_holdout_confidence`` dict still carries the full
    reasoning sentence in ``reason`` — it goes into the title
    tooltip so users can hover for the long-form explanation.
    """
    color = conf["color"]
    n = int(conf.get("n") or 0)
    if n <= 0:
        # No held-out data — keep the line so the user still sees
        # that the trainer hasn't written one yet.
        text = html.escape(conf.get("label", "No held-out data"))
        reason = html.escape(conf.get("reason", ""))
        out.append(
            f"<p class='small gray' "
            f"style='margin:0 0 12px 0;' title='{reason}'>"
            f"Held-out test set: <span style='color:{color};"
            f"font-weight:600;'>{text}</span></p>"
        )
    else:
        n_pos = int(conf.get("n_pos") or 0)
        n_neg = int(conf.get("n_neg") or 0)
        label = html.escape(conf.get("label", ""))
        reason = html.escape(conf.get("reason", ""))
        out.append(
            f"<p class='small gray' "
            f"style='margin:0 0 12px 0;' title='{reason}'>"
            f"Held-out test set: <b style='color:#c9d1d9;'>"
            f"{n:,} predictions</b> "
            f"<span class='gray'>({n_pos:,} positives / "
            f"{n_neg:,} negatives)</span> · "
            f"<span style='color:{color};font-weight:600;'>{label}</span>"
            f"</p>"
        )
    for line in (extra_lines or []):
        out.append(f"<p class='small gray' style='margin:0 0 4px 0;'>"
                   f"{line}</p>")


def calibration_from_holdout(pairs: List[Tuple[float, int]],
                               n_bins: int = 10) -> List[dict]:
    """Decile calibration on the trainer's held-out predictions:
    predicted-prob bin → observed positive-class rate. Same shape as
    the prior closed-bet version, just sourced from training-time
    evaluation instead of paper trades.
    """
    edges = [i / n_bins for i in range(n_bins + 1)]
    bins = [{"lo": edges[i], "hi": edges[i + 1],
             "n": 0, "wins": 0} for i in range(n_bins)]
    for p, y in pairs:
        idx = min(n_bins - 1, max(0, int(p * n_bins)))
        bins[idx]["n"] += 1
        if y == 1:
            bins[idx]["wins"] += 1
    return bins


def calibration_from_live_bets(db_path: str,
                                 n_bins: int = 10) -> List[dict]:
    """Live-bets calibration overlay for the Models tab.

    Pulls every closed position with a recorded model-yes-probability
    at entry and an outcome (realized_pnl_cents). Each bet is mapped
    back to the implicit "did the contract resolve YES?" event so the
    bucket curve matches the holdout calibration's semantics:
        YES bet won  → contract resolved YES
        YES bet lost → contract resolved NO
        NO  bet won  → contract resolved NO
        NO  bet lost → contract resolved YES
    Returns bins with the same shape as ``calibration_from_holdout``
    so a single render path can overlay both series.
    """
    if not Path(db_path).exists():
        return []
    try:
        with closing(_conn(db_path)) as c:
            cols = {r["name"] for r in
                    c.execute("PRAGMA table_info(positions)").fetchall()}
            if "model_yes_prob_at_entry" not in cols:
                return []
            rows = c.execute(
                "SELECT model_yes_prob_at_entry, side, realized_pnl_cents "
                "FROM positions "
                "WHERE status = 'closed' "
                "  AND model_yes_prob_at_entry IS NOT NULL "
                "  AND realized_pnl_cents IS NOT NULL"
            ).fetchall()
    except (sqlite3.OperationalError, sqlite3.DatabaseError):
        return []
    edges = [i / n_bins for i in range(n_bins + 1)]
    bins = [{"lo": edges[i], "hi": edges[i + 1],
             "n": 0, "wins": 0} for i in range(n_bins)]
    for r in rows:
        try:
            p_yes = float(r["model_yes_prob_at_entry"])
            side = (r["side"] or "").upper()
            pnl = int(r["realized_pnl_cents"])
        except (TypeError, ValueError):
            continue
        won = pnl > 0
        if side == "YES":
            resolved_yes = won
        elif side == "NO":
            resolved_yes = not won
        else:
            continue
        idx = min(n_bins - 1, max(0, int(p_yes * n_bins)))
        bins[idx]["n"] += 1
        if resolved_yes:
            bins[idx]["wins"] += 1
    return bins


def _read_holdout_predictions(csv_path: str) -> List[Tuple[float, int]]:
    """Load (predicted_prob, actual_label) pairs from a bot's
    holdout_predictions.csv. The trainer writes this file on each
    retrain — it carries the model's evaluation against the held-out
    historical test set, which is what the user sees as "the
    model's accuracy" on the Models tab.

    Returns an empty list when the file is missing or unreadable
    (e.g. a bot whose trainer hasn't been redeployed yet).
    """
    p = Path(csv_path)
    if not p.exists():
        return []
    out: List[Tuple[float, int]] = []
    try:
        with p.open("r") as f:
            rd = csv.DictReader(f)
            for row in rd:
                try:
                    prob = float(row.get("predicted_prob") or 0.0)
                    label = int(float(row.get("actual_label") or 0))
                except (TypeError, ValueError):
                    continue
                out.append((prob, 1 if label else 0))
    except (OSError, csv.Error):
        return []
    return out


def confusion_from_holdout(pairs: List[Tuple[float, int]],
                            threshold: float = 0.5) -> dict:
    """Build a confusion matrix from the trainer's held-out
    (prob, label) pairs. The model's "prediction" is whether the
    predicted prob exceeds the threshold; the "actual" is the
    historical ground-truth label.
    """
    out = {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "n": 0}
    for p, y in pairs:
        pred = 1 if p >= threshold else 0
        if pred == 1 and y == 1:
            out["tp"] += 1
        elif pred == 1 and y == 0:
            out["fp"] += 1
        elif pred == 0 and y == 0:
            out["tn"] += 1
        else:
            out["fn"] += 1
        out["n"] += 1
    return out


def fetch_per_strike_accuracy(db_path: str) -> List[dict]:
    """Accuracy by strike-band on closed bets. Strike floor/cap come
    from market_views (positions doesn't carry the strike directly).
    """
    if not Path(db_path).exists():
        return []
    try:
        with closing(_conn(db_path)) as c:
            rows = c.execute(
                "SELECT "
                "  (SELECT mv.strike_low FROM market_views mv "
                "    WHERE mv.ticker = p.ticker ORDER BY mv.id DESC LIMIT 1) "
                "    AS floor_strike, "
                "  (SELECT mv.strike_high FROM market_views mv "
                "    WHERE mv.ticker = p.ticker ORDER BY mv.id DESC LIMIT 1) "
                "    AS cap_strike, "
                "  (SELECT mv.direction FROM market_views mv "
                "    WHERE mv.ticker = p.ticker ORDER BY mv.id DESC LIMIT 1) "
                "    AS direction, "
                "  realized_pnl_cents "
                "FROM positions p "
                "WHERE p.status='closed'"
            ).fetchall()
    except (sqlite3.OperationalError, sqlite3.DatabaseError):
        return []
    grouped: dict = {}
    for r in rows:
        key = (r["floor_strike"], r["cap_strike"], r["direction"])
        g = grouped.setdefault(key, {
            "floor_strike": r["floor_strike"],
            "cap_strike": r["cap_strike"],
            "direction": r["direction"],
            "n": 0, "wins": 0, "losses": 0,
        })
        g["n"] += 1
        try:
            pnl = float(r["realized_pnl_cents"] or 0)
        except (TypeError, ValueError):
            pnl = 0.0
        if pnl > 0:
            g["wins"] += 1
        else:
            g["losses"] += 1
    out = []
    for g in grouped.values():
        if g["n"] == 0:
            continue
        g["accuracy"] = g["wins"] / g["n"]
        out.append(g)
    out.sort(key=lambda g: g["n"], reverse=True)
    return out


def roc_from_holdout(pairs: List[Tuple[float, int]]) -> List[dict]:
    """Sweep thresholds across the trainer's held-out predictions to
    produce an ROC curve. Returns a list of {fpr, tpr, threshold}
    dicts. Empty when the file is missing or only one class is
    present in the holdout.
    """
    if not pairs:
        return []
    n_pos = sum(1 for _, y in pairs if y == 1)
    n_neg = sum(1 for _, y in pairs if y == 0)
    if n_pos == 0 or n_neg == 0:
        return []
    sorted_pairs = sorted(pairs, key=lambda x: -x[0])
    tp = fp = 0
    points: List[dict] = [{"fpr": 0.0, "tpr": 0.0, "threshold": 1.0}]
    last_p: float | None = None
    for p, y in sorted_pairs:
        if last_p is not None and p != last_p:
            points.append({
                "fpr": fp / n_neg,
                "tpr": tp / n_pos,
                "threshold": last_p,
            })
        if y == 1:
            tp += 1
        else:
            fp += 1
        last_p = p
    points.append({"fpr": 1.0, "tpr": 1.0,
                    "threshold": sorted_pairs[-1][0]})
    return points


def fetch_model_overview(db_path: str, fi_path: str,
                          features: List[dict]) -> dict:
    """Roll-up of training-derived stats about the model: when it was
    last retrained, how many features were considered vs kept, how
    stable the kept features were across walk-forward folds, etc.

    Everything here comes from training-time artifacts (the saved
    model.pkl on disk, the feature_importance.csv that the trainer
    writes alongside it). No live Kalshi data is consulted, so the
    snapshot reads "what is this model, and how was it trained?"
    rather than "what's the model saying right now?".
    """
    out = {
        "last_retrained": None,
        "n_considered": len(features),
        "n_kept": sum(1 for f in features if f.get("selected")),
        "n_stable_all_folds": 0,
        "n_stable_4_folds": 0,
        "max_folds": 5,
        "top_feature": None,
        "top_importance": None,
        "snapshots_recorded": None,
        "snapshot_first": None,
        "snapshot_last": None,
    }
    # Last retrain — the model artifact's mtime is the canonical
    # signal since the trainer writes the file at the end of every
    # retrain. Different bots ship the artifact under different names
    # (model.pkl on the python-pickle bots, model.joblib on the
    # legacy gas-prices stack) and stage it under data/ vs artifacts/
    # vs models/ — fall through every plausible location.
    pkl = _find_training_artifact(
        db_path, "model.pkl", "model.joblib",
        "natgas_price.pkl", "peak_load.pkl",
    )
    if pkl.exists():
        try:
            mt = datetime.fromtimestamp(pkl.stat().st_mtime, tz=timezone.utc)
            out["last_retrained"] = mt.strftime("%Y-%m-%d %H:%M UTC")
        except (OSError, OverflowError):
            pass
    # Walk-forward fold stability (positive_folds is how many of the
    # 5 folds had positive permutation importance for that feature).
    if features:
        max_folds = max(int(f.get("positive_folds") or 0) for f in features)
        if max_folds:
            out["max_folds"] = max_folds
        kept = [f for f in features if f.get("selected")]
        out["n_stable_all_folds"] = sum(
            1 for f in kept
            if int(f.get("positive_folds") or 0) >= out["max_folds"]
        )
        out["n_stable_4_folds"] = sum(
            1 for f in kept
            if int(f.get("positive_folds") or 0) >= max(1, out["max_folds"] - 1)
        )
        top = max(kept, key=lambda f: f.get("mean_importance") or 0.0,
                   default=None)
        if top:
            out["top_feature"] = top.get("feature")
            out["top_importance"] = top.get("mean_importance")
    # Snapshot range — proxy for "how long has this model been live?".
    if Path(db_path).exists():
        try:
            with closing(_conn(db_path)) as c:
                row = c.execute(
                    "SELECT COUNT(*) AS n, MIN(captured_at) AS first, "
                    "       MAX(captured_at) AS last "
                    "FROM model_snapshots"
                ).fetchone()
            if row:
                out["snapshots_recorded"] = int(row["n"] or 0)
                out["snapshot_first"] = row["first"]
                out["snapshot_last"] = row["last"]
        except (sqlite3.OperationalError, sqlite3.DatabaseError):
            pass
    return out


def _svg_feature_importance(features: List[dict],
                             max_height_px: int = 600) -> str:
    """Horizontal-bar chart of every feature, sorted by mean_importance
    descending. Selected features render in green; rejected (didn't
    survive the walk-forward stability filter) in muted gray with a
    smaller bar so the user can see "tried, dropped" alongside "kept".
    """
    if not features:
        return ("<div class='empty'>"
                "Feature importance not yet written for this bot — "
                "the file lands after the next retrain.</div>")
    feats = sorted(features,
                    key=lambda f: f.get("mean_importance") or 0.0,
                    reverse=True)
    n = len(feats)
    row_h = 18
    pad_t, pad_b, pad_l, pad_r = 18, 24, 220, 60
    height = pad_t + pad_b + n * row_h
    width = 760
    inner_w = width - pad_l - pad_r
    # Normalize to the largest abs importance so the longest bar fills
    # most of the inner width.
    max_imp = max((abs(f.get("mean_importance") or 0.0) for f in feats),
                   default=1.0) or 1.0
    parts: List[str] = []
    parts.append(
        f"<svg viewBox='0 0 {width} {height}' "
        f"style='width:100%;height:auto;max-height:{max_height_px}px;"
        f"display:block;background:#0d1117;border:1px solid #21262d;"
        f"border-radius:6px;'>"
    )
    parts.append(f"<text x='{pad_l - 8}' y='{pad_t - 4}' fill='#8b949e' "
                  f"font-size='11' text-anchor='end'>feature</text>")
    parts.append(f"<text x='{pad_l + 4}' y='{pad_t - 4}' fill='#8b949e' "
                  f"font-size='11'>importance</text>")
    for i, f in enumerate(feats):
        y = pad_t + i * row_h + 2
        imp = f.get("mean_importance") or 0.0
        bar_w = max(1.0, abs(imp) / max_imp * inner_w)
        sel = bool(f.get("selected"))
        pf = int(f.get("positive_folds") or 0)
        # Green for selected (kept), muted for unselected (rejected).
        bar_color = "#3fb950" if sel else "#484f58"
        text_color = "#c9d1d9" if sel else "#8b949e"
        name = html.escape(f.get("feature") or "")
        # Truncate very long names with ellipsis (the full name lives
        # in the title attribute for hover).
        display_name = (name if len(name) <= 28
                          else name[:25] + "…")
        parts.append(
            f"<g><title>{name} · imp {imp:.4f} · {pf}/5 folds · "
            f"{'kept' if sel else 'rejected'}</title>"
            f"<text x='{pad_l - 8}' y='{y + 12}' fill='{text_color}' "
            f"font-size='11' text-anchor='end' "
            f"font-family='ui-monospace,SFMono-Regular,monospace'>"
            f"{display_name}</text>"
            f"<rect x='{pad_l}' y='{y + 3}' width='{bar_w:.1f}' "
            f"height='10' fill='{bar_color}' rx='1'/>"
            f"<text x='{pad_l + bar_w + 4:.1f}' y='{y + 12}' "
            f"fill='#8b949e' font-size='10'>{imp:.4f}</text>"
            f"</g>"
        )
    parts.append("</svg>")
    return "".join(parts)


def _svg_calibration(bins: List[dict],
                       live_bins: List[dict] | None = None) -> str:
    """Reliability diagram — predicted-prob bin midpoint on X, observed
    win-rate on Y, point size scales with bin sample count. Diagonal
    reference line shows perfect calibration.

    ``live_bins`` (optional) overlays a second series sourced from the
    bot's live closed-bet ledger. Holdout = blue (training-time
    expectation); live = orange (what the bot is actually getting).
    Divergence between the two is the drift signal.
    """
    populated = [b for b in bins if b.get("n", 0) > 0]
    live_populated = [b for b in (live_bins or []) if b.get("n", 0) > 0]
    if not populated and not live_populated:
        return ("<div class='empty'>Not enough closed bets yet to "
                "draw a calibration curve.</div>")
    width, height = 460, 320
    pad_l, pad_r, pad_t, pad_b = 50, 20, 24, 36
    inner_w = width - pad_l - pad_r
    inner_h = height - pad_t - pad_b
    parts: List[str] = []
    parts.append(
        f"<svg viewBox='0 0 {width} {height}' "
        f"style='width:100%;height:auto;max-height:340px;"
        f"display:block;background:#0d1117;border:1px solid #21262d;"
        f"border-radius:6px;'>"
    )
    # Axes — 0..1 both directions.
    parts.append(
        f"<line x1='{pad_l}' y1='{pad_t}' x2='{pad_l}' "
        f"y2='{pad_t + inner_h}' stroke='#21262d'/>"
        f"<line x1='{pad_l}' y1='{pad_t + inner_h}' "
        f"x2='{pad_l + inner_w}' y2='{pad_t + inner_h}' stroke='#21262d'/>"
    )
    # Gridlines + labels at each decile.
    for k in range(0, 11, 2):
        frac = k / 10.0
        x = pad_l + frac * inner_w
        y = pad_t + (1 - frac) * inner_h
        parts.append(
            f"<line x1='{x}' x2='{x}' y1='{pad_t}' "
            f"y2='{pad_t + inner_h}' stroke='#161b22'/>"
            f"<text x='{x}' y='{pad_t + inner_h + 14}' fill='#8b949e' "
            f"font-size='10' text-anchor='middle'>{int(frac*100)}%</text>"
            f"<line x1='{pad_l}' x2='{pad_l + inner_w}' "
            f"y1='{y}' y2='{y}' stroke='#161b22'/>"
            f"<text x='{pad_l - 6}' y='{y + 3}' fill='#8b949e' "
            f"font-size='10' text-anchor='end'>{int(frac*100)}%</text>"
        )
    # Diagonal: perfect calibration.
    parts.append(
        f"<line x1='{pad_l}' y1='{pad_t + inner_h}' "
        f"x2='{pad_l + inner_w}' y2='{pad_t}' stroke='#484f58' "
        f"stroke-dasharray='4,3'/>"
    )
    # Polyline through populated bins so the reliability shape is easy
    # to follow even when some deciles are sparsely populated.
    pts: List[Tuple[float, float]] = []
    n_total = sum(b.get("n", 0) for b in populated) or 1
    for b in populated:
        mid = (b["lo"] + b["hi"]) / 2.0
        rate = b["wins"] / b["n"]
        x = pad_l + mid * inner_w
        y = pad_t + (1 - rate) * inner_h
        pts.append((x, y))
    poly = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    parts.append(
        f"<polyline points='{poly}' fill='none' "
        f"stroke='#58a6ff' stroke-width='2'/>"
    )
    # Points sized by bin n. Tooltip shows the raw count + win rate.
    for b, (x, y) in zip(populated, pts):
        size = max(3, min(14, (b["n"] / n_total) * 60))
        parts.append(
            f"<g><title>{b['lo']*100:.0f}–{b['hi']*100:.0f}%: "
            f"{b['wins']}/{b['n']} won "
            f"({b['wins']/b['n']*100:.0f}%)</title>"
            f"<circle cx='{x:.1f}' cy='{y:.1f}' r='{size:.1f}' "
            f"fill='#58a6ff' fill-opacity='0.7' stroke='#58a6ff'/></g>"
        )
    # Live-bets overlay: same polyline + circle treatment in orange so
    # the user can see drift at a glance — when the orange line drops
    # below the blue (holdout) line, the model is losing more bets
    # than it expected to in that bucket.
    if live_populated:
        live_pts: List[Tuple[float, float]] = []
        n_live_total = sum(b.get("n", 0) for b in live_populated) or 1
        for b in live_populated:
            mid = (b["lo"] + b["hi"]) / 2.0
            rate = b["wins"] / b["n"]
            x = pad_l + mid * inner_w
            y = pad_t + (1 - rate) * inner_h
            live_pts.append((x, y))
        live_poly = " ".join(f"{x:.1f},{y:.1f}" for x, y in live_pts)
        parts.append(
            f"<polyline points='{live_poly}' fill='none' "
            f"stroke='#e3934d' stroke-width='2' stroke-dasharray='5,3'/>"
        )
        for b, (x, y) in zip(live_populated, live_pts):
            size = max(3, min(14, (b["n"] / n_live_total) * 60))
            parts.append(
                f"<g><title>Live · {b['lo']*100:.0f}–{b['hi']*100:.0f}%: "
                f"{b['wins']}/{b['n']} resolved YES "
                f"({b['wins']/b['n']*100:.0f}%)</title>"
                f"<circle cx='{x:.1f}' cy='{y:.1f}' r='{size:.1f}' "
                f"fill='#e3934d' fill-opacity='0.7' stroke='#e3934d'/></g>"
            )
    # Legend — placed inside the chart so the two series stay visually
    # tied to the lines they label.
    legend_y = pad_t + 6
    parts.append(
        f"<g><line x1='{pad_l + 6}' x2='{pad_l + 26}' y1='{legend_y}' "
        f"y2='{legend_y}' stroke='#58a6ff' stroke-width='2'/>"
        f"<text x='{pad_l + 30}' y='{legend_y + 3}' fill='#8b949e' "
        f"font-size='10'>Holdout</text></g>"
    )
    if live_populated:
        parts.append(
            f"<g><line x1='{pad_l + 90}' x2='{pad_l + 110}' "
            f"y1='{legend_y}' y2='{legend_y}' stroke='#e3934d' "
            f"stroke-width='2' stroke-dasharray='5,3'/>"
            f"<text x='{pad_l + 114}' y='{legend_y + 3}' fill='#8b949e' "
            f"font-size='10'>Live</text></g>"
        )
    # Axis labels.
    parts.append(
        f"<text x='{pad_l + inner_w/2}' y='{height - 6}' fill='#8b949e' "
        f"font-size='11' text-anchor='middle'>Predicted probability</text>"
        f"<text x='15' y='{pad_t + inner_h/2}' fill='#8b949e' "
        f"font-size='11' text-anchor='middle' "
        f"transform='rotate(-90 15 {pad_t + inner_h/2})'>"
        f"Observed win rate</text>"
    )
    parts.append("</svg>")
    return "".join(parts)


def _svg_confusion(cm: dict) -> str:
    """2×2 confusion-matrix grid, cell shaded by share of total, labels
    showing absolute count + row %.
    """
    n = int(cm.get("n") or 0)
    if n == 0:
        return ("<div class='empty'>No closed bets yet — confusion "
                "matrix will populate after the first WIN/LOSS.</div>")
    tp = int(cm.get("tp") or 0)
    fp = int(cm.get("fp") or 0)
    fn = int(cm.get("fn") or 0)
    tn = int(cm.get("tn") or 0)
    cells = [
        ("TP", tp, "Confident bet · won",   "#3fb950", 0, 0),
        ("FP", fp, "Confident bet · lost",  "#f85149", 0, 1),
        ("FN", fn, "Low-conf bet · won",    "#d29922", 1, 0),
        ("TN", tn, "Low-conf bet · lost",   "#484f58", 1, 1),
    ]
    width, height = 360, 260
    pad_l, pad_t = 60, 60
    cell_w = (width - pad_l - 20) / 2
    cell_h = (height - pad_t - 30) / 2
    parts: List[str] = []
    parts.append(
        f"<svg viewBox='0 0 {width} {height}' "
        f"style='width:100%;height:auto;max-width:380px;"
        f"display:block;background:#0d1117;border:1px solid #21262d;"
        f"border-radius:6px;'>"
    )
    # Outer axis labels.
    parts.append(
        f"<text x='{pad_l + cell_w}' y='20' fill='#8b949e' "
        f"font-size='11' text-anchor='middle'>Outcome</text>"
        f"<text x='{pad_l + cell_w/2}' y='40' fill='#8b949e' "
        f"font-size='11' text-anchor='middle'>Won</text>"
        f"<text x='{pad_l + cell_w + cell_w/2}' y='40' fill='#8b949e' "
        f"font-size='11' text-anchor='middle'>Lost</text>"
        f"<text x='15' y='{pad_t + cell_h}' fill='#8b949e' "
        f"font-size='11' text-anchor='middle' "
        f"transform='rotate(-90 15 {pad_t + cell_h})'>Confidence</text>"
        f"<text x='{pad_l - 6}' y='{pad_t + cell_h/2 + 3}' "
        f"fill='#8b949e' font-size='11' text-anchor='end'>≥ 0.5</text>"
        f"<text x='{pad_l - 6}' y='{pad_t + cell_h + cell_h/2 + 3}' "
        f"fill='#8b949e' font-size='11' text-anchor='end'>&lt; 0.5</text>"
    )
    for label, count, tip, base, row, col in cells:
        x = pad_l + col * cell_w
        y = pad_t + row * cell_h
        share = count / n
        parts.append(
            f"<g><title>{label} — {tip}: {count} of {n} "
            f"({share*100:.0f}%)</title>"
            f"<rect x='{x:.1f}' y='{y:.1f}' "
            f"width='{cell_w:.1f}' height='{cell_h:.1f}' "
            f"fill='{base}' fill-opacity='{0.18 + share*0.45:.2f}' "
            f"stroke='#21262d'/>"
            f"<text x='{x + cell_w/2:.1f}' y='{y + cell_h/2 - 2:.1f}' "
            f"fill='#c9d1d9' font-size='22' font-weight='700' "
            f"text-anchor='middle'>{count}</text>"
            f"<text x='{x + cell_w/2:.1f}' y='{y + cell_h/2 + 18:.1f}' "
            f"fill='#8b949e' font-size='11' "
            f"text-anchor='middle'>{label}</text></g>"
        )
    parts.append("</svg>")
    return "".join(parts)


# Rich feature metadata: source label + colour + plain-English
# description + canonical source URL. Drives both the legend / chart
# colour-coding AND the "all features and their sources" table on
# every bot's Models tab, so the user can audit not just *what* a
# feature is but *where it came from* in one place.
#
# Each rule is a substring matcher. Order matters: more specific keys
# (tennis "form_last", "h2h_") sit above more generic ones (NBA
# "diff_") so e.g. ``diff_form_last5`` lands in the tennis bucket
# rather than the NBA bucket. FRED ids and ETF tickers sit above the
# catch-all "derived transform" bucket for the same reason.
FEATURE_RULES: List[dict] = [
    # ── Billboard Hot 100: real weekly chart history from the
    # utdata/rwd-billboard-data public mirror of billboard.com.
    # All 15 features in the Hot 100 Top-10 model are derived
    # entirely from this one source. Split into four logical groups
    # so the chart legend and table can show users which part of
    # the chart panel each feature came from.
    {"patterns": ("artist_total_prior_top10s",
                   "artist_weeks_since_last_top10",
                   "artist_prior_top10_count"),
     "label": "Billboard Hot 100 (artist history)", "color": "#58a6ff",
     "description": "How often this artist has had songs in the Hot 100 top 10 in the past, and how recently. An artist who just had a top-10 hit is more likely to put another song there.",
     "link": "https://www.billboard.com/charts/hot-100/"},
    {"patterns": ("peak_position_so_far", "weeks_on_chart",
                   "debut_rank", "weeks_since_debut",
                   "best_3wk_rank", "rank_change_last_week",
                   "weeks_in_top10_so_far", "weeks_in_top40_so_far"),
     "label": "Billboard Hot 100 (song trajectory)", "color": "#58a6ff",
     "description": "The song's own chart history before this week — how high it has climbed, how long it has been on the chart, how it has moved week-to-week, and how many weeks it has already spent in the top 10 or top 40.",
     "link": "https://www.billboard.com/charts/hot-100/"},
    {"patterns": ("debut_month_sin", "debut_month_cos", "debut_dow"),
     "label": "Billboard Hot 100 (release timing)", "color": "#58a6ff",
     "description": "When the song first appeared on the Hot 100 (month of the year and day of the week). Captures any seasonal pattern in which release windows tend to produce top-10 hits.",
     "link": "https://www.billboard.com/charts/hot-100/"},
    {"patterns": ("competition_count",),
     "label": "Billboard Hot 100 (weekly competition)", "color": "#58a6ff",
     "description": "How crowded the chart's top 40 is with other fresh debuts this week. A busy release week means more competition for a top-10 slot.",
     "link": "https://www.billboard.com/charts/hot-100/"},

    # ── Tennis / Table tennis: Jeff Sackmann dataset + bot-computed Elo
    {"patterns": ("surface_elo", "style_elo"),
     "label": "Elo (bot-computed)", "color": "#bc8cff",
     "description": "How strong the player is on this specific court surface or against this style of opponent. Higher number means a better player.",
     "link": "https://en.wikipedia.org/wiki/Elo_rating_system"},
    {"patterns": ("h2h_",),
     "label": "Head-to-head (Sackmann)", "color": "#58a6ff",
     "description": "How the two players have done against each other in their past matches.",
     "link": "https://github.com/JeffSackmann/tennis_atp"},
    {"patterns": ("form_last",),
     "label": "Form (Sackmann)", "color": "#58a6ff",
     "description": "What fraction of their recent matches the player won — a measure of how hot they are right now.",
     "link": "https://github.com/JeffSackmann/tennis_atp"},
    {"patterns": ("serve_pts", "return_pts", "bp_saved",
                   "deuce_win_pct", "closing_win_pct",
                   "point_win_pct", "game_margin",
                   "comeback_rate"),
     "label": "Match stats (Sackmann)", "color": "#58a6ff",
     "description": "How well the player has been serving, returning, and winning tight points in their recent matches.",
     "link": "https://github.com/JeffSackmann/tennis_atp"},
    {"patterns": ("days_rest", "matches_last_7d"),
     "label": "Schedule (Sackmann)", "color": "#58a6ff",
     "description": "How rested the player is — days since their last match and how many matches they've played this week.",
     "link": "https://github.com/JeffSackmann/tennis_atp"},
    {"patterns": ("hand_matchup", "diff_hand"),
     "label": "Player profile (Sackmann)", "color": "#58a6ff",
     "description": "Whether the matchup is lefty-vs-righty, lefty-vs-lefty, etc — these matchups play out differently.",
     "link": "https://github.com/JeffSackmann/tennis_atp"},
    {"patterns": ("rank_diff",),
     "label": "ATP/WTA rankings", "color": "#58a6ff",
     "description": "Gap between the two players' official world rankings.",
     "link": "https://github.com/JeffSackmann/tennis_atp"},
    {"patterns": ("round_rank", "level_rank", "is_bo7"),
     "label": "Tournament metadata", "color": "#58a6ff",
     "description": "How big the tournament is and how late in the bracket the match sits (later rounds in bigger tournaments play differently).",
     "link": "https://github.com/JeffSackmann/tennis_atp"},
    # ── NBA / generic Elo: pattern is "_elo" or "elo_" (not plain
    # "elo") so it doesn't false-match the substring inside words
    # like "below" / "above" / "develop".
    {"patterns": ("_elo", "elo_"),
     "label": "Elo (bot-computed)", "color": "#bc8cff",
     "description": "How strong this team or player has been recently. Goes up after wins and down after losses, scaled by the margin of victory.",
     "link": "https://en.wikipedia.org/wiki/Elo_rating_system"},
    {"patterns": ("_b2b", "b2b_"),
     "label": "Schedule (bot-computed)", "color": "#bc8cff",
     "description": "Whether the team is playing on the second night of a back-to-back — a fatigue signal that affects performance.",
     "link": "https://github.com/swar/nba_api"},
    # ── NBA: nba_api advanced box-score stats ───────────────────────
    {"patterns": ("_off_rating", "_def_rating", "_net_rating",
                   "_efg_pct", "_oreb_pct", "_tov_pct", "_ft_per_fga",
                   "_fg3m", "_pace", "_win_r", "_team_win",
                   "_team_"),
     "label": "nba_api advanced stats", "color": "#58a6ff",
     "description": "How efficiently the team has been scoring, defending, shooting threes, rebounding, etc — averaged over their recent games.",
     "link": "https://github.com/swar/nba_api"},
    # ── FRED macro series — one entry per series so the link points at
    # the exact series page. Narrow patterns first.
    {"patterns": ("nonfarm_payrolls", "payems"),
     "label": "FRED PAYEMS", "color": "#3fb950",
     "description": "How many people are employed in the US (excluding farm workers). Updated monthly by the government — the headline jobs number.",
     "link": "https://fred.stlouisfed.org/series/PAYEMS"},
    {"patterns": ("treasury_10y", "dgs10"),
     "label": "FRED DGS10", "color": "#3fb950",
     "description": "Interest rate on a 10-year US government bond. Higher means borrowing is more expensive across the economy.",
     "link": "https://fred.stlouisfed.org/series/DGS10"},
    {"patterns": ("treasury_2y", "dgs2"),
     "label": "FRED DGS2", "color": "#3fb950",
     "description": "Interest rate on a 2-year US government bond. Reflects what markets expect short-term rates to do.",
     "link": "https://fred.stlouisfed.org/series/DGS2"},
    {"patterns": ("wti_oil", "dcoilwtico"),
     "label": "FRED DCOILWTICO", "color": "#3fb950",
     "description": "Price of US benchmark crude oil per barrel.",
     "link": "https://fred.stlouisfed.org/series/DCOILWTICO"},
    {"patterns": ("henry_hub", "mhhngsp"),
     "label": "FRED MHHNGSP", "color": "#3fb950",
     "description": "Benchmark price of US natural gas.",
     "link": "https://fred.stlouisfed.org/series/MHHNGSP"},
    {"patterns": ("vix", "vixcls"),
     "label": "FRED VIXCLS", "color": "#3fb950",
     "description": "How much volatility traders expect in the stock market over the next month — known as the 'fear gauge'.",
     "link": "https://fred.stlouisfed.org/series/VIXCLS"},
    {"patterns": ("unemployment_rate", "unrate"),
     "label": "FRED UNRATE", "color": "#3fb950",
     "description": "Percentage of Americans who want a job but don't have one.",
     "link": "https://fred.stlouisfed.org/series/UNRATE"},
    {"patterns": ("continuing_claims", "ccsa"),
     "label": "FRED CCSA", "color": "#3fb950",
     "description": "Number of people still receiving unemployment benefits this week.",
     "link": "https://fred.stlouisfed.org/series/CCSA"},
    {"patterns": ("initial_claims", "icsa"),
     "label": "FRED ICSA", "color": "#3fb950",
     "description": "Number of people who filed for unemployment for the first time this week.",
     "link": "https://fred.stlouisfed.org/series/ICSA"},
    {"patterns": ("ppi", "ppiaco"),
     "label": "FRED PPIACO", "color": "#3fb950",
     "description": "How much prices changed at the wholesale level (what factories charge stores). Leads consumer prices.",
     "link": "https://fred.stlouisfed.org/series/PPIACO"},
    {"patterns": ("headline_cpi", "cpiaucsl"),
     "label": "FRED CPIAUCSL", "color": "#3fb950",
     "description": "Overall consumer price level — what a typical basket of goods and services costs Americans.",
     "link": "https://fred.stlouisfed.org/series/CPIAUCSL"},
    {"patterns": ("core_cpi", "core_mom", "cpilfesl"),
     "label": "FRED CPILFESL", "color": "#3fb950",
     "description": "Consumer prices excluding food and gas (which swing a lot month to month). A cleaner read on underlying inflation.",
     "link": "https://fred.stlouisfed.org/series/CPILFESL"},
    {"patterns": ("used_cars_cpi", "cuur0000seta02"),
     "label": "FRED CUUR0000SETA02", "color": "#3fb950",
     "description": "How much used-car prices have changed.",
     "link": "https://fred.stlouisfed.org/series/CUUR0000SETA02"},
    {"patterns": ("fed_funds_rate", "fedfunds"),
     "label": "FRED FEDFUNDS", "color": "#3fb950",
     "description": "The interest rate the Federal Reserve targets. Sets the floor for borrowing costs across the economy.",
     "link": "https://fred.stlouisfed.org/series/FEDFUNDS"},
    {"patterns": ("industrial_production", "industrial_prod", "indpro"),
     "label": "FRED INDPRO", "color": "#3fb950",
     "description": "How much US factories, mines, and utilities are producing.",
     "link": "https://fred.stlouisfed.org/series/INDPRO"},
    {"patterns": ("umich_inflation", "mich"),
     "label": "FRED MICH", "color": "#3fb950",
     "description": "How much inflation regular Americans expect over the next year (University of Michigan survey).",
     "link": "https://fred.stlouisfed.org/series/MICH"},
    {"patterns": ("consumer_sentiment", "umcsent"),
     "label": "FRED UMCSENT", "color": "#3fb950",
     "description": "How optimistic regular Americans feel about the economy (University of Michigan survey).",
     "link": "https://fred.stlouisfed.org/series/UMCSENT"},
    {"patterns": ("cleveland_expinf", "expinf1yr"),
     "label": "FRED EXPINF1YR", "color": "#3fb950",
     "description": "How much inflation experts expect over the next year (Cleveland Fed model).",
     "link": "https://fred.stlouisfed.org/series/EXPINF1YR"},
    {"patterns": ("m2_yoy", "m2sl"),
     "label": "FRED M2SL", "color": "#3fb950",
     "description": "How much money is circulating in the US economy (cash, checking, savings accounts).",
     "link": "https://fred.stlouisfed.org/series/M2SL"},
    {"patterns": ("retail_gas", "gasregw"),
     "label": "FRED GASREGW", "color": "#3fb950",
     "description": "Average price at the pump for regular gas in the US.",
     "link": "https://fred.stlouisfed.org/series/GASREGW"},
    {"patterns": ("jolts_layoffs", "jtsldl"),
     "label": "FRED JTSLDL (JOLTS)", "color": "#3fb950",
     "description": "How many people were laid off or fired across the US in the latest month.",
     "link": "https://fred.stlouisfed.org/series/JTSLDL"},
    {"patterns": ("jolts_hires", "jtshil"),
     "label": "FRED JTSHIL (JOLTS)", "color": "#3fb950",
     "description": "How many people were hired across the US in the latest month.",
     "link": "https://fred.stlouisfed.org/series/JTSHIL"},
    {"patterns": ("jolts_quits", "jtsqul"),
     "label": "FRED JTSQUL (JOLTS)", "color": "#3fb950",
     "description": "How many people quit their job in the latest month. Higher means workers feel confident they can find another job.",
     "link": "https://fred.stlouisfed.org/series/JTSQUL"},
    {"patterns": ("jolts_openings", "jtsjol"),
     "label": "FRED JTSJOL (JOLTS)", "color": "#3fb950",
     "description": "How many job openings are posted across the US right now.",
     "link": "https://fred.stlouisfed.org/series/JTSJOL"},
    {"patterns": ("unemp_5_14", "uemp5to14"),
     "label": "FRED UEMP5TO14", "color": "#3fb950",
     "description": "How many people have been unemployed for between 5 and 14 weeks.",
     "link": "https://fred.stlouisfed.org/series/UEMP5TO14"},
    {"patterns": ("unemp_27plus", "uemp27ov"),
     "label": "FRED UEMP27OV", "color": "#3fb950",
     "description": "How many people have been unemployed for 27 weeks or more — the long-term unemployed.",
     "link": "https://fred.stlouisfed.org/series/UEMP27OV"},
    # NOTE: pattern requires the trailing underscore so "unemployment"
    # in the Google Trends search-term "filed_for_unemployment" doesn't
    # get misattributed to a duration bucket.
    {"patterns": ("uemp_", "unemp_"),
     "label": "FRED UEMP* (duration buckets)", "color": "#3fb950",
     "description": "How many people are unemployed, grouped by how long they've been out of work.",
     "link": "https://fred.stlouisfed.org/categories/12"},
    {"patterns": ("durable_orders", "dgorder"),
     "label": "FRED DGORDER", "color": "#3fb950",
     "description": "Orders for big-ticket items expected to last three years or more — cars, appliances, machinery. A sign of business investment.",
     "link": "https://fred.stlouisfed.org/series/DGORDER"},
    {"patterns": ("policy_uncertainty", "usepuindxd"),
     "label": "FRED USEPUINDXD", "color": "#3fb950",
     "description": "How uncertain US government policy is right now, measured from news coverage of policy disputes.",
     "link": "https://fred.stlouisfed.org/series/USEPUINDXD"},
    {"patterns": ("trade_weighted_dollar", "dtwexbgs"),
     "label": "FRED DTWEXBGS", "color": "#3fb950",
     "description": "How strong the US dollar is, measured against a basket of other countries' currencies.",
     "link": "https://fred.stlouisfed.org/series/DTWEXBGS"},
    # ── Alt-data / non-FRED sources ─────────────────────────────────
    {"patterns": ("google_trends",),
     "label": "Google Trends", "color": "#d29922",
     "description": "How often Americans are searching Google for terms like 'how to file unemployment' or 'laid off'. A real-time signal of job losses.",
     "link": "https://trends.google.com/trends/"},
    {"patterns": ("layoffs_fyi",),
     "label": "layoffs.fyi", "color": "#d29922",
     "description": "Number of tech-industry layoffs tracked on layoffs.fyi this week. A leading signal for the official jobs data.",
     "link": "https://layoffs.fyi/"},
    {"patterns": ("challenger",),
     "label": "Challenger Gray & Christmas", "color": "#d29922",
     "description": "Total layoffs US companies have announced in the latest month (tracked by the consulting firm Challenger, Gray & Christmas).",
     "link": "https://www.challengergray.com/blog/category/job-cuts-report/"},
    {"patterns": ("warn",),
     "label": "WARN notices", "color": "#d29922",
     "description": "Official mass-layoff notices that US employers are legally required to file before doing big layoffs.",
     "link": "https://www.dol.gov/agencies/eta/layoffs/warn"},
    # ── Survivor elimination: show / game / social signal ──────────
    # Order matters — survivor's reddit_* patterns sit above the
    # generic `reddit` rule so unemployment's reddit_layoffs_* still
    # falls through to r/layoffs.
    {"patterns": ("reddit_mention", "reddit_boot", "reddit_sentiment",
                   "reddit_visibility", "reddit_target"),
     "label": "Reddit r/survivor", "color": "#d29922",
     "description": "Signal scraped from the r/survivor subreddit — how often a contestant is mentioned, how often the community picks them as the next boot, and how positive or negative the discussion around them is.",
     "link": "https://reddit.com/r/survivor"},
    {"patterns": ("season", "episode", "remaining", "is_finale",
                   "pre_merge_phase", "merged", "swap_phase",
                   "tribe_size", "starting_tribe_size",
                   "episode_share"),
     "label": "Show structure", "color": "#bc8cff",
     "description": "Where we are in the season — which episode, how many contestants are left, whether tribes have merged or swapped.",
     "link": "https://survivor.fandom.com/wiki/Main_Page"},
    {"patterns": ("immunity_won", "tribe_immunity", "has_idol",
                   "advantages_held", "idols_played_this_ep",
                   "vote_steals_active"),
     "label": "Game state / advantages", "color": "#3fb950",
     "description": "Whether the contestant has an idol, an advantage, or won immunity this episode — concrete protections that change boot risk.",
     "link": "https://survivor.fandom.com/wiki/Hidden_Immunity_Idol"},
    {"patterns": ("confessional_count", "confessional_share",
                   "visibility_score", "visibility_spike",
                   "negative_edit_score", "narrative_intensity",
                   "strategic_isolation"),
     "label": "Edit / on-show signal", "color": "#58a6ff",
     "description": "Signal extracted from the episode edit — how much screen time the contestant gets, how positive or negative the framing is, and whether the editors are setting them up as the boot.",
     "link": "https://survivor.fandom.com/wiki/Survivor_(franchise)"},
    {"patterns": ("in_main_alliance", "prior_votes_against",
                   "times_targeted", "swing_vote_potential",
                   "voting_minority_score",
                   "same_starting_tribe_remaining"),
     "label": "Alliance / voting state", "color": "#d29922",
     "description": "Where the contestant sits politically — whether they're in the majority alliance, how many votes they've taken in past tribals, and how often they've been targeted.",
     "link": "https://survivor.fandom.com/wiki/Alliance"},
    {"patterns": ("is_returnee", "season_returnee_count",
                   "prior_perf_score", "is_returnee_first_three"),
     "label": "Returnee history", "color": "#8b949e",
     "description": "Whether the contestant has played Survivor before and how they did — returnees behave (and get edited) differently from first-timers.",
     "link": "https://survivor.fandom.com/wiki/Returning_player"},
    {"patterns": ("reddit",),
     "label": "Reddit r/layoffs", "color": "#d29922",
     "description": "How many posts about losing a job were submitted to the r/layoffs subreddit.",
     "link": "https://reddit.com/r/layoffs"},
    # ── Retail-gas / energy: futures, ETFs, EIA Weekly Status ───────
    {"patterns": ("rbob_gasoline_futures", "rbob_"),
     "label": "CME RBOB futures", "color": "#58a6ff",
     "description": "Wholesale gasoline price (what gas stations pay to buy gas). Moves before pump prices.",
     "link": "https://www.cmegroup.com/markets/energy/refined-products/rbob-gasoline-physical.html"},
    {"patterns": ("brent_futures", "brent_spot", "brent_wti_spread"),
     "label": "ICE Brent crude", "color": "#58a6ff",
     "description": "Price of Brent crude oil — the international benchmark used to price most of the world's oil.",
     "link": "https://www.theice.com/products/219/Brent-Crude-Futures"},
    {"patterns": ("wti_futures", "wti_spot", "wti_term_structure"),
     "label": "NYMEX WTI crude", "color": "#58a6ff",
     "description": "Price of West Texas Intermediate — the US benchmark for crude oil.",
     "link": "https://www.cmegroup.com/markets/energy/crude-oil/light-sweet-crude.html"},
    {"patterns": ("natural_gas_futures",),
     "label": "CME Henry Hub NG futures", "color": "#58a6ff",
     "description": "Price of natural gas futures — what wholesale buyers pay for delivery in the coming month.",
     "link": "https://www.cmegroup.com/markets/energy/natural-gas/natural-gas.html"},
    {"patterns": ("heating_oil_futures",),
     "label": "NYMEX heating-oil futures", "color": "#58a6ff",
     "description": "Price of heating oil futures (also used to price diesel).",
     "link": "https://www.cmegroup.com/markets/energy/refined-products/heating-oil.html"},
    {"patterns": ("dxy_dollar_index",),
     "label": "ICE Dollar Index (DXY)", "color": "#58a6ff",
     "description": "How strong the US dollar is against a basket of major currencies (euro, yen, pound, etc).",
     "link": "https://www.theice.com/products/194/US-Dollar-Index-Futures"},
    {"patterns": ("ovx",),
     "label": "CBOE Oil VIX (OVX)", "color": "#58a6ff",
     "description": "How volatile traders expect oil prices to be over the next month — oil's version of the stock-market 'fear gauge'.",
     "link": "https://www.cboe.com/tradable_products/vix/oil_volatility/"},
    {"patterns": ("energy_sector_etf", "xle"),
     "label": "XLE ETF", "color": "#58a6ff",
     "description": "Price of the ETF that holds the big US energy stocks (Exxon, Chevron, etc) — a proxy for how the oil & gas sector is doing.",
     "link": "https://finance.yahoo.com/quote/XLE"},
    {"patterns": ("uga_gasoline_etf", "uga_"),
     "label": "UGA ETF", "color": "#58a6ff",
     "description": "Price of an ETF that directly tracks gasoline prices.",
     "link": "https://finance.yahoo.com/quote/UGA"},
    {"patterns": ("uso_oil_etf",),
     "label": "USO ETF", "color": "#58a6ff",
     "description": "Price of an ETF that directly tracks the price of oil.",
     "link": "https://finance.yahoo.com/quote/USO"},
    {"patterns": ("usl_12mo_oil",),
     "label": "USL ETF", "color": "#58a6ff",
     "description": "Price of an ETF that tracks oil at several future delivery dates (smoother than tracking just one).",
     "link": "https://finance.yahoo.com/quote/USL"},
    {"patterns": ("refinery_utilization", "crude_imports",
                   "crude_stocks", "gasoline_imports",
                   "gasoline_stocks", "gasoline_product_supplied",
                   "distillate_crack"),
     "label": "EIA Weekly Petroleum Status Report", "color": "#3fb950",
     "description": "Weekly government data on how much oil and gasoline is being produced, refined, imported, and held in storage.",
     "link": "https://www.eia.gov/petroleum/supply/weekly/"},
    {"patterns": ("natgas_to_oil", "rbob_minus_brent",
                   "rbob_minus_wti", "rbob_to_wti"),
     "label": "Energy spread (bot-computed)", "color": "#bc8cff",
     "description": "Price difference (or ratio) between two energy products — used to spot when one is cheap relative to the other.",
     "link": "https://www.cmegroup.com/markets/energy/refined-products/rbob-gasoline-physical.html"},
    {"patterns": ("hurricane",),
     "label": "NOAA / NHC hurricane data", "color": "#3fb950",
     "description": "Whether a hurricane is currently active in the Atlantic. Hurricanes shut down Gulf oil rigs and refineries.",
     "link": "https://www.nhc.noaa.gov/"},
    {"patterns": ("summer_driving", "memorial_july4"),
     "label": "Seasonal / calendar", "color": "#d29922",
     "description": "Whether we're in summer driving season or near a major holiday weekend (both push up gas demand).",
     "link": ""},
    {"patterns": ("gas_price_anchor", "gas_pct_above", "gas_pct_below",
                   "gas_range", "gas_zscore", "gas_change_consistency"),
     "label": "Retail-gas target derivative (bot-computed)", "color": "#8b949e",
     "description": "How today's gas price compares to its recent history — its average, range, and how far above or below normal it sits.",
     "link": "https://fred.stlouisfed.org/series/GASREGW"},
    # ── Natural-gas-specific data ───────────────────────────────────
    {"patterns": ("ng_storage_bcf", "storage_change_wow", "storage_lag"),
     "label": "EIA NG Weekly Storage Report", "color": "#3fb950",
     "description": "How much natural gas is being held in underground storage tanks across the US. Low storage means tight supply.",
     "link": "https://www.eia.gov/dnav/ng/ng_stor_wkly_s1_w.htm"},
    {"patterns": ("ng_production", "production_lag", "production_yoy"),
     "label": "EIA NG production", "color": "#3fb950",
     "description": "How much natural gas is being pumped out of US wells.",
     "link": "https://www.eia.gov/naturalgas/production/"},
    {"patterns": ("region_gulf", "region_midwest", "region_northeast",
                   "region_south", "region_west"),
     "label": "NOAA regional weather", "color": "#3fb950",
     "description": "Temperature and weather in a specific region of the US. Hot or cold weather drives heating and cooling demand for natural gas.",
     "link": "https://www.ncei.noaa.gov/access/monitoring/dyk/heating-cooling-degree-information"},
    {"patterns": ("gulf_wind", "gulf_storm", "gulf_max_wind"),
     "label": "NOAA Gulf-of-Mexico weather", "color": "#3fb950",
     "description": "Wind and storm activity in the Gulf of Mexico, where much of the US's oil and gas is produced.",
     "link": "https://www.nhc.noaa.gov/"},
    {"patterns": ("lng_wind", "lng_storm", "lng_temp", "lng_terminal"),
     "label": "NOAA LNG-terminal weather", "color": "#3fb950",
     "description": "Weather near the terminals that ship US natural gas overseas. Bad weather can disrupt exports.",
     "link": "https://www.eia.gov/naturalgas/storage/dashboard/"},
    {"patterns": ("national_avg_temp", "national_cdd", "national_hdd",
                   "national_humidity", "national_wind"),
     "label": "NOAA national weather", "color": "#3fb950",
     "description": "Average US-wide temperature, humidity, and wind, weighted so populated areas count more.",
     "link": "https://www.ncei.noaa.gov/access/monitoring/dyk/heating-cooling-degree-information"},
    {"patterns": ("cdd", "hdd"),
     "label": "NOAA HDD/CDD", "color": "#3fb950",
     "description": "A measure of how cold or hot the weather is — basically the size of the gap between the day's temperature and 65°F. Predicts heating and cooling demand.",
     "link": "https://www.ncei.noaa.gov/access/monitoring/dyk/heating-cooling-degree-information"},
    {"patterns": ("humidity", "temp_lag", "wind_lag", "wind_rolling",
                   "heat_wave_days", "cold_wave_days"),
     "label": "NOAA weather", "color": "#3fb950",
     "description": "Local temperature, humidity, wind speed, or extreme-weather flag from US weather stations.",
     "link": "https://www.ncei.noaa.gov/access/monitoring/dyk/heating-cooling-degree-information"},
    # ── Time-of-period / seasonal flags ─────────────────────────────
    {"patterns": ("week_sin", "week_cos", "week_of_year",
                   "month_sin", "month_cos", "month", "quarter",
                   "day_of_year", "dow_sin", "dow_cos", "day_of_week",
                   "holiday", "is_holiday", "is_weekend", "is_thursday",
                   "is_winter", "is_summer", "is_shoulder", "winter"),
     "label": "Seasonal / calendar", "color": "#d29922",
     "description": "What time of year, day of week, or whether it's a holiday — lets the model pick up seasonal patterns.",
     "link": ""},
    # ── Bot-computed transforms of the target itself ────────────────
    {"patterns": ("target_lag", "target_rolling"),
     "label": "Target derivative (bot-computed)", "color": "#8b949e",
     "description": "What the thing we're predicting has done in the recent past — its previous values and rolling averages.",
     "link": ""},
    {"patterns": ("log_return", "roc_", "trend_dev", "trend_sma"),
     "label": "Derived transform (bot-computed)", "color": "#8b949e",
     "description": "How fast and in what direction the target has been changing — its momentum and trend.",
     "link": ""},
    {"patterns": ("_lag_", "rolling_", "ma13_", "ma52_",
                   "_change_", "_zscore", "_mean_", "_std_",
                   "rolling_mean", "_diff", "_surprise"),
     "label": "Derived transform (bot-computed)", "color": "#8b949e",
     "description": "A past value, recent average, or 'surprise vs expectations' calculated from one of the inputs above.",
     "link": ""},
    # ── NBA: catch-all for derived diff / home / away features. Sits
    # AFTER the tennis-specific rules so ``diff_form_last5`` already
    # got bucketed into "Form (Sackmann)" by then.
    {"patterns": ("diff_", "home_", "away_"),
     "label": "nba_api derived (diff)", "color": "#58a6ff",
     "description": "How the home and away teams compare on a specific stat (home value minus away value).",
     "link": "https://github.com/swar/nba_api"},
]


# Per-base feature descriptions. Maps the un-transformed feature root
# (e.g. ``rbob_gasoline_futures_last``) to a plain-English sentence
# describing what the raw data series is. The transform suffix (lag,
# return, volatility, rolling, etc.) is parsed off and appended at
# render time, so each fully-named feature ends up with a unique
# description even when several share a base.
#
# Order matters: more specific keys first (e.g.
# ``rbob_gasoline_futures_last`` before ``rbob_``). Lookup is by
# longest matching prefix.
_FEATURE_BASES: List[Tuple[str, str]] = [
    # ── Tennis / Table-tennis match-level features ──────────────────
    ("diff_surface_elo_pre",  "Pre-match Elo rating gap between the two players on this specific court surface — accounts for surface specialists."),
    ("diff_style_elo_pre",    "Pre-match Elo rating gap between the two players against opponents of this play style."),
    ("diff_elo_pre",          "Pre-match Elo rating gap between the two players. Higher = player A is more likely to win."),
    ("diff_days_rest",        "Gap in days since each player's last match. Positive = player A has had more rest."),
    ("diff_avg_serve_pts_won_10",  "Difference between the two players in % of points won on serve, averaged over each player's last 10 matches."),
    ("diff_avg_return_pts_won_10", "Difference in % of points won on return, averaged over each player's last 10 matches."),
    ("diff_avg_bp_saved_10",  "Difference in % of break points saved (when serving from behind), averaged over each player's last 10 matches."),
    ("diff_avg_game_margin_10",  "Difference in average game margin (how decisively each player wins) over their last 10 matches."),
    ("diff_avg_point_win_pct_10","Difference in overall point-win % between the two players over their last 10 matches."),
    ("diff_std_game_margin_10",  "Difference in how consistent each player's game margins have been over their last 10 matches. Lower = steadier."),
    ("diff_std_point_win_pct_10","Difference in how consistent each player's point-win rates have been over their last 10 matches."),
    ("diff_closing_win_pct_10",  "Difference in clutch factor: % of close games each player has won over their last 10 matches."),
    ("diff_deuce_win_pct_10",    "Difference in % of deuce points each player has won over their last 10 matches."),
    ("diff_comeback_rate_20",    "Difference in how often each player comes back from behind to win, over their last 20 matches."),
    ("diff_form_last5",  "Difference in win rate over each player's most recent 5 matches."),
    ("diff_form_last10", "Difference in win rate over each player's most recent 10 matches."),
    ("diff_form_last20", "Difference in win rate over each player's most recent 20 matches."),
    ("diff_matches_last_7d", "Difference in how many matches each player has played in the past 7 days. Heavy recent schedule = potential fatigue."),
    ("diff_hand_left", "Whether the two players' handedness pairing includes a leftie (lefties play differently)."),
    ("h2h_a_wins_last5",  "How many of their last 5 meetings player A has won against player B."),
    ("h2h_a_wins_minus_b_wins", "Net head-to-head record across all past meetings (A's wins minus B's wins)."),
    ("hand_matchup_lr", "Whether this is a lefty-vs-righty matchup. Lefties have a small structural advantage on tour."),
    ("is_bo7", "Whether the match is best-of-7 games (vs the standard best-of-5). Longer formats favour the more consistent player."),
    ("rank_diff",   "Gap between the two players' official world rankings (ATP/WTA)."),
    ("level_rank",  "How prestigious the tournament is (e.g. Grand Slam > Masters > 250). Bigger events draw stronger fields and play more conservatively."),
    ("round_rank",  "How deep in the bracket the match sits (1st round = early, final = late). Later rounds tend to be tighter."),

    # ── NBA matchup features ────────────────────────────────────────
    ("home_elo_pre", "Pre-game Elo rating of the home team."),
    ("away_elo_pre", "Pre-game Elo rating of the away team."),
    ("diff_elo",     "Home minus away Elo rating before the game."),
    ("home_b2b",     "Whether the home team is playing on the second night of a back-to-back."),
    ("away_b2b",     "Whether the away team is playing on the second night of a back-to-back."),
    ("diff_off_rating", "Difference between home and away in offensive efficiency (points scored per 100 possessions)."),
    ("diff_def_rating", "Difference between home and away in defensive efficiency (points allowed per 100 possessions)."),
    ("diff_net_rating", "Difference between home and away in net rating (offense minus defense per 100 possessions)."),
    ("diff_pace",       "Difference between home and away in pace of play (possessions per 48 minutes)."),
    ("diff_efg_pct",    "Difference in effective field-goal % (gives extra credit for 3-pointers)."),
    ("diff_oreb_pct",   "Difference in offensive rebounding rate."),
    ("diff_tov_pct",    "Difference in turnover rate."),
    ("diff_ft_per_fga", "Difference in how often the team gets to the free-throw line per shot attempt."),
    ("diff_fg3m",       "Difference in 3-pointers made per game."),
    ("diff_margin",     "Difference in average scoring margin in recent games."),
    ("diff_team_win",   "Difference in recent win rate between the two teams."),

    # ── Natural-gas-specific raw series ─────────────────────────────
    ("ng_storage_bcf",     "Total natural gas held in US underground storage tanks (billions of cubic feet). Low = supply is tight."),
    ("ng_production_bcfd", "Total US natural gas production, in billions of cubic feet per day."),
    ("region_gulf_temp_f",      "Average temperature in the Gulf region (Fahrenheit)."),
    ("region_midwest_temp_f",   "Average temperature in the Midwest region (Fahrenheit)."),
    ("region_northeast_temp_f", "Average temperature in the Northeast region (Fahrenheit)."),
    ("region_south_temp_f",     "Average temperature in the southern region (Fahrenheit)."),
    ("region_west_temp_f",      "Average temperature in the western region (Fahrenheit)."),
    ("region_gulf_cdd",      "Cooling-degree-days in the Gulf region — how warm it's been there."),
    ("region_midwest_cdd",   "Cooling-degree-days in the Midwest — how warm it's been there."),
    ("region_northeast_cdd", "Cooling-degree-days in the Northeast — how warm it's been there."),
    ("region_south_cdd",     "Cooling-degree-days in the southern region — how warm it's been there."),
    ("region_west_cdd",      "Cooling-degree-days in the western region — how warm it's been there."),
    ("region_gulf_hdd",      "Heating-degree-days in the Gulf region — how cold it's been there."),
    ("region_midwest_hdd",   "Heating-degree-days in the Midwest — how cold it's been there."),
    ("region_northeast_hdd", "Heating-degree-days in the Northeast — how cold it's been there."),
    ("region_south_hdd",     "Heating-degree-days in the southern region — how cold it's been there."),
    ("region_west_hdd",      "Heating-degree-days in the western region — how cold it's been there."),
    ("gulf_wind",       "Wind speed over the Gulf of Mexico — high winds disrupt offshore oil and gas operations."),
    ("gulf_storm",      "Whether a named storm is currently active in the Gulf of Mexico."),
    ("gulf_max_wind",   "Peak wind gust recorded over the Gulf of Mexico."),
    ("lng_wind",        "Wind speed at US LNG export terminals — high winds halt tanker loading."),
    ("lng_storm",       "Whether a storm is currently affecting US LNG export terminals."),
    ("lng_temp",        "Temperature at US LNG export terminals."),
    ("lng_terminal_avg",   "Average operating conditions across US LNG export terminals."),
    ("lng_terminal_storm", "Whether any storm has touched a US LNG terminal."),
    ("lng_terminal_wind",  "Wind speed at US LNG export terminals."),
    ("national_avg_temp",     "Average US temperature, weighted so heavily-populated areas count more."),
    ("national_cdd",          "US-wide cooling-degree-days, population-weighted."),
    ("national_hdd",          "US-wide heating-degree-days, population-weighted."),
    ("national_humidity_pct", "Average US humidity, population-weighted."),
    ("national_wind_mph",     "Average US wind speed (mph), population-weighted."),
    ("cdd_sum_3d",     "Total cooling-degree-days over the past 3 days."),
    ("hdd_sum_3d",     "Total heating-degree-days over the past 3 days."),
    ("cold_wave_days", "Number of recent consecutive days flagged as a cold wave (extreme cold)."),
    ("heat_wave_days", "Number of recent consecutive days flagged as a heat wave (extreme heat)."),
    ("gulf_storm_count",   "Number of named storms currently active in the Gulf of Mexico."),
    ("gulf_storm_active",  "1 if any named storm is active in the Gulf, else 0."),
    ("lng_storm_count",    "Number of storms currently affecting US LNG export terminals."),
    ("cdd",      "Cooling-degree-days — a measure of how warm the day was (how far the average temperature sat above 65°F). Predicts AC / power demand."),
    ("hdd",      "Heating-degree-days — a measure of how cold the day was (how far the average temperature sat below 65°F). Predicts heating demand."),
    ("humidity", "Local humidity."),
    ("temp",     "Local temperature."),
    ("wind",     "Local wind speed."),
    ("is_winter",   "Whether it's currently winter (drives natural-gas heating demand)."),
    ("is_summer",   "Whether it's currently summer (drives cooling / electricity demand)."),
    ("is_shoulder", "Whether we're in the shoulder season between winter and summer (mild weather, low energy demand)."),
    ("is_thursday", "Whether today is Thursday — the day EIA releases its weekly natural-gas storage report. Markets often move ahead of the print."),
    ("is_weekend",  "Whether today is Saturday or Sunday."),
    ("is_holiday",  "Whether today is a US federal holiday."),
    ("holiday",     "Whether today is a US federal holiday."),
    ("day_of_week", "Day of the week (0 = Monday … 6 = Sunday)."),
    ("day_of_year", "Day of the year (1–366)."),
    ("week_of_year","Week of the year (1–52)."),
    ("dow_sin",   "Day-of-week encoded as a sine wave so the model can pick up weekly cycles."),
    ("dow_cos",   "Day-of-week encoded as a cosine wave (paired with dow_sin to mark position in the week)."),
    ("week_sin",  "Week-of-year encoded as a sine wave so the model can pick up seasonal patterns."),
    ("week_cos",  "Week-of-year encoded as a cosine wave (paired with week_sin to mark position in the year)."),
    ("month_sin", "Month-of-year encoded as a sine wave for seasonality."),
    ("month_cos", "Month-of-year encoded as a cosine wave for seasonality."),
    ("quarter",   "What quarter of the year it is (1–4)."),
    ("month",     "What month it is (1 = January … 12 = December)."),
    ("winter",         "Whether it's currently winter."),
    ("summer_driving", "Whether we're in the US summer driving season (Memorial Day → Labor Day). Gasoline demand peaks here."),
    ("memorial_july4", "Whether we're near Memorial Day or July 4th — major holiday weekends spike gasoline demand."),
    ("hurricane",      "Whether a hurricane is currently active in the Atlantic basin."),
    ("log_return_abs",   "Absolute size of the most recent daily price moves (regardless of direction)."),
    ("log_return_accel", "Whether the rate-of-change in price is speeding up or slowing down."),
    ("log_return_std",   "How spread-out recent daily returns have been."),
    ("log_return_vol",   "Realized volatility of recent daily price returns."),
    ("log_return",       "Daily log return of the price (a smoother measure than % change)."),
    ("roc_7",  "Rate of change over the past 7 days (% change over a week)."),
    ("roc_30", "Rate of change over the past 30 days (% change over a month)."),
    ("roc_90", "Rate of change over the past 90 days (% change over a quarter)."),
    ("trend_dev_30", "How far the price has drifted away from its 30-day moving average."),
    ("trend_dev_90", "How far the price has drifted away from its 90-day moving average."),
    ("trend_sma7_minus_sma30",  "7-day moving average minus the 30-day moving average. Positive = short-term uptrend."),
    ("trend_sma30_minus_sma90", "30-day moving average minus the 90-day moving average. Positive = medium-term uptrend."),
    ("target_rolling_90_std",  "Standard deviation of natural-gas price over the past 90 days — a measure of recent volatility."),
    ("target",      "Natural-gas closing price."),
    ("storage_change_wow", "Week-over-week change in US natural-gas storage levels."),
    ("storage",     "US natural-gas storage level."),
    ("production_yoy", "Year-over-year change in US natural-gas production."),
    ("production",  "US natural-gas production (billions of cubic feet per day)."),

    # ── Retail-gas / energy raw series ──────────────────────────────
    ("rbob_gasoline_futures_last",  "Wholesale gasoline price — RBOB futures close. What gas stations pay to buy gasoline; moves before pump prices do."),
    ("rbob_gasoline_futures_mean",  "Average wholesale gasoline price (RBOB futures) over the trading week."),
    ("rbob_minus_brent_per_gallon", "Gap (per gallon) between US wholesale gasoline and Brent crude. Widens when refining margins are healthy."),
    ("rbob_minus_wti_per_gallon",   "Gap (per gallon) between US wholesale gasoline and WTI crude — essentially the gasoline refining margin."),
    ("rbob_to_wti_per_gallon_ratio","Ratio of wholesale gasoline to WTI crude (per gallon). Indicates the relative profit margin for refining."),
    ("brent_spot",         "Brent crude oil spot price — the international benchmark for oil."),
    ("brent_futures_last", "Brent crude oil futures closing price."),
    ("brent_wti_spread",   "Price gap between Brent and WTI crude. Widens when it's harder to export US oil."),
    ("wti_spot",            "West Texas Intermediate (WTI) crude oil spot price — the US benchmark."),
    ("wti_futures_last",    "WTI crude oil futures closing price."),
    ("wti_term_structure",  "Shape of the WTI futures curve — whether near-term oil is cheaper or pricier than longer-dated. Reveals supply tightness."),
    ("crude_imports",        "Volume of crude oil imported into the US."),
    ("crude_stocks_ex_spr",  "US crude oil inventories, excluding the Strategic Petroleum Reserve."),
    ("gasoline_imports",     "Volume of gasoline imported into the US."),
    ("gasoline_stocks_total","Total US gasoline inventories held by refiners and distributors."),
    ("gasoline_product_supplied", "How much gasoline US refiners delivered to the market this week — a proxy for consumer demand."),
    ("distillate_crack",     "Profit margin from refining crude oil into distillates (diesel, heating oil)."),
    ("heating_oil_futures_last", "Heating oil futures closing price (also used to price diesel)."),
    ("natural_gas_futures_last", "Natural gas futures closing price."),
    ("dxy_dollar_index_last",   "US Dollar Index (DXY) close — how strong the dollar is against major currencies. Stronger dollar tends to push oil down."),
    ("trade_weighted_dollar",   "Broad measure of the US dollar against many trading-partner currencies."),
    ("ovx_level",               "Oil VIX — how volatile traders expect oil prices to be over the next month."),
    ("energy_sector_etf_last",  "XLE ETF closing price — tracks the big US energy companies (Exxon, Chevron, etc)."),
    ("uga_gasoline_etf_last",   "UGA ETF closing price — directly tracks gasoline futures."),
    ("uso_oil_etf_last",        "USO ETF closing price — directly tracks oil futures."),
    ("usl_12mo_oil_etf_last",   "USL ETF closing price — tracks oil at 12 future delivery dates (smoother than just front-month)."),
    ("refinery_utilization",    "Percentage of US refinery capacity currently in use."),
    ("industrial_production",   "US industrial production index — how busy factories, mines and utilities are."),
    ("natgas_to_oil_ratio",     "Ratio of natural-gas to crude-oil prices. Reveals which energy source is cheap relative to the other."),
    ("gas_pct_above_26w_high",  "How far today's retail gas price sits above its 26-week high (% above)."),
    ("gas_pct_above_13w_high",  "How far today's retail gas price sits above its 13-week high (% above)."),
    ("gas_pct_above_4w_high",   "How far today's retail gas price sits above its 4-week high (% above)."),
    ("gas_pct_below_26w_high",  "How far today's retail gas price sits below its 26-week high (% below)."),
    ("gas_pct_below_13w_high",  "How far today's retail gas price sits below its 13-week high (% below)."),
    ("gas_pct_below_4w_high",   "How far today's retail gas price sits below its 4-week high (% below)."),
    ("gas_pct_below_26w_low",   "How far today's retail gas price sits below its 26-week low (% below)."),
    ("gas_pct_below_13w_low",   "How far today's retail gas price sits below its 13-week low (% below)."),
    ("gas_pct_above_26w_low",   "How far today's retail gas price sits above its 26-week low (% above)."),
    ("gas_pct_above_13w_low",   "How far today's retail gas price sits above its 13-week low (% above)."),
    ("gas_pct_above_4w_low",    "How far today's retail gas price sits above its 4-week low (% above)."),
    ("gas_range_4w",            "Spread (high minus low) of retail gas price over the past 4 weeks."),
    ("gas_range_13w",           "Spread of retail gas price over the past 13 weeks."),
    ("gas_zscore_13w",          "How many standard deviations today's gas price sits from its 13-week average."),
    ("gas_zscore_26w",          "How many standard deviations today's gas price sits from its 26-week average."),
    ("gas_zscore_52w",          "How many standard deviations today's gas price sits from its 52-week average."),
    ("gas_price_anchor",        "Recent retail gas price used as the starting point for projecting future prices."),
    ("gas_change_consistency",  "Whether recent week-over-week gas-price moves have all gone the same direction (consistent trend) or zig-zagged."),

    # ── Unemployment alt-data ───────────────────────────────────────
    ("google_trends_filed_for_unemployment", "How often Americans Google 'filed for unemployment' — a real-time read on new claimants."),
    ("google_trends_unemployment_benefits",  "How often Americans Google 'unemployment benefits' — signals would-be filers."),
    ("google_trends_how_to_file_unemployment", "How often Americans Google 'how to file unemployment' — likely first-time filers."),
    ("google_trends_laid_off",   "How often Americans Google 'laid off'."),
    ("google_trends_lost_my_job","How often Americans Google 'lost my job'."),
    ("google_trends",            "Volume of unemployment-related Google searches across the US."),
    ("layoffs_fyi",  "Number of tech-industry layoffs tracked on layoffs.fyi this week."),
    ("challenger",   "Total job-cut announcements in the latest Challenger, Gray & Christmas monthly report."),
    ("warn",         "Mass-layoff notices US employers have officially filed with their state."),
    ("reddit",       "Number of posts about losing a job submitted to r/layoffs."),

    # ── Unemployment / macro raw series (FRED friendly names) ──────
    ("nonfarm_payrolls",  "Total US employment (excluding farm workers) — the headline jobs number."),
    ("treasury_10y",      "Interest rate on a 10-year US government bond."),
    ("treasury_2y",       "Interest rate on a 2-year US government bond."),
    ("wti_oil",           "WTI crude oil price — the US benchmark."),
    ("henry_hub",         "Henry Hub natural-gas benchmark price."),
    ("vix",               "Stock-market expected volatility over the next month — the 'fear gauge'."),
    ("unemployment_rate", "% of Americans who want a job but don't have one."),
    ("continuing_claims", "Number of people still receiving unemployment benefits."),
    ("initial_claims",    "Number of people who filed for unemployment for the first time this week."),
    ("ppi",               "Producer Price Index — what wholesalers charge stores."),
    ("headline_cpi",      "Headline consumer price level (everything in the basket)."),
    ("core_mom",          "Month-over-month change in core inflation (excluding food and energy)."),
    ("core_cpi",          "Consumer price level excluding food and gas."),
    ("used_cars_cpi",     "Consumer-price-index sub-component for used cars and trucks."),
    ("fed_funds_rate",    "The Federal Reserve's target interest rate."),
    ("industrial_prod",   "US industrial production index."),
    ("umich_inflation",   "1-year inflation expectations from the U-Michigan consumer survey."),
    ("consumer_sentiment","U-Michigan consumer sentiment index — how Americans feel about the economy."),
    ("cleveland_expinf",  "Cleveland Fed model's expectation for inflation 1 year out."),
    ("m2_yoy",            "Year-over-year change in M2 money supply (cash, checking, savings accounts circulating in the US economy)."),
    ("retail_gas",        "US average pump price for regular gasoline."),
    ("jolts_layoffs",     "How many workers were laid off across the US in the latest month."),
    ("jolts_hires",       "How many workers were hired across the US in the latest month."),
    ("jolts_quits",       "How many workers quit their job in the latest month."),
    ("jolts_openings",    "How many job openings are posted across the US."),
    ("unemp_5_14_weeks",  "Number of Americans who have been unemployed for 5–14 weeks."),
    ("unemp_27plus_weeks","Number of Americans who have been unemployed for 27+ weeks (long-term unemployed)."),
    ("durable_orders",    "Orders for big-ticket items expected to last 3+ years."),
    ("policy_uncertainty","Measure of US economic-policy uncertainty from news coverage of policy disputes."),

    # ── Unemployment-bot base names not covered elsewhere ───────────
    ("yield_curve_spread", "Gap between long-term and short-term Treasury yields. An inverted (negative) spread has historically preceded recessions."),
    ("claims",   "Number of new unemployment filings this week (the thing we're predicting)."),
    ("change",   "Week-over-week change in the target (initial-claims count)."),
    ("jolts_layoffs_to_hires", "Ratio of layoffs to hires from the JOLTS report — how loose vs tight the job market is."),

    # ── CPI-bot raw-series base names ───────────────────────────────
    ("cleveland_expinf_1y", "Cleveland Fed model's 1-year-ahead expected inflation."),
    ("umich_inflation_1y",  "1-year-ahead inflation expectations from the U-Michigan consumer survey."),
    ("headline_cpi_mom",    "Month-over-month change in the overall consumer price level (the thing we're predicting on the headline CPI bot)."),
    ("core_mom",            "Month-over-month change in core inflation (excluding food and energy) — the thing we're predicting on the core-CPI bot."),
    ("retail_gas_mom",      "Month-over-month change in US average pump prices for regular gasoline."),
    ("used_cars_cpi_mom",   "Month-over-month change in the used-car CPI sub-component."),
    ("wti_oil_mom",         "Month-over-month change in WTI crude oil price."),
    ("ppi_mom",             "Month-over-month change in the Producer Price Index."),

    # ── NBA non-rolling features ────────────────────────────────────
    ("h2h_wins_before", "Number of times these two teams have met before this season — head-to-head sample size."),
    ("elo_win_prob_home", "Pre-game probability the home team wins, implied by the Elo ratings and home-court bonus."),
    ("elo_diff",        "Pre-game Elo rating gap between the two teams (home minus away), plus the home-court bonus."),
    ("rest_diff",       "Home team's days of rest minus the away team's."),
    ("b2b_diff",        "Home team's back-to-back flag minus the away team's. ±1 = one team is on a back-to-back, the other isn't."),
    ("home_elo_pre",    "Pre-game Elo rating of the home team."),
    ("away_elo_pre",    "Pre-game Elo rating of the away team."),
    ("home_days_rest",  "Days since the home team's last game."),
    ("away_days_rest",  "Days since the away team's last game."),
    ("home_b2b",        "1 if the home team is on the second night of a back-to-back, else 0."),
    ("away_b2b",        "1 if the away team is on the second night of a back-to-back, else 0."),
    ("home_long_rest",  "1 if the home team has had 4+ days of rest, else 0."),
    ("away_long_rest",  "1 if the away team has had 4+ days of rest, else 0."),
    ("home_games_into_season", "How many games the home team has played this season so far."),
    ("away_games_into_season", "How many games the away team has played this season so far."),
]


# Per-stat NBA descriptions. Keys are the lowercased stat names
# (matches whatever sits between HOME_TEAM_ / AWAY_TEAM_ / DIFF_ and
# the rolling-window suffix). Values are the human-readable name for
# the stat, ready to drop into a sentence like
# "Home team's <stat>, averaged over their last 10 games."
_NBA_STAT_DESCRIPTIONS: dict = {
    "off_rating":  "offensive efficiency (points scored per 100 possessions)",
    "def_rating":  "defensive efficiency (points allowed per 100 possessions)",
    "net_rating":  "net rating (offense minus defense per 100 possessions)",
    "pace":        "pace of play (possessions per 48 minutes)",
    "efg_pct":     "effective field-goal percentage (gives extra credit for made 3-pointers)",
    "tov_pct":     "turnover rate (turnovers per possession)",
    "oreb_pct":    "offensive-rebound rate (% of own missed shots rebounded)",
    "ft_per_fga":  "free-throw trips per field-goal attempt (how often the team draws fouls)",
    "margin":      "scoring margin (points scored minus points allowed)",
    "win":         "win rate",
    "fg3m":        "3-pointers made per game",
    "fg3a":        "3-point attempts per game",
}


def _describe_nba_rolling(name: str) -> str:
    """If ``name`` is an NBA rolling-stat feature
    (``HOME_TEAM_<stat>_R<N>``, ``AWAY_TEAM_<stat>_R<N>``, or
    ``DIFF_<stat>_R<N>``), return a feature-specific description.
    Returns "" otherwise so the caller can fall through.
    """
    n = (name or "").lower()
    m = re.match(
        r"^(home_team_|away_team_|diff_)(.+?)_r(\d+)$", n
    )
    if not m:
        return ""
    prefix, stat, window = m.group(1), m.group(2), int(m.group(3))
    stat_desc = _NBA_STAT_DESCRIPTIONS.get(stat)
    if not stat_desc:
        return ""
    if prefix == "home_team_":
        return (f"Home team's {stat_desc}, averaged over "
                f"their last {window} games.")
    if prefix == "away_team_":
        return (f"Away team's {stat_desc}, averaged over "
                f"their last {window} games.")
    # diff_
    return (f"Home minus away differential in {stat_desc}, "
            f"averaged over each team's last {window} games.")


def _period_unit(base: str, plural: bool = True,
                  full_name: str = "",
                  cadence: str = "") -> str:
    """Pick the right time-unit (day / week / month) for transforms
    attached to this base.

    Daily-cadence bots (natural-gas) lag and roll in DAYS. Monthly-
    cadence bot (CPI) lags and rolls in MONTHS. Weekly bots (retail-
    gas, unemployment) lag and roll in WEEKS.

    Resolution order:
      1. ``cadence`` argument (e.g. "months") — most reliable when
         the caller has scanned the whole feature list.
      2. Per-feature monthly markers (``_mom`` / ``_1y_`` / etc.)
      3. Daily vocabulary in the base.
      4. Default to weekly.
    """
    if cadence:
        if cadence.startswith("month"):
            return "months" if plural else "month"
        if cadence.startswith("day"):
            return "days" if plural else "day"
        if cadence.startswith("week"):
            return "weeks" if plural else "week"

    daily_exact = {
        "target", "production", "storage", "log_return",
        "log_return_abs", "log_return_accel", "log_return_std",
        "log_return_vol", "temp", "wind", "humidity",
        "cdd", "hdd",
    }
    daily_prefixes = (
        "target_", "production_", "storage_", "log_return_",
        "roc_", "roc", "trend_dev", "trend_sma",
        "region_", "gulf_", "lng_", "national_", "ng_",
        "cdd_", "hdd_", "humidity_", "temp_", "wind_",
        "cold_wave", "heat_wave",
    )
    # CPI features carry a ``_mom`` / ``_1y_`` tag in their full
    # name — fall back to that when the batch cadence isn't passed.
    if full_name and any(t in full_name for t in (
        "_mom", "_1y_", "headline_cpi", "core_mom",
        "cleveland_expinf", "umich_inflation_1y",
    )):
        return "months" if plural else "month"

    is_daily = base in daily_exact or any(
        base.startswith(p) for p in daily_prefixes
    )
    if is_daily:
        return "days" if plural else "day"
    return "weeks" if plural else "week"


def _detect_bot_cadence(features: List[dict]) -> str:
    """Detect the bot's cadence by scanning the whole feature list.

    Returns one of ``"days"``, ``"weeks"``, ``"months"``. This is
    far more reliable than per-feature detection because shared
    bases (e.g. ``vix_zscore_N``) appear in both monthly (CPI) and
    weekly (unemployment) bots — only the surrounding feature set
    tells you which.
    """
    names = " ".join((f.get("feature") or "").lower() for f in features)
    if any(t in names for t in (
        "_mom", "headline_cpi_mom", "core_mom",
        "cleveland_expinf", "umich_inflation_1y",
    )):
        return "months"
    if any(t in names for t in (
        "target_lag", "target_rolling", "ng_storage",
        "ng_production", "region_", "log_return", "trend_dev",
    )):
        return "days"
    return "weeks"


def _strip_transform_suffix(name: str, cadence: str = "") -> Tuple[str, str]:
    """Peel off the transform suffix and return (base, transform_text).

    Recognised suffixes (in the order they're stripped off the right
    end of the name):

      * ``_lag_N``                       →  "lagged N {unit}"
      * ``_change_lag_N``                →  "week-/month-over-period change, lagged N"
      * ``_surprise_vs_Nw_avg``          →  "vs its N-week average"
      * ``_anomaly_30d``                 →  "anomaly vs the 30-day norm"
      * ``_zscore_N``                    →  "z-scored vs the past N-{unit} window"
      * ``_dev_ma_N`` / ``_dev_N``       →  "deviation from N-{unit} moving average"
      * ``_rolling_N``                   →  "N-{unit} rolling average"
      * ``_mean_N``                      →  "N-{unit} mean"
      * ``_return_Nw`` / ``_volatility_Nw`` /
        ``_change_Nw`` / ``_change_Nm``  →  "{N}-week / month return etc"
      * ``_yoy``                         →  "year-over-year change"
    """
    transforms: List[Any] = []
    PLACEHOLDER_LAG = "<lag>"
    PLACEHOLDER_ROLL = "<roll>"
    PLACEHOLDER_ZSCORE = "<zscore>"
    PLACEHOLDER_DEVMA = "<devma>"
    PLACEHOLDER_MEAN = "<mean>"
    PLACEHOLDER_CHANGE_LAG = "<changelag>"
    original = name

    # _lag_N — the outermost transform. Unit is resolved later once
    # we know the base.
    m = re.search(r"_lag_(\d+)$", name)
    if m:
        transforms.append((PLACEHOLDER_LAG, int(m.group(1))))
        name = name[: m.start()]

    # _change_lag_N is sometimes written ``_change_lag_N`` (no week
    # qualifier) — interpret as "1-period change, lagged N".
    m = re.search(r"_change$", name)
    if m and transforms and isinstance(transforms[-1], tuple) and transforms[-1][0] == PLACEHOLDER_LAG:
        # We just stripped a _lag_N off; if what remains ends in
        # _change, treat the pair as a single change-then-lag transform.
        lag_n = transforms.pop()[1]
        transforms.append((PLACEHOLDER_CHANGE_LAG, lag_n))
        name = name[: m.start()]

    # _surprise_vs_Nw_avg — sits between the base and the lag.
    m = re.search(r"_surprise_vs_(\d+)w_avg$", name)
    if m:
        transforms.append(f"surprise vs the {m.group(1)}-week average")
        name = name[: m.start()]

    # _anomaly_30d
    m = re.search(r"_anomaly_30d$", name)
    if m:
        transforms.append("anomaly vs the 30-day norm")
        name = name[: m.start()]

    # _zscore_N — z-scored vs an N-period look-back window.
    m = re.search(r"_zscore_(\d+)$", name)
    if m:
        transforms.append((PLACEHOLDER_ZSCORE, int(m.group(1))))
        name = name[: m.start()]

    # _dev_ma_N — deviation from N-period moving average. NB: we do
    # NOT match a bare ``_dev_N`` here, because some bases end in
    # ``trend_dev_30`` / ``trend_dev_90`` and the literal ``_30``
    # suffix is part of the base, not a transform.
    m = re.search(r"_dev_ma(\d+)$", name)
    if m:
        transforms.append((PLACEHOLDER_DEVMA, int(m.group(1))))
        name = name[: m.start()]

    # _rolling_N — unit is resolved against the base.
    m = re.search(r"_rolling_(\d+)$", name)
    if m:
        transforms.append((PLACEHOLDER_ROLL, int(m.group(1))))
        name = name[: m.start()]

    # _mean_N — N-period rolling mean (no explicit unit in name).
    m = re.search(r"_mean_(\d+)$", name)
    if m:
        transforms.append((PLACEHOLDER_MEAN, int(m.group(1))))
        name = name[: m.start()]

    # _return_Nw / _volatility_Nw / _change_Nw / _change_Nm
    # (units explicit in name — no period-unit resolution needed).
    for pat, fmt in (
        (r"_return_(\d+)w$",     "{n}-week log return"),
        (r"_volatility_(\d+)w$", "{n}-week realized volatility"),
        (r"_change_(\d+)w$",     "{n}-week % change"),
        (r"_change_(\d+)m$",     "{n}-month % change"),
    ):
        m = re.search(pat, name)
        if m:
            transforms.append(fmt.format(n=m.group(1)))
            name = name[: m.start()]
            break

    # NB: ``_yoy`` is intentionally NOT stripped here. It's usually
    # baked into the BASE name (``m2_yoy``, ``production_yoy``) — the
    # underlying series is already a year-over-year measure — so
    # stripping it would lose the "_yoy" suffix from the base lookup.

    base = name

    # Resolve placeholders now that we know the base + original name.
    def _unit(plural):
        return _period_unit(base, plural=plural, full_name=original,
                             cadence=cadence)
    resolved: List[str] = []
    for t in transforms:
        if isinstance(t, tuple) and t[0] == PLACEHOLDER_LAG:
            n = t[1]
            unit = _unit(plural=(n != 1))
            resolved.append(f"lagged {n} {unit}")
        elif isinstance(t, tuple) and t[0] == PLACEHOLDER_CHANGE_LAG:
            n = t[1]
            unit = _unit(plural=(n != 1))
            resolved.append(f"1-{unit.rstrip('s')} change, lagged {n} {unit}")
        elif isinstance(t, tuple) and t[0] == PLACEHOLDER_ROLL:
            n = t[1]
            unit = _unit(plural=True)
            resolved.append(f"{n}-{unit.rstrip('s')} rolling average")
        elif isinstance(t, tuple) and t[0] == PLACEHOLDER_MEAN:
            n = t[1]
            unit = _unit(plural=True)
            resolved.append(f"{n}-{unit.rstrip('s')} rolling mean")
        elif isinstance(t, tuple) and t[0] == PLACEHOLDER_ZSCORE:
            n = t[1]
            unit = _unit(plural=True)
            resolved.append(
                f"z-scored against its past {n}-{unit.rstrip('s')} "
                f"window (how many standard deviations from the mean)"
            )
        elif isinstance(t, tuple) and t[0] == PLACEHOLDER_DEVMA:
            n = t[1]
            unit = _unit(plural=True)
            resolved.append(
                f"deviation from its {n}-{unit.rstrip('s')} moving average"
            )
        else:
            resolved.append(t)

    # Reverse so the inner-most transform reads first.
    resolved.reverse()
    if not resolved:
        return base, ""
    return base, " — " + ", ".join(resolved) + "."


def _base_description(base: str) -> str:
    """Look up the plain-English description for a feature's base.

    Picks the LONGEST matching prefix so e.g. ``headline_cpi_mom``
    resolves to the ``headline_cpi_mom`` entry (not the shorter
    ``headline_cpi`` one), independent of declaration order. Returns
    an empty string when nothing matches — caller falls back to the
    rule's generic description.
    """
    best_prefix = ""
    best_desc = ""
    for prefix, desc in _FEATURE_BASES:
        if base == prefix or base.startswith(prefix + "_"):
            if len(prefix) > len(best_prefix):
                best_prefix, best_desc = prefix, desc
    return best_desc


def _describe_feature(name: str, cadence: str = "") -> str:
    """Produce a unique, feature-specific description.

    The base-prefix description + the parsed transform suffix combine
    to a sentence like: "Wholesale gasoline price (RBOB futures
    close) ... — 4-week log return, lagged 1 week."

    NBA rolling-stat features (``HOME_TEAM_OFF_RATING_R10`` and
    friends) are handled by a dedicated parser first, since their
    naming convention doesn't fit the base-prefix model.
    """
    # NBA rolling stats — try the dedicated parser before falling
    # through to the generic base + transform pipeline.
    nba = _describe_nba_rolling(name)
    if nba:
        return nba

    lower = (name or "").lower()
    base, transform = _strip_transform_suffix(lower, cadence=cadence)
    desc = _base_description(base)
    if not desc:
        # No specific base match — return empty so feature_metadata
        # falls back to the rule's generic description.
        return ""
    # Trim the trailing period on the base before appending the
    # transform fragment so the punctuation reads cleanly.
    if transform:
        if desc.endswith("."):
            desc = desc[:-1]
        return desc + transform
    return desc


def feature_metadata(name: str, cadence: str = "") -> dict:
    """Map a feature name to its source label, colour, plain-English
    description, and link to where the raw data comes from.

    ``cadence`` is an optional hint ("days" / "weeks" / "months")
    derived from the surrounding feature set — pass it when a
    feature's name alone is ambiguous (e.g. ``vix_zscore_3`` could
    be monthly in CPI's panel or weekly in unemployment's). When
    omitted, the function falls back to per-feature heuristics.
    """
    n = (name or "").lower()
    for rule in FEATURE_RULES:
        for pat in rule["patterns"]:
            if pat in n:
                specific = _describe_feature(name, cadence=cadence)
                return {
                    "label": rule["label"],
                    "color": rule["color"],
                    "description": specific or rule["description"],
                    "link": rule["link"],
                }
    specific = _describe_feature(name, cadence=cadence)
    return {"label": "Other", "color": "#6e7681",
            "description": specific or
                "Source for this feature hasn't been documented yet.",
            "link": ""}


def feature_source(name: str) -> Tuple[str, str]:
    """Backwards-compatible (label, color) lookup. Thin wrapper over
    feature_metadata so attribution lives in exactly one place.
    """
    md = feature_metadata(name)
    return (md["label"], md["color"])


def _svg_feature_importance_vertical(features: List[dict]) -> str:
    """Full-width vertical bar chart of feature importance. Labels
    sit on the X axis below each bar, rotated -45° so they read
    diagonally up-and-to-the-right (anchor at the start, text
    extends up from there). Y axis = mean permutation importance
    from the historical walk-forward training set.

    Bars are colour-coded by data source (see feature_source).
    Selected (kept) features render at full opacity; rejected
    candidates render dimmed.
    """
    if not features:
        return ("<div class='empty'>"
                "Feature importance not yet written for this bot — "
                "the file lands after the next retrain.</div>")
    feats = sorted(features,
                    key=lambda f: f.get("mean_importance") or 0.0,
                    reverse=True)
    n = len(feats)
    # Dynamic right + bottom padding so the rotated labels never
    # clip the SVG edge. A 36-char monospace at 11px is ~250px wide;
    # rotated 45° its bounding box projects ~180px horizontally and
    # ~180px vertically from the anchor. We pad slightly beyond that
    # so the label tail has breathing room from the edge.
    char_w_px = 7
    longest_chars = max((len(f.get("feature") or "") for f in feats),
                         default=0)
    label_text_px = max(80, longest_chars * char_w_px)
    # cos(45°) = sin(45°) ≈ 0.707; round up + buffer.
    label_extent = int(label_text_px * 0.72) + 20
    width = 1180
    # Wider left pad so the rotated y-axis title sits clear of the
    # numeric tick labels.
    pad_l = 76
    pad_r = max(60, label_extent + 18)
    pad_t = 18
    pad_b = max(160, label_extent + 32)
    inner_w = width - pad_l - pad_r
    height = pad_t + pad_b + 260
    inner_h = height - pad_t - pad_b
    max_imp = max(
        (abs(f.get("mean_importance") or 0.0) for f in feats),
        default=1.0,
    ) or 1.0
    bar_pitch = inner_w / max(1, n)
    bar_w = max(8.0, min(28.0, bar_pitch * 0.62))
    parts: List[str] = []
    parts.append(
        f"<svg viewBox='0 0 {width} {height}' "
        f"style='width:100%;height:auto;display:block;"
        f"background:#0d1117;border:1px solid #21262d;border-radius:6px;'>"
    )
    # Y-axis title — rotated 90° CCW, sits in the wider left pad so
    # the user can always tell what the bar heights represent.
    parts.append(
        f"<text x='16' y='{pad_t + inner_h/2}' fill='#8b949e' "
        f"font-size='11' text-anchor='middle' "
        f"transform='rotate(-90 16 {pad_t + inner_h/2})'>"
        f"Mean permutation importance</text>"
    )
    # Y-axis gridlines + labels.
    for k in range(5):
        frac = k / 4.0
        v = frac * max_imp
        y = pad_t + (1 - frac) * inner_h
        parts.append(
            f"<line x1='{pad_l}' x2='{pad_l + inner_w}' "
            f"y1='{y:.1f}' y2='{y:.1f}' stroke='#161b22'/>"
            f"<text x='{pad_l - 6}' y='{y + 3:.1f}' fill='#8b949e' "
            f"font-size='10' text-anchor='end'>{v:.4f}</text>"
        )
    # Baseline X axis.
    parts.append(
        f"<line x1='{pad_l}' x2='{pad_l + inner_w}' "
        f"y1='{pad_t + inner_h}' y2='{pad_t + inner_h}' "
        f"stroke='#21262d'/>"
    )
    for i, f in enumerate(feats):
        imp = abs(f.get("mean_importance") or 0.0)
        h = (imp / max_imp) * inner_h
        x = pad_l + i * bar_pitch + (bar_pitch - bar_w) / 2
        y = pad_t + (inner_h - h)
        sel = bool(f.get("selected"))
        pf = int(f.get("positive_folds") or 0)
        name = f.get("feature") or ""
        src_label, src_color = feature_source(name)
        opacity = 1.0 if sel else 0.42
        text_color = "#c9d1d9" if sel else "#8b949e"
        # Position the tick label BELOW the X axis (where x-axis
        # labels conventionally sit) and rotate +45° clockwise so
        # the text extends down-and-to-the-right from the anchor.
        # text-anchor='start' anchors the LEFT edge of the text at
        # the bar's centre; the rotated label then leans away from
        # its bar into the bottom-padding region without crossing
        # back into the plot area.
        label_x = pad_l + i * bar_pitch + bar_pitch / 2
        label_y = pad_t + inner_h + 10
        parts.append(
            f"<g><title>{html.escape(name)} · {html.escape(src_label)} "
            f"· imp {imp:.4f} · {pf}/5 folds · "
            f"{'kept' if sel else 'rejected'}</title>"
            f"<rect x='{x:.1f}' y='{y:.1f}' width='{bar_w:.1f}' "
            f"height='{h:.1f}' fill='{src_color}' "
            f"fill-opacity='{opacity:.2f}' rx='1'/>"
            f"<text x='{label_x:.1f}' y='{label_y:.1f}' "
            f"fill='{text_color}' font-size='11' "
            f"font-family='ui-monospace,SFMono-Regular,monospace' "
            f"text-anchor='start' "
            f"transform='rotate(45 {label_x:.1f} {label_y:.1f})'>"
            f"{html.escape(name)}</text>"
            f"</g>"
        )
    parts.append("</svg>")
    return "".join(parts)


def _readable_feature_name(name: str) -> str:
    """Render a raw feature identifier in a more reader-friendly form.

    Replaces underscores with spaces and capitalizes the first letter;
    leaves embedded numbers / abbreviations alone. The raw name still
    appears beneath it in monospace for users who need the canonical
    identifier.
    """
    s = (name or "").replace("_", " ").strip()
    return s[:1].upper() + s[1:] if s else ""


def _render_feature_source_table(features: List[dict]) -> str:
    """Aligned feature table with the importance bar on the right.

    Columns: readable feature name, plain-English description, source
    name (clickable link to the canonical source page), and importance
    (a colour-coded horizontal bar with the scalar beside it). Rows
    are sorted by importance descending; only features the
    walk-forward stability filter kept are shown.
    """
    if not features:
        return ""
    kept = [f for f in features if f.get("selected")]
    if not kept:
        return ("<div class='empty' style='margin-top:12px;'>"
                "No features survived the stability filter on the "
                "last retrain — the model is in degenerate state.</div>")
    kept.sort(key=lambda f: f.get("mean_importance") or 0.0, reverse=True)
    cadence = _detect_bot_cadence(features)
    max_imp = max(
        (abs(f.get("mean_importance") or 0.0) for f in kept),
        default=1.0,
    ) or 1.0

    rows: List[str] = []
    for f in kept:
        name = f.get("feature") or ""
        imp = float(f.get("mean_importance") or 0.0)
        md = feature_metadata(name, cadence=cadence)
        link_url = md.get("link") or ""
        bar_pct = (abs(imp) / max_imp) * 100.0
        readable = _readable_feature_name(name)
        if link_url:
            src_cell = (
                f"<a href='{html.escape(link_url)}' target='_blank' "
                f"rel='noopener noreferrer' class='ft-src'>"
                f"<span class='ft-dot' style='background:"
                f"{html.escape(md['color'])};'></span>"
                f"{html.escape(md['label'])} ↗</a>"
            )
        else:
            src_cell = (
                f"<span class='ft-src ft-src-nolink'>"
                f"<span class='ft-dot' style='background:"
                f"{html.escape(md['color'])};'></span>"
                f"{html.escape(md['label'])}</span>"
            )
        rows.append(
            f"<div class='ft-row' "
            f"title='{html.escape(name)} · imp {imp:.4f}'>"
            f"<div class='ft-name-cell' title='{html.escape(name)}'>"
            f"{html.escape(readable)}</div>"
            f"<div class='ft-desc'>{html.escape(md['description'])}"
            f"</div>"
            f"<div class='ft-src-cell'>{src_cell}</div>"
            f"<div class='ft-bar-cell'>"
            f"<div class='ft-bar-track'>"
            f"<div class='ft-bar-fill' style='width:{bar_pct:.1f}%;"
            f"background:{html.escape(md['color'])};'></div>"
            f"</div>"
            f"<div class='ft-bar-imp'>{imp:.4f}</div>"
            f"</div>"
            f"</div>"
        )

    parts = [
        f"<h3 class='subhead'>Feature importance "
        f"<span class='small gray'>({len(kept)} kept by the "
        f"stability filter)</span></h3>",
        "<div class='ft-layout'>",
        "<div class='ft-head'>"
        "<div>Feature</div>"
        "<div>Description</div>"
        "<div>Source</div>"
        "<div>Importance</div>"
        "</div>",
        "<div class='ft-body'>",
        *rows,
        "</div>",
        "</div>",
        _FEATURE_DETAIL_CSS_JS,
    ]
    return "".join(parts)


_FEATURE_DETAIL_CSS_JS = """
<style>
.ft-layout {
  border: 1px solid #21262d;
  border-radius: 6px;
  overflow: hidden;
  margin-top: 4px;
  background: #0d1117;
}
.ft-head {
  display: grid;
  grid-template-columns: 200px 1fr 170px 220px;
  gap: 16px;
  padding: 8px 12px;
  background: #0d1117;
  border-bottom: 1px solid #21262d;
  color: #8b949e;
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  position: sticky;
  top: 0;
  z-index: 1;
}
.ft-body {
  max-height: 600px;
  overflow-y: auto;
}
.ft-row {
  display: grid;
  grid-template-columns: 200px 1fr 170px 220px;
  gap: 16px;
  padding: 10px 12px;
  border-bottom: 1px solid #161b22;
  align-items: center;
}
.ft-row:last-child { border-bottom: none; }
.ft-row:hover { background: #161b22; }
.ft-name-cell {
  font-size: 13px;
  color: #c9d1d9;
  font-weight: 500;
  line-height: 1.3;
  min-width: 0;
}
.ft-desc {
  color: #c9d1d9;
  font-size: 12px;
  line-height: 1.45;
}
.ft-src-cell { font-size: 12px; }
.ft-src {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  color: #58a6ff;
  text-decoration: none;
}
.ft-src:hover { text-decoration: underline; }
.ft-src-nolink { color: #8b949e; }
.ft-dot {
  display: inline-block;
  width: 8px;
  height: 8px;
  border-radius: 2px;
  flex: 0 0 auto;
}
.ft-bar-cell {
  display: grid;
  grid-template-columns: 1fr 56px;
  gap: 8px;
  align-items: center;
  /* visual divider so the importance side reads as its own column */
  border-left: 1px solid #21262d;
  padding-left: 12px;
  margin-left: -8px;
}
.ft-head > div:last-child {
  border-left: 1px solid #21262d;
  padding-left: 12px;
  margin-left: -8px;
}
.ft-bar-track {
  position: relative;
  height: 10px;
  background: #161b22;
  border-radius: 2px;
  overflow: hidden;
}
.ft-bar-fill {
  position: absolute;
  top: 0; left: 0; bottom: 0;
  border-radius: 2px;
}
.ft-bar-imp {
  font-size: 11px;
  color: #8b949e;
  text-align: right;
  font-variant-numeric: tabular-nums;
}
@media (max-width: 900px) {
  .ft-head, .ft-row {
    grid-template-columns: 160px 1fr 140px 160px;
    gap: 10px;
  }
}
@media (max-width: 720px) {
  .ft-head { display: none; }
  .ft-row {
    grid-template-columns: 1fr;
    gap: 6px;
  }
  .ft-bar-cell {
    border-left: none;
    padding-left: 0;
    margin-left: 0;
  }
}
</style>
"""


def _svg_roc_curve(points: List[dict],
                    auc_scalar: float | None = None) -> str:
    """ROC curve SVG. Diagonal reference + the actual TPR/FPR sweep.
    Drops to an empty-state when there are no closed bets yet —
    surfaces the scalar trained AUC in that case so the panel still
    has a numeric anchor.
    """
    width, height = 460, 320
    pad_l, pad_r, pad_t, pad_b = 50, 20, 24, 36
    inner_w = width - pad_l - pad_r
    inner_h = height - pad_t - pad_b
    parts: List[str] = []
    parts.append(
        f"<svg viewBox='0 0 {width} {height}' "
        f"style='width:100%;height:auto;display:block;"
        f"background:#0d1117;border:1px solid #21262d;"
        f"border-radius:6px;'>"
    )
    # Axes + decile gridlines.
    for k in range(0, 11, 2):
        frac = k / 10.0
        x = pad_l + frac * inner_w
        y = pad_t + (1 - frac) * inner_h
        parts.append(
            f"<line x1='{x}' x2='{x}' y1='{pad_t}' "
            f"y2='{pad_t + inner_h}' stroke='#161b22'/>"
            f"<text x='{x}' y='{pad_t + inner_h + 14}' fill='#8b949e' "
            f"font-size='10' text-anchor='middle'>{int(frac*100)}%</text>"
            f"<line x1='{pad_l}' x2='{pad_l + inner_w}' "
            f"y1='{y}' y2='{y}' stroke='#161b22'/>"
            f"<text x='{pad_l - 6}' y='{y + 3}' fill='#8b949e' "
            f"font-size='10' text-anchor='end'>{int(frac*100)}%</text>"
        )
    # Random-baseline diagonal.
    parts.append(
        f"<line x1='{pad_l}' y1='{pad_t + inner_h}' "
        f"x2='{pad_l + inner_w}' y2='{pad_t}' stroke='#484f58' "
        f"stroke-dasharray='4,3'/>"
    )
    # Axis labels.
    parts.append(
        f"<text x='{pad_l + inner_w/2}' y='{height - 6}' fill='#8b949e' "
        f"font-size='11' text-anchor='middle'>"
        f"False positive rate</text>"
        f"<text x='15' y='{pad_t + inner_h/2}' fill='#8b949e' "
        f"font-size='11' text-anchor='middle' "
        f"transform='rotate(-90 15 {pad_t + inner_h/2})'>"
        f"True positive rate</text>"
    )
    if not points:
        # Empty state — surface the trained AUC as a numeric anchor
        # so the section isn't dead until bets close.
        scalar = (f"{auc_scalar*100:.0f}%"
                   if isinstance(auc_scalar, (int, float))
                   else "—")
        parts.append(
            f"<text x='{pad_l + inner_w/2}' "
            f"y='{pad_t + inner_h/2 - 8}' fill='#8b949e' "
            f"font-size='12' text-anchor='middle'>"
            f"No closed bets yet</text>"
            f"<text x='{pad_l + inner_w/2}' "
            f"y='{pad_t + inner_h/2 + 12}' fill='#c9d1d9' "
            f"font-size='13' text-anchor='middle'>"
            f"trained ROC AUC: {scalar}</text>"
        )
        parts.append("</svg>")
        return "".join(parts)
    # Plot the curve.
    pts = []
    for p in points:
        x = pad_l + p["fpr"] * inner_w
        y = pad_t + (1 - p["tpr"]) * inner_h
        pts.append((x, y))
    poly = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    parts.append(
        f"<polyline points='{poly}' fill='none' "
        f"stroke='#58a6ff' stroke-width='2'/>"
    )
    # Trapezoid AUC for the legend label.
    auc = 0.0
    for p1, p2 in zip(points, points[1:]):
        # x = fpr, height = tpr; integrate.
        auc += (p2["fpr"] - p1["fpr"]) * (p1["tpr"] + p2["tpr"]) / 2.0
    parts.append(
        f"<text x='{pad_l + inner_w - 6}' y='{pad_t + inner_h - 6}' "
        f"fill='#58a6ff' font-size='12' font-weight='600' "
        f"text-anchor='end'>AUC {auc*100:.0f}%</text>"
    )
    parts.append("</svg>")
    return "".join(parts)


def _render_hedge_audit(out: List[str], audit: dict) -> None:
    """Hedge effectiveness audit on the Models tab.

    Two-state render: when there are zero hedge-exited bets, show an
    explainer card so the user understands what's measured. Otherwise,
    show a 4-card summary: hedged-bet count, actual P&L, counterfactual
    P&L (if held to settlement), and delta. Green delta means the hedge
    on net made money vs holding; red means it cost money.
    """
    out.append(
        "<h3 class='subhead'>Hedge effectiveness "
        "<span class='small gray'>(profit-lock + stop-loss exits "
        "vs. counterfactual hold-to-settlement)</span></h3>"
    )
    n_hedged = audit.get("n_hedged", 0) or 0
    if n_hedged == 0:
        out.append(
            "<div class='empty'>No hedge events recorded yet — the "
            "hedge daemon's profit-lock and stop-loss thresholds "
            "haven't fired on any closed position. This card will "
            "populate once they do.</div>"
        )
        return
    n_known = audit.get("n_with_settlement", 0) or 0
    actual = audit.get("actual_pnl_cents", 0) or 0
    counter = audit.get("counterfactual_pnl_cents", 0) or 0
    delta = audit.get("delta_cents", 0) or 0
    n_saved = audit.get("n_hedge_saved", 0) or 0
    n_cost = audit.get("n_hedge_cost", 0) or 0

    def _money(c: int) -> str:
        sign = "+" if c > 0 else ("−" if c < 0 else "")
        return f"{sign}${abs(c)/100:.2f}"

    delta_cls = ("green" if delta > 0
                  else ("red" if delta < 0 else "gray"))
    out.append("<div class='row compact'>")
    out.append(
        f"<div class='card'><div class='label' "
        f"title='Total closed positions whose error_type started with "
        f"hedge_ — profit-lock or stop-loss exits.'>"
        f"Hedged exits</div>"
        f"<div class='value'>{n_hedged}</div></div>"
    )
    out.append(
        f"<div class='card'><div class='label' "
        f"title='Actual realized P&amp;L summed across hedged-out "
        f"positions.'>Actual P&amp;L</div>"
        f"<div class='value {('green' if actual>0 else 'red' if actual<0 else 'gray')}'>"
        f"{_money(actual)}</div></div>"
    )
    out.append(
        f"<div class='card'><div class='label' "
        f"title='Counterfactual P&amp;L if every hedged bet had been "
        f"held to natural settlement, summed over the "
        f"{n_known} of {n_hedged} bets whose contracts have settled.'>"
        f"If held to settle</div>"
        f"<div class='value {('green' if counter>0 else 'red' if counter<0 else 'gray')}'>"
        f"{_money(counter)}</div></div>"
    )
    out.append(
        f"<div class='card'><div class='label' "
        f"title='Actual − counterfactual. Positive: the hedge saved "
        f"money vs holding. Negative: the hedge cost money — you "
        f"would have done better letting positions run. {n_saved} "
        f"hedge exits paid off, {n_cost} hedged out too early.'>"
        f"Hedge delta</div>"
        f"<div class='value {delta_cls}'>{_money(delta)}</div></div>"
    )
    out.append("</div>")


def _render_ev_realized_table(out: List[str],
                                buckets: List[dict]) -> None:
    """Predicted-vs-realized EV bucket table for the Models tab.

    Each row is one predicted-EV bucket (e.g. 4–7¢) with the bot's
    average predicted edge, the realized cents-per-contract over the
    closed bets in that bucket, win rate, and total P&L. The Realized
    column is colour-coded against the predicted EV so the user can
    skim for buckets where the edge held vs. where it disappeared.
    """
    # Show the section even when there's nothing to plot — the empty
    # state explains *why* (no closed bets / unsupported schema).
    out.append(
        "<h3 class='subhead'>Predicted vs realized EV "
        "<span class='small gray'>(closed bets, bucketed by entry EV)"
        "</span></h3>"
    )
    has_data = any((b.get("count") or 0) > 0 for b in buckets)
    if not buckets or not has_data:
        out.append(
            "<div class='empty'>No closed bets with a recorded EV "
            "estimate yet — this table populates once the bot has "
            "closed at least one position with "
            "<code>expected_ev_at_entry</code> set.</div>"
        )
        return
    out.append(
        "<table><thead><tr>"
        "<th title='Predicted EV at entry, in cents per contract.'>"
        "EV bucket</th>"
        "<th class='num'>Bets</th>"
        "<th class='num' title='Mean predicted EV across the bucket "
        "(cents per contract).'>Predicted</th>"
        "<th class='num' title='Mean realized P&amp;L per contract "
        "across the bucket (cents). If the edge survived contact "
        "with the market, this tracks the Predicted column.'>"
        "Realized</th>"
        "<th class='num' title='Fraction of bets in this bucket that "
        "closed with positive P&amp;L.'>Win %</th>"
        "<th class='num' title='Sum of realized P&amp;L across every "
        "bet in this bucket, in dollars.'>Total P&amp;L</th>"
        "</tr></thead><tbody>"
    )
    for b in buckets:
        n = b.get("count") or 0
        pred = b.get("predicted_ev_cents")
        realized = b.get("realized_per_contract_cents")
        win = b.get("win_rate")
        total = b.get("total_pnl_cents") or 0
        # Empty buckets render as dashes — keeps the bucket ladder
        # visible even when one tier has no trades yet.
        if n == 0:
            out.append(
                f"<tr><td>{html.escape(b['label'])}</td>"
                f"<td class='num gray'>0</td>"
                f"<td class='num gray'>—</td>"
                f"<td class='num gray'>—</td>"
                f"<td class='num gray'>—</td>"
                f"<td class='num gray'>—</td></tr>"
            )
            continue
        # Colour the Realized cell against the bucket's predicted EV.
        # Green = edge largely survived (realized ≥ predicted × 0.5
        # OR realized within 1¢ of predicted on tiny EV).
        # Yellow = edge eroded but still positive realized.
        # Red = realized ≤ 0 where predicted was positive.
        if realized is None or pred is None:
            real_cls = "gray"
        elif pred <= 0:
            real_cls = "green" if realized > 0 else "red"
        elif realized >= max(pred * 0.5, pred - 1.0):
            real_cls = "green"
        elif realized > 0:
            real_cls = "yellow"
        else:
            real_cls = "red"
        win_cls = "green" if (win or 0) > 0.5 else ("red" if win is not None and win < 0.5 else "gray")
        total_dollars = total / 100.0
        total_cls = "green" if total > 0 else ("red" if total < 0 else "gray")
        # Tag low-sample buckets so the user knows not to overweight them.
        noisy = (" <span class='small gray'>noisy</span>"
                  if n < 10 else "")
        out.append(
            f"<tr><td>{html.escape(b['label'])}{noisy}</td>"
            f"<td class='num'>{n}</td>"
            f"<td class='num'>{pred:+.2f}¢</td>"
            f"<td class='num {real_cls}'>{realized:+.2f}¢</td>"
            f"<td class='num {win_cls}'>{(win or 0)*100:.0f}%</td>"
            f"<td class='num {total_cls}'>"
            f"{'+' if total > 0 else ('−' if total < 0 else '')}"
            f"${abs(total_dollars):.2f}</td></tr>"
        )
    out.append("</tbody></table>")


def _render_model_view_toggle(out: List[str], bot_key: str,
                                 current_view: str,
                                 current_bot: str) -> None:
    """Pre-game / In-game pill toggle for sport bots. Selecting a tab
    swaps the ``?model_view=`` query param via a full-page nav so the
    URL stays shareable.
    """
    bot_qs = (f"&bot={html.escape(current_bot)}" if current_bot else "")
    options = [
        ("pregame", "Pre-game"),
        ("ingame", "In-game"),
    ]
    out.append("<div class='model-view-toggle'>")
    for key, label in options:
        active = " model-view-active" if key == current_view else ""
        href = f"?tab=models{bot_qs}&model_view={key}"
        out.append(
            f"<a class='model-view-pill{active}' "
            f"href='{html.escape(href)}'>{html.escape(label)}</a>"
        )
    out.append("</div>")


def _ingame_proxy_metrics(bot: dict,
                            threshold: float = 0.15) -> Optional[dict]:
    """Proxy classifier metrics for the in-game model's divergence
    feature, the only part of the model that can be replayed from
    the data we currently log.

    For each closed bet:
        prediction = 1 if divergence_at_entry > ``threshold`` else 0
                     (heuristic: 'this bet should win because the
                      market was overreacting at entry')
        actual     = 1 if realized_pnl > 0 else 0

    Returns ``{accuracy, precision, recall, f1, roc_auc, n}`` or
    ``None`` when there aren't enough closed bets to score.

    Live-state features (NBA lead/time, tennis live_prob_*) can't
    be replayed without historical state snapshots and are not
    measured here. The Models view labels this clearly.
    """
    rows = _ingame_backtest_rows(bot)
    if not rows or len(rows) < 5:
        return None
    tp = fp = tn = fn = 0
    pos_scores: List[float] = []
    neg_scores: List[float] = []
    for r in rows:
        pred = 1 if r["divergence"] > threshold else 0
        actual = 1 if r["pnl_cents"] > 0 else 0
        if pred == 1 and actual == 1:
            tp += 1
        elif pred == 1 and actual == 0:
            fp += 1
        elif pred == 0 and actual == 0:
            tn += 1
        else:
            fn += 1
        if actual == 1:
            pos_scores.append(r["divergence"])
        else:
            neg_scores.append(r["divergence"])
    n = len(rows)
    accuracy = (tp + tn) / n
    precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = ((2.0 * precision * recall) / (precision + recall)
           if (precision + recall) > 0 else 0.0)
    # Wilcoxon-Mann-Whitney AUC: P(random positive > random
    # negative) using divergence as the score.
    pairs = 0
    correct = 0.0
    for ps in pos_scores:
        for ns in neg_scores:
            pairs += 1
            if ps > ns:
                correct += 1.0
            elif ps == ns:
                correct += 0.5
    auc = (correct / pairs) if pairs > 0 else None
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": auc,
        "n": n,
        "threshold": threshold,
    }


def _render_ingame_model_view(out: List[str], bot: dict,
                                 active_bets: List[dict],
                                 closed_positions: List[dict] | None = None,
                                 ) -> None:
    """In-game model view for a sport bot's Models tab.

    Shows three sections:
      1. What the in-game model does for this sport (description).
      2. Currently-open positions with their live in-game prediction.
      3. The features the model uses, with sources called out.

    The pre-game model's content (Accuracy / F1 / ROC / etc.) is on
    the Pre-game tab — completely separate.
    """
    bot_key = bot.get("key", "")
    name = bot.get("name", bot_key)
    SPORT_DESC = {
        "nba": (
            "Live game state pulled from ESPN's public scoreboard "
            "every 30 seconds. Win probability derived from the "
            "canonical basketball logistic on lead / √(seconds "
            "remaining), then overlaid with cross-sport market "
            "features (velocity, volatility, divergence from "
            "pre-game prior). High confidence after Q1; low while "
            "the live state is still volatile."
        ),
        "tennis": (
            "Trusts the bot's own watchlist.json live_prob fields "
            "(computed per-point from the live data feed) as the "
            "authoritative live signal. Layers market overreaction "
            "detection on top — when the market has moved sharply "
            "away from a stable live estimate, the model says "
            "HOLD. Confidence ramps from set 1 through set 3+."
        ),
        "table-tennis": (
            "Same architecture as tennis — the bot's per-point "
            "live model is authoritative. Confidence ladder is "
            "tighter since table-tennis sets settle quickly."
        ),
        "darts": (
            "Same architecture as tennis. Confidence bumps higher "
            "after a single set because darts settles faster within "
            "a set than tennis does."
        ),
    }
    desc = SPORT_DESC.get(bot_key, "")

    # ── Headline metric cards (proxy classifier on the divergence
    # feature; the live-state portion isn't backtestable yet). Same
    # 6-card layout as the pre-game view so the user can scan
    # accuracy / F1 / precision / recall / ROC AUC / features side
    # by side.
    proxy = _ingame_proxy_metrics(bot)

    def _pct(v: object, decimals: int = 0) -> str:
        if v is None:
            return "—"
        try:
            return f"{float(v)*100:.{decimals}f}%"
        except (TypeError, ValueError):
            return "—"

    # Count features the heuristic actually uses today (the "Live"
    # rows in the SPORT_FEATURES table below).
    feat_table = {
        # Counts the green "Live" rows in SPORT_FEATURES below.
        "nba": 17,           # score, time, velocity, vol, divergence,
                              # espn_proj, injuries, foul trouble,
                              # box-score gaps, home/away, news,
                              # vegas, recent_form, cdn_pace,
                              # cdn_ft_rate, cdn_fouls, cdn_starter_pm
        "tennis": 9,         # live_prob, score, injury_flag, pre_game,
                              # divergence (now real via snapshotter),
                              # bot_confidence/volatility, bot_rec,
                              # comeback_prob, espn_news
        "table-tennis": 7,   # tennis features minus comeback + espn_news
        "darts": 7,          # same as table-tennis
    }
    live_feature_count = feat_table.get(bot_key, 0)

    out.append("<div class='row compact'>")
    if proxy is None:
        out.append(
            "<div class='card'><div class='label'>Sample size</div>"
            "<div class='value gray'>too small</div></div>"
        )
        for _ in range(5):
            out.append(
                "<div class='card'><div class='label'>—</div>"
                "<div class='value gray'>—</div></div>"
            )
    else:
        n_bets_title = (f"Proxy backtest on {proxy['n']} closed bets "
                         f"(divergence > {proxy['threshold']*100:.0f}% "
                         f"→ predict win).")
        for label, value, kind in [
            ("Accuracy", proxy["accuracy"], "pct"),
            ("F1", proxy["f1"], "pct"),
            ("Precision", proxy["precision"], "pct"),
            ("Recall", proxy["recall"], "pct"),
            ("ROC AUC", proxy["roc_auc"], "pct"),
            ("Features", live_feature_count, "count"),
        ]:
            if kind == "count":
                shown = str(value)
                title_attr = ("Number of features the in-game heuristic "
                               "currently consumes (the Live rows in the "
                               "Features in play table below).")
            else:
                shown = _pct(value, 0) if value is not None else "—"
                title_attr = n_bets_title
            out.append(
                f"<div class='card'><div class='label' "
                f"title='{html.escape(title_attr)}'>"
                f"{html.escape(label)}</div>"
                f"<div class='value'>{html.escape(shown)}</div></div>"
            )
    out.append("</div>")
    out.append(
        "<p class='small gray' style='margin:-6px 0 14px 0;'>"
        "Metrics are computed on a proxy classifier — "
        "<strong>divergence-at-entry &gt; "
        f"{(proxy['threshold']*100 if proxy else 15):.0f}%</strong> as "
        "the predictor of 'this bet will win'. The live-state "
        "features (NBA lead/time, tennis live_prob) need historical "
        "snapshots we don't yet log; those will start showing up "
        "in these numbers once snapshotting lands.</p>"
    )

    out.append(
        "<h3 class='subhead'>How the in-game model works "
        "<span class='small gray'>(heuristic baseline — see "
        "in_game/README.md for the training-pipeline roadmap)"
        "</span></h3>"
    )
    if desc:
        out.append(f"<p class='small' style='color:#c9d1d9;"
                    f"margin:0 0 14px 0;'>{html.escape(desc)}</p>")

    # ── Section 2: current live predictions ─────────────────────────
    live_rows: List[dict] = []
    for ab in (active_bets or []):
        ig = ab.get("_in_game") or {}
        if not ig:
            continue
        live_rows.append({"bet": ab, "ig": ig})
    out.append(
        f"<h3 class='subhead'>Currently-open positions "
        f"<span class='small gray'>({len(live_rows)} with a live "
        f"prediction)</span></h3>"
    )
    if not live_rows:
        out.append(
            "<div class='empty'>No open positions with a confident "
            "live prediction right now. The in-game model only "
            "speaks for sport positions whose match is past the "
            "start gate; pre-tip or pre-match positions show up "
            "here once the game is well underway.</div>"
        )
    else:
        out.append(
            "<table><thead><tr>"
            "<th>Ticker</th><th>Side</th>"
            "<th class='num'>Entry</th>"
            "<th class='num'>Live prob</th>"
            "<th class='num'>Confidence</th>"
            "<th>Action</th><th>Reason</th>"
            "</tr></thead><tbody>"
        )
        ACTION_LABEL = {
            "exit_now": ("EXIT", "red"),
            "let_run": ("RUN", "green"),
            "hold": ("HOLD", "yellow"),
            "neutral": ("—", "gray"),
        }
        for r in live_rows:
            b = r["bet"]
            ig = r["ig"]
            ticker = html.escape(b.get("ticker") or
                                   b.get("match_id") or "—")
            side = html.escape((b.get("side") or "").upper())
            entry_c = b.get("entry_price_cents")
            entry_str = (f"{int(entry_c)}¢" if entry_c is not None
                          else "—")
            lp = ig.get("live_prob_yes") or 0.0
            conf = ig.get("confidence") or 0.0
            action = (ig.get("action") or "neutral").lower()
            label, cls = ACTION_LABEL.get(action, ("—", "gray"))
            reason = html.escape(ig.get("reason") or "")
            out.append(
                f"<tr><td class='mono'>{ticker}</td>"
                f"<td>{side}</td>"
                f"<td class='num'>{entry_str}</td>"
                f"<td class='num'>{lp*100:.0f}%</td>"
                f"<td class='num'>{conf*100:.0f}%</td>"
                f"<td><span class='in-game-pill ig-{cls}'>"
                f"{label}</span></td>"
                f"<td class='small gray'>{reason}</td></tr>"
            )
        out.append("</tbody></table>")

    # ── Section 3: features the model uses ──────────────────────────
    SPORT_FEATURES = {
        "nba": [
            ("Score differential", "Live", "ESPN scoreboard"),
            ("Time remaining", "Live", "ESPN scoreboard (period + clock)"),
            ("Market velocity", "Live", "market_views history (cents/min)"),
            ("Market volatility", "Live", "market_views history (stdev)"),
            ("Divergence vs pre-game", "Live",
             "market_views + position.model_yes_prob_at_entry"),
            ("ESPN win projection", "Live",
             "ESPN /summary?event= predictor block"),
            ("Critical injury counts (per team)", "Live",
             "ESPN /summary injuries block"),
            ("Foul trouble (players w/ ≥4 PF)", "Live",
             "ESPN /summary boxscore.players[*]"),
            ("Live FG% / FT% / 3P% / AST / REB / TO gaps", "Live",
             "ESPN /summary boxscore.teams[*].statistics"),
            ("Home-court advantage (time-decayed)", "Live",
             "ESPN scoreboard homeAway flag"),
            ("Recent injury news mentions (12h)", "Live",
             "ESPN /news?limit=40 keyword scan (in_game/news_signals)"),
            ("Vegas consensus win prob (moneyline)", "Live",
             "ESPN /summary pickcenter.{home,away}TeamOdds.moneyLine"),
            ("Recent form (W-L in last 5)", "Live",
             "ESPN /summary lastFiveGames.events[].gameResult"),
            ("Live pace (possessions / 48 min)", "Live",
             "NBA.com CDN /boxscore.homeTeam.statistics.pace"),
            ("Free-throw rate gap", "Live",
             "NBA.com CDN boxscore: FTA / FGA per team"),
            ("Fouled-out + foul-trouble counts (CDN)", "Live",
             "NBA.com CDN boxscore.players foulsPersonal"),
            ("Starter vs bench +/- splits", "Live",
             "NBA.com CDN boxscore.players starter + plusMinusPoints"),
            ("Shooting % vs xFG", "TODO",
             "shot-chart endpoint + offline xFG training set "
             "(richer than the raw FG% we already have)"),
            ("Lineup combinations on floor", "TODO",
             "NBA.com CDN playbyplay sub events → reconstruct "
             "5-man unit. Doable next session."),
            ("Per-player minutes restriction", "TODO",
             "team injury report + minutes feed"),
        ],
        "tennis": [
            ("Live win probability", "Live",
             "watchlist.json live_prob_a / live_prob_b"),
            ("Current score state", "Live",
             "watchlist.json current_score"),
            ("Injury news flag", "Live",
             "watchlist.json injury_news_flag"),
            ("Pre-game prior", "Live",
             "watchlist.json pre_match_prob_a"),
            ("Market divergence", "Live",
             "watchlist.json yes_ask_cents_a/_b"),
            ("Bot confidence + volatility score", "Live",
             "watchlist.json confidence_score + volatility_score"),
            ("Bot recommended action (signal only)", "Live",
             "watchlist.json recommended_action"),
            ("Comeback probability from set state", "Live",
             "Klaassen-Magnus style heuristic on parsed set wins"),
            ("Recent ESPN injury news mentions (12h)", "Live",
             "ESPN tennis /news feed keyword scan"),
            ("Breakpoint conversion %", "TODO",
             "live point-by-point feed (the bot has it but doesn't "
             "yet expose it on the watchlist row)"),
            ("Serve velocity decline", "TODO",
             "radar gun data; only via broadcast-paired feeds"),
        ],
        "table-tennis": [
            ("Live win probability", "Live",
             "watchlist.json live_prob_a / live_prob_b"),
            ("Current score state", "Live",
             "watchlist.json current_score"),
            ("Reaction-time decay", "TODO",
             "would require frame-level video analysis"),
            ("Spin/style matchup", "TODO",
             "historical match outcomes by player style"),
        ],
        "darts": [
            ("Live win probability", "Live",
             "watchlist.json live_prob_a / live_prob_b"),
            ("Set state", "Live", "watchlist.json current_score"),
            ("Checkout %", "TODO",
             "live leg-by-leg feed (bot has access; not yet exposed "
             "on watchlist row)"),
            ("Leg-streak persistence", "TODO",
             "per-tick watchlist with running leg stats"),
        ],
    }
    feats = SPORT_FEATURES.get(bot_key, [])
    if feats:
        out.append(
            "<h3 class='subhead'>Features in play "
            "<span class='small gray'>(Live = currently used; "
            "TODO = would need an additional feed or trained "
            "model — see in_game/README.md)</span></h3>"
        )
        out.append(
            "<table><thead><tr>"
            "<th>Feature</th><th>Status</th><th>Source</th>"
            "</tr></thead><tbody>"
        )
        for label, status, source in feats:
            status_cls = "green" if status == "Live" else "gray"
            out.append(
                f"<tr><td>{html.escape(label)}</td>"
                f"<td class='{status_cls}'>{html.escape(status)}</td>"
                f"<td class='small gray'>{html.escape(source)}</td>"
                f"</tr>"
            )
        out.append("</tbody></table>")

    # ── Coefficients (what weights the heuristic uses today) ────────
    _render_ingame_coefficients(out, bot_key)

    # ── Historical backtest of the testable features ────────────────
    _render_ingame_backtest(out, bot)

    # ── Recent predictions log + outcome reconciliation ─────────────
    _render_ingame_predictions_log(out, bot, closed_positions or [])


# ──────────────────────────────────────────────────────────────────
# In-game model coefficients — the weights and constants baked into
# the heuristic implementations under ``in_game/``. Listed here so
# the user can see them at a glance and (later) compare them to
# what a trained model would set.
#
# Each entry: (label, current_weight, role description, status)
# Status values:
#   ``tuned``    — hand-set in code; defensible heuristic
#   ``learned``  — would come from a trained model; currently
#                  hand-set as a placeholder
# ──────────────────────────────────────────────────────────────────
_INGAME_COEFFICIENTS: Dict[str, List[tuple]] = {
    "nba": [
        ("Lead / √(sec remaining) coefficient", 0.045,
         "Logistic weight in win_prob = σ(c · lead / √(time)). "
         "Canonical basketball value (Brian Burke).", "tuned"),
        ("Confidence past Q1, low volatility", 0.70,
         "Output confidence when game is past Q1 and live "
         "market vol < 1.5", "tuned"),
        ("Confidence past Q1, high volatility", 0.45,
         "Output confidence when past Q1 but vol ≥ 1.5", "tuned"),
        ("Confidence pre-Q1", 0.25,
         "Output confidence within the first 12 game-minutes — "
         "still too noisy to trust the lead-based model",
         "tuned"),
        ("Market velocity cap", 1.00,
         "Max absolute cents/min movement the velocity feature "
         "reports (clipped to keep linear combos sane)",
         "tuned"),
        ("Divergence reversion damp", 0.50,
         "Fraction of (current_market − pre_game) the in-game "
         "model pulls back toward the pre-game prior",
         "tuned"),
        ("Volatility damp ceiling", 4.00,
         "Volatility above this completely cancels the "
         "reversion pull (market is in too much flux to trust "
         "pre-game prior)",
         "tuned"),
        ("ESPN win-projection weight", 0.25,
         "Weight assigned to ESPN's own predicted win % when "
         "blending it as a third opinion alongside state_prob "
         "and reversion-adjusted prior",
         "tuned"),
        ("Injury / foul-trouble nudge (per player)", 0.015,
         "Each critical injury OR foul-troubled key player on "
         "our team drops live_prob by this; same magnitude on "
         "the opp team raises it",
         "tuned"),
        ("Box-score gap nudge (per pp)", 0.001,
         "Per-pp coefficient on live FG% / FT% / 3P% / AST gap "
         "vs opponent. Turnover gap weighted 3×, rebound gap "
         "weighted 1.5×",
         "tuned"),
        ("Home-court advantage (peak)", 0.025,
         "Maximum nudge applied when our team is at home "
         "(decays from full strength at tip to 30% in the "
         "final minute of regulation)",
         "tuned"),
        ("Injury news nudge (per article, 12h)", 0.010,
         "Each ESPN news article mentioning our team in an "
         "injury context within the last 12 hours nudges "
         "live_prob down by this; same on the opp team raises it",
         "tuned"),
        ("Vegas (pickcenter) blend weight", 0.15,
         "Sportsbook consensus moneyline → implied win prob, "
         "blended in at this weight as a fourth opinion. "
         "Smaller than ESPN's 25% because the pre-game model "
         "already implicitly factors in line movement",
         "tuned"),
        ("Recent-form nudge (per 5 games)", 0.015,
         "Per (our_wins − opp_wins) / 5 differential nudge from "
         "lastFiveGames. Captures momentum carry-over",
         "tuned"),
        ("CDN: FT-rate gap nudge (per pp)", 0.003,
         "NBA.com CDN: per-pp coefficient on (our FT/FGA − opp "
         "FT/FGA). Getting to the line is a stable team edge.",
         "tuned"),
        ("CDN: fouled-out player nudge", 0.030,
         "Each one of our players with 6+ PF drops live_prob by "
         "this. Sourced from NBA.com CDN boxscore.players[*]",
         "tuned"),
        ("CDN: foul-trouble (4-5 PF) nudge", 0.010,
         "Smaller nudge than fouled-out; reflects coach managing "
         "minutes on a player nearing disqualification",
         "tuned"),
        ("CDN: starter +/- amplification", 0.001,
         "Per-point coefficient on average starter plus-minus. "
         "Negative starter +/- with positive bench +/- means "
         "rotation is working against us",
         "tuned"),
        ("CDN: pace × lead amplification", 0.0005,
         "Per (pace − 100 league avg) × sign(lead). High pace "
         "amplifies existing leads (fewer variance possessions "
         "left); low pace gives trailing team more comeback room",
         "tuned"),
        ("EXIT_NOW threshold", 0.30,
         "Recommend exit when our_side_prob falls below this "
         "with confidence ≥ 0.5",
         "tuned"),
        ("LET_RUN threshold", 0.10,
         "Recommend let-run when our_side_prob exceeds entry "
         "by this much",
         "tuned"),
    ],
    "tennis": [
        ("Confidence in set 1", 0.35, "Low — first set is noisy",
         "tuned"),
        ("Confidence in set 2", 0.55, "Medium — pattern emerging",
         "tuned"),
        ("Confidence in set 3+", 0.75, "High — match well-determined",
         "tuned"),
        ("Injury-flag confidence haircut", 0.25,
         "Subtract this from confidence when injury_news_flag is set",
         "tuned"),
        ("Divergence floor (overreaction)", 0.15,
         "Minimum |market − pre_game| before the reversion pull "
         "kicks in",
         "tuned"),
        ("Reversion pull strength", 0.30,
         "Fraction of (market − pre_game) the in-game estimate "
         "pulls back",
         "tuned"),
        ("Divergence threshold for HOLD", 0.20,
         "|market − pre_game| above this with low live volatility "
         "→ market overreaction → HOLD",
         "tuned"),
        ("Bot confidence cap (over our model)", 0.10,
         "In-game confidence can't exceed the tennis bot's own "
         "confidence_score by more than this; caps our claims "
         "when the bot itself is unsure",
         "tuned"),
        ("Bot-volatility haircut threshold", 0.50,
         "Tennis bot's volatility_score above this triggers a "
         "confidence haircut (calibrated to bot's own threshold)",
         "tuned"),
        ("Bot-volatility haircut weight", 0.40,
         "Multiplier on (bot_volatility − threshold) when "
         "applying the haircut; capped at 0.30 max",
         "tuned"),
        ("Comeback-prob exit threshold", 0.10,
         "Recommend EXIT when historical comeback probability "
         "from current set state falls below this (match "
         "essentially decided)",
         "tuned"),
        ("Injury-news confidence haircut (per article)", 0.10,
         "Each recent ESPN tennis article matching a player name + "
         "injury keyword drops in-game confidence by this; "
         "max 3 articles compound",
         "tuned"),
        ("EXIT_NOW threshold", 0.30,
         "Recommend exit when our_bet_prob falls below this with "
         "confidence ≥ 0.5",
         "tuned"),
        ("LET_RUN threshold", 0.10,
         "Recommend let-run when our_bet_prob exceeds entry by "
         "this much",
         "tuned"),
    ],
    "table-tennis": [
        ("Same coefficients as tennis", 0.0,
         "Table tennis uses the tennis heuristic with no overrides",
         "tuned"),
    ],
    "darts": [
        ("Confidence bump after 1 set", 0.10,
         "Add this to the tennis-derived confidence once one "
         "set has completed (darts sets settle faster)",
         "tuned"),
        ("Inherits all tennis coefficients", 0.0,
         "Confidence ladder + reversion logic from tennis.py",
         "tuned"),
    ],
}


def _render_ingame_coefficients(out: List[str], bot_key: str) -> None:
    """Coefficient table + bar chart of relative weights for the
    sport bot's in-game heuristic. Mirrors the pre-game model's
    top-features visual idiom so the two views read consistently.
    """
    coefs = _INGAME_COEFFICIENTS.get(bot_key, [])
    out.append(
        "<h3 class='subhead'>Coefficients "
        "<span class='small gray'>(heuristic — hand-tuned today, "
        "would be learned in a trained version)</span></h3>"
    )
    if not coefs:
        out.append(
            "<div class='empty'>No coefficient table registered "
            "for this sport.</div>"
        )
        return
    # Table view — full detail.
    out.append(
        "<table><thead><tr>"
        "<th>Coefficient</th>"
        "<th class='num'>Value</th>"
        "<th>Role</th>"
        "<th>Status</th>"
        "</tr></thead><tbody>"
    )
    for label, value, role, status in coefs:
        status_cls = "green" if status == "tuned" else "yellow"
        status_label = ("Hand-tuned" if status == "tuned"
                          else "TODO: learn from data")
        v_str = (f"{value:.3f}" if isinstance(value, float)
                  else str(value))
        out.append(
            f"<tr><td>{html.escape(label)}</td>"
            f"<td class='num'>{html.escape(v_str)}</td>"
            f"<td class='small gray'>{html.escape(role)}</td>"
            f"<td class='{status_cls}'>{html.escape(status_label)}</td>"
            f"</tr>"
        )
    out.append("</tbody></table>")


def _render_ingame_backtest(out: List[str], bot: dict) -> None:
    """Historical backtest of the in-game model's most testable
    feature: pre-game/market divergence at entry. Closed positions
    are bucketed by |pre_game_prob − market_implied_prob| at entry,
    and we report the realized P&L per bucket. A real 'market
    overreaction' signal shows up as high-divergence buckets that
    correlate with positive realized P&L.

    Coefficients that depend on live game state (lead/time, set
    score, live volatility) can't be replayed from the data we
    currently store. They're marked as not-yet-backtestable above
    in the coefficient table.
    """
    bot_key = bot.get("key", "")
    out.append(
        "<h3 class='subhead'>Historical backtest "
        "<span class='small gray'>(divergence-at-entry feature, "
        "closed bets only — the live-state features can't be "
        "replayed without snapshot data we don't yet log)"
        "</span></h3>"
    )
    rows = _ingame_backtest_rows(bot)
    if not rows:
        out.append(
            "<div class='empty'>No closed bets with the data we "
            "need (pre-game model probability + entry market "
            "price) on file yet.</div>"
        )
        return
    # Buckets in 5pp width up to 0.30, then a tail bucket for huge
    # divergences. Each row: bets, mean divergence, win rate,
    # mean realized cents per contract, total P&L.
    buckets = [
        ("0–5%",    0.00, 0.05),
        ("5–10%",   0.05, 0.10),
        ("10–15%",  0.10, 0.15),
        ("15–20%",  0.15, 0.20),
        ("20–30%",  0.20, 0.30),
        ("30%+",    0.30, 1.01),
    ]
    out.append(
        "<table><thead><tr>"
        "<th>Divergence @ entry</th>"
        "<th class='num'>Bets</th>"
        "<th class='num'>Mean div.</th>"
        "<th class='num'>Win %</th>"
        "<th class='num'>¢/contract</th>"
        "<th class='num'>Total P&amp;L</th>"
        "</tr></thead><tbody>"
    )
    any_data = False
    for label, lo, hi in buckets:
        bucket = [r for r in rows if lo <= r["divergence"] < hi]
        n = len(bucket)
        if n == 0:
            out.append(
                f"<tr><td>{html.escape(label)}</td>"
                f"<td class='num gray'>0</td>"
                f"<td class='num gray'>—</td>"
                f"<td class='num gray'>—</td>"
                f"<td class='num gray'>—</td>"
                f"<td class='num gray'>—</td></tr>"
            )
            continue
        any_data = True
        mean_div = sum(r["divergence"] for r in bucket) / n * 100.0
        wins = sum(1 for r in bucket if r["pnl_cents"] > 0)
        win_pct = wins / n
        cents_per = sum(r["pnl_per_contract"] for r in bucket) / n
        total = sum(r["pnl_cents"] for r in bucket)
        win_cls = ("green" if win_pct > 0.5
                    else ("red" if win_pct < 0.5 else "gray"))
        cents_cls = ("green" if cents_per > 0
                      else ("red" if cents_per < 0 else "gray"))
        total_cls = ("green" if total > 0
                      else ("red" if total < 0 else "gray"))
        cents_sign = "+" if cents_per > 0 else ("−" if cents_per < 0 else "")
        total_sign = "+" if total > 0 else ("−" if total < 0 else "")
        out.append(
            f"<tr><td>{html.escape(label)}</td>"
            f"<td class='num'>{n}</td>"
            f"<td class='num'>{mean_div:.1f}%</td>"
            f"<td class='num {win_cls}'>{win_pct*100:.0f}%</td>"
            f"<td class='num {cents_cls}'>"
            f"{cents_sign}{abs(cents_per):.2f}¢</td>"
            f"<td class='num {total_cls}'>"
            f"{total_sign}${abs(total)/100:.2f}</td></tr>"
        )
    out.append("</tbody></table>")
    if not any_data:
        out.append(
            "<p class='small gray' style='margin-top:8px;'>"
            "No closed bets fell into any divergence bucket yet — "
            "the backtest populates once the bot closes positions "
            "with both a pre-game model probability and an entry "
            "market price on record.</p>"
        )


def _ingame_backtest_rows(bot: dict) -> List[dict]:
    """Pull closed bets for the bot and compute the divergence /
    realized-P&L pairs the backtest needs. Empty when the bot has
    no closed bets or the schema is missing required columns.
    """
    bot_key = bot.get("key", "")
    rows: List[dict] = []
    if bot.get("dashboard_type") == "tennis":
        # Tennis-shape: closed_positions in sim_state.json. The rollup
        # shape exposes entry_price_cents (market view) + the bot's
        # pre-game model probability under model_yes_prob_at_entry.
        # Divergence = |model − market| on the side bet (tennis rows
        # are always YES-side in the rollup output).
        from . import tennis as _tennis
        for p in _tennis.closed_positions_for_rollup(
                bot.get("sim_state_path"), limit=500):
            entry_cents = p.get("entry_price_cents")
            entry_model = p.get("model_yes_prob_at_entry")
            pnl = p.get("realized_pnl_cents")
            contracts = p.get("contracts") or 1
            if (entry_cents is None or entry_model is None
                    or pnl is None):
                continue
            try:
                market_prob = float(entry_cents) / 100.0
                div = abs(float(entry_model) - market_prob)
                pnl_int = int(pnl)
            except (TypeError, ValueError):
                continue
            rows.append({
                "divergence": div,
                "pnl_cents": pnl_int,
                "pnl_per_contract": pnl_int / max(1, int(contracts)),
            })
        return rows
    # Standard sim.db (NBA).
    db_path = bot.get("db_path") or ""
    if not db_path or not Path(db_path).exists():
        return []
    try:
        with closing(_conn(db_path)) as c:
            cols = {r["name"] for r in
                    c.execute("PRAGMA table_info(positions)").fetchall()}
            if "model_yes_prob_at_entry" not in cols:
                return []
            db_rows = c.execute(
                "SELECT side, entry_price_cents, contracts, "
                "       realized_pnl_cents, model_yes_prob_at_entry "
                "FROM positions WHERE status = 'closed' "
                "  AND realized_pnl_cents IS NOT NULL "
                "  AND entry_price_cents IS NOT NULL "
                "  AND model_yes_prob_at_entry IS NOT NULL"
            ).fetchall()
    except (sqlite3.OperationalError, sqlite3.DatabaseError):
        return []
    for r in db_rows:
        side = (r["side"] or "").upper()
        try:
            entry_c = int(r["entry_price_cents"])
            pnl_int = int(r["realized_pnl_cents"])
            contracts = int(r["contracts"] or 1)
            model_yes = float(r["model_yes_prob_at_entry"])
        except (TypeError, ValueError):
            continue
        market_yes_for_side = (entry_c / 100.0 if side == "YES"
                                 else (100 - entry_c) / 100.0)
        model_yes_for_side = (model_yes if side == "YES"
                                else 1.0 - model_yes)
        div = abs(model_yes_for_side - market_yes_for_side)
        rows.append({
            "divergence": div,
            "pnl_cents": pnl_int,
            "pnl_per_contract": pnl_int / max(1, contracts),
        })
    return rows


def _render_ingame_predictions_log(out: List[str], bot: dict,
                                       closed_positions: List[dict]
                                       ) -> None:
    """Recent predictions panel for the In-game view.

    Reads the tail of ``data/in_game_predictions.jsonl`` filtered to
    this bot, joins each entry against the closed-bet ledger
    (matching on ticker), and surfaces the outcome (WON / LOST /
    OPEN) so the user can see whether the model's calls held up.
    """
    bot_key = bot.get("key", "")
    out.append(
        "<h3 class='subhead'>Recent predictions "
        "<span class='small gray'>(every confident action "
        "transition the in-game model has logged for this bot, "
        "newest first)</span></h3>"
    )
    try:
        from .in_game import logger as _ig_logger
        entries = _ig_logger.read_for_bot(bot_key, limit=40)
    except Exception:  # noqa: BLE001
        entries = []
    if not entries:
        out.append(
            "<div class='empty'>No predictions logged yet. The log "
            "populates once the in-game model issues a confident "
            "EXIT / RUN / HOLD action and that action changes from "
            "the previous one for the same ticker.</div>"
        )
        return
    # Index closed positions by ticker for outcome lookup. We use
    # the most-recent close per ticker so re-traded contracts use
    # the latest realized P&L.
    by_ticker: Dict[str, dict] = {}
    for c in (closed_positions or []):
        t = c.get("ticker") or ""
        if not t:
            continue
        # Newer close wins (later exited_at).
        prev = by_ticker.get(t)
        if prev is None:
            by_ticker[t] = c
            continue
        if (c.get("exited_at") or "") > (prev.get("exited_at") or ""):
            by_ticker[t] = c
    out.append(
        "<table><thead><tr>"
        "<th>When</th><th>Ticker</th><th>Side</th>"
        "<th>Action</th>"
        "<th class='num'>Live prob</th>"
        "<th class='num'>Confidence</th>"
        "<th>Outcome</th><th>Reason</th>"
        "</tr></thead><tbody>"
    )
    ACTION_LABEL = {
        "exit_now": ("EXIT", "ig-red"),
        "let_run": ("RUN", "ig-green"),
        "hold": ("HOLD", "ig-yellow"),
    }
    for e in entries:
        ts = (e.get("ts") or "")[:19].replace("T", " ")
        ticker = e.get("ticker") or ""
        side = (e.get("side") or "—")
        action = (e.get("action") or "").lower()
        a_label, a_cls = ACTION_LABEL.get(action, ("—", "ig-gray"))
        lp = e.get("live_prob_yes") or 0.0
        conf = e.get("confidence") or 0.0
        reason = e.get("reason") or ""
        match = by_ticker.get(ticker)
        if match:
            pnl = int(match.get("realized_pnl_cents") or 0)
            if pnl > 0:
                outcome_label = "WON"
                outcome_cls = "green"
            elif pnl < 0:
                outcome_label = "LOST"
                outcome_cls = "red"
            else:
                outcome_label = "FLAT"
                outcome_cls = "gray"
            outcome_html = (
                f"<span class='{outcome_cls}' "
                f"title='Realized P&amp;L: "
                f"{'+' if pnl > 0 else ('−' if pnl < 0 else '')}"
                f"${abs(pnl)/100:.2f}'>{outcome_label}</span>"
            )
        else:
            outcome_html = "<span class='gray'>OPEN</span>"
        out.append(
            f"<tr><td class='small gray'>{html.escape(ts)}</td>"
            f"<td class='mono'>{html.escape(ticker)}</td>"
            f"<td>{html.escape(str(side))}</td>"
            f"<td><span class='in-game-pill {a_cls}'>"
            f"{a_label}</span></td>"
            f"<td class='num'>{lp*100:.0f}%</td>"
            f"<td class='num'>{conf*100:.0f}%</td>"
            f"<td>{outcome_html}</td>"
            f"<td class='small gray'>{html.escape(reason)}</td></tr>"
        )
    out.append("</tbody></table>")


def _render_models_panel(out: List[str], bot: dict, model: dict | None,
                          display: dict | None,
                          available_bots: List[dict],
                          current_bot: str,
                          model_view: str = "pregame",
                          bot_active_bets: List[dict] | None = None,
                          bot_closed_positions: List[dict] | None = None,
                          ) -> None:
    """Per-bot Models tab content. Standard sim.db bots get the full
    deep-dive (headline metrics card row, full feature list bar chart,
    calibration curve, predicted-vs-realized EV, hedge audit, etc.).
    Tennis dispatches into its own renderer.

    Sport bots (NBA, tennis, table-tennis, darts) get a Pre-game /
    In-game toggle at the top so the user can switch between the
    standard pre-game view and the in-game advisory model's view.
    The two are completely separate — toggling never mixes the
    populations.
    """
    bot_key = (bot or {}).get("key", "")
    is_sport_bot = bot_key in {"nba", "tennis", "table-tennis", "darts"}
    # Every model page uses the same section-header layout so the
    # "Model" title and the body content sit at the same vertical
    # position regardless of bot. Sport bots fill the right-hand
    # slot with the real Pre-game / In-game toggle; non-sport bots
    # render an invisible toggle of identical dimensions so the
    # header row has the same height byte-for-byte. (A bare
    # min-height won't do it — the toggle's actual rendered height
    # depends on the page's font / line-height, which we can't
    # predict precisely from CSS alone.)
    out.append("<div class='section'>")
    out.append("<div class='section-header'><h2>Model</h2>")
    if is_sport_bot and bot:
        _render_model_view_toggle(out, bot_key, model_view, current_bot)
    else:
        out.append(
            "<div class='model-view-toggle' "
            "style='visibility:hidden;' aria-hidden='true'>"
            "<span class='model-view-pill'>Pre-game</span>"
            "<span class='model-view-pill'>In-game</span>"
            "</div>"
        )
    out.append("</div>")
    out.append("<div class='body'>")
    # Bot filter moved above the tab bar (per user request).
    if not bot:
        out.append("<div class='empty'>Bot not found.</div>")
        out.append("</div></div>")
        return
    if is_sport_bot and model_view == "ingame":
        _render_ingame_model_view(out, bot, bot_active_bets or [],
                                     bot_closed_positions or [])
        out.append("</div></div>")
        return
    # Tennis-shape bots (tennis / table-tennis / darts) don't have a
    # sim.db — they keep their model artifacts in metrics.json +
    # coefficients.json. Delegate to the tennis renderer; Phase 2b
    # will replace this with a unified section-by-section layout.
    if bot.get("dashboard_type") == "tennis":
        from . import tennis as _tennis
        metrics = _tennis.load_metrics(bot.get("metrics_path"))
        coefficients = _tennis.load_coefficients(bot.get("coefficients_path"))
        sim_state = _tennis.load_sim_state(bot.get("sim_state_path"))
        out.append(_tennis._render_tennis_models_page(
            metrics, coefficients, sim_state,
            metrics_path=bot.get("metrics_path"),
        ))
        out.append("</div></div>")
        return
    # Billboard also has no sim.db — same JSON-source pattern as
    # tennis. The billboard renderer reproduces the SAME visual
    # sections this function produces for sim.db bots (headline
    # metrics cards → top features → ROC → calibration → empty-state
    # stubs for the closed-bet-driven sections), so the Models tab
    # is visually identical to retail-gas-prices'.
    if bot.get("dashboard_type") == "billboard":
        from . import billboard as _billboard
        _billboard.render_models_panel(out, bot)
        out.append("</div></div>")
        return
    db_path = bot.get("db_path") or ""
    if not db_path or not Path(db_path).exists():
        _render_bot_unavailable(out, bot.get("key", ""))
        out.append("</div></div>")
        return

    # Holdout predictions drive the confidence tier (rendered down
    # next to the ROC + Confusion + Calibration block, where the user
    # naturally looks for held-out context) and the chart data. Load
    # once here so the page only reads the file a single time.
    holdout_path = _find_training_artifact(
        db_path, "holdout_predictions.csv")
    pairs = _read_holdout_predictions(str(holdout_path))
    conf = _holdout_confidence(pairs)

    # Training artifacts are loaded once here so both the headline
    # metrics (above the fold) and the feature chart / table below
    # share the same `feats` + `overview` data.
    fi_path = _find_training_artifact(
        db_path, "feature_importance.csv")
    feats = _read_feature_importance(str(fi_path))
    overview = fetch_model_overview(db_path, str(fi_path), feats)
    n_total = overview["n_considered"]
    n_kept = overview["n_kept"]
    top_imp = overview.get("top_importance")
    top_imp_str = (f"{top_imp:.4f}" if isinstance(top_imp, (int, float))
                    else "—")

    # ── Model coefficient cards — positioned at the top of the panel,
    # using the same .row / .card styling as the Home tab. Surfaces
    # the bottom-line training metrics (Accuracy / F1 / Precision /
    # Recall / ROC AUC / Features) up front before the deep-dive
    # chart and confusion matrix below.
    def _pct(v: object, decimals: int = 0) -> str:
        if v is None:
            return "—"
        try:
            return f"{float(v)*100:.{decimals}f}%"
        except (TypeError, ValueError):
            return "—"

    out.append("<div class='row compact'>")
    if not model:
        out.append("<div class='card'><div class='label'>Model</div>"
                   "<div class='value'>No snapshot yet</div></div>")
    else:
        def _int_str(v: object) -> str:
            if v is None:
                return "—"
            try:
                return f"{int(v):,}"
            except (TypeError, ValueError):
                return "—"

        def _short_date(v: object) -> str:
            if not v:
                return "—"
            s = str(v)
            # captured_at is an ISO8601 string like "2026-05-20T14:22:34..."
            return s[:10] if len(s) >= 10 else s

        metric_cards = [
            ("Accuracy",
             "Held-out classifier accuracy on the trainer's test set.",
             _pct(model.get("classifier_accuracy"), 1)),
            ("F1",
             "Harmonic mean of precision and recall on the test set.",
             _pct(model.get("training_f1"))),
            ("Precision",
             "Fraction of predicted positives that were correct.",
             _pct(model.get("training_precision"))),
            ("Recall",
             "Fraction of actual positives the model caught.",
             _pct(model.get("training_recall"))),
            ("ROC AUC",
             "Area under the ROC curve on the test set. "
             "0.5 = random, 1.0 = perfect.",
             _pct(model.get("training_roc_auc"))),
            ("Features",
             "Number of input features the model uses.",
             (str(int(model.get("feature_count")))
              if model.get("feature_count") is not None else "—")),
            ("Train rows",
             "Number of historical observations the model trained on. "
             "More rows = the model has seen more market regimes.",
             _int_str(model.get("rows_train"))),
            ("Test rows",
             "Held-out test rows the metrics above are computed on.",
             _int_str(model.get("rows_test"))),
            ("Last trained",
             "Date the current model snapshot was captured.",
             _short_date(model.get("captured_at"))),
        ]
        for label, title, value in metric_cards:
            out.append(
                f"<div class='card'><div class='label' "
                f"title='{html.escape(title)}'>"
                f"{html.escape(label)}</div>"
                f"<div class='value'>{html.escape(value)}</div></div>"
            )
    out.append("</div>")

    # ── Top features — bars + readable table in one aligned panel ───
    out.append(_render_feature_source_table(feats))

    # ── ROC curve + confusion matrix — both sourced from the
    # trainer's held-out predictions (data/holdout_predictions.csv).
    # That file is the model's evaluation against historical
    # ground-truth (game outcomes / claims releases / etc.) — what
    # the user sees as "the model's accuracy", separate from any
    # closed-bet noise.
    auc_scalar = (model or {}).get("training_roc_auc")
    # `pairs` was already loaded up top; reuse it here so the page only
    # reads holdout_predictions.csv once.
    roc_points = roc_from_holdout(pairs)
    n_pairs = len(pairs)
    holdout_blurb = (
        "Sourced from the trainer's held-out historical test set — "
        "predictions the model never saw during training, so the "
        "numbers below are its honest evaluation against past reality, "
        "not a re-run of the live closed-bet ledger."
    )
    out.append(
        f"<p class='small gray' style='margin:0 0 6px 0;'>"
        f"{html.escape(holdout_blurb)}</p>"
    )
    # Held-out row count + confidence tier — same data the section
    # headers below quote, surfaced once here as a compact one-liner
    # so the trust signal lives next to the held-out plots it grades.
    _render_confidence_card(out, conf)
    out.append("<h3 class='subhead' style='margin-top:0;'>"
                "ROC curve <span class='small gray'>(historical "
                f"held-out test set, {n_pairs:,} predictions)</span></h3>")
    out.append(_svg_roc_curve(roc_points, auc_scalar=auc_scalar))

    # ── Calibration curve from the held-out predictions ─────────────
    out.append("<h3 class='subhead'>Calibration "
                "<span class='small gray'>(historical held-out test "
                "set, predicted prob vs observed positive rate)"
                "</span></h3>")
    cal_bins = calibration_from_holdout(pairs, n_bins=10)
    live_cal_bins = calibration_from_live_bets(db_path, n_bins=10)
    out.append(_svg_calibration(cal_bins, live_bins=live_cal_bins))

    # ── Per-strike accuracy ─────────────────────────────────────────
    rows = fetch_per_strike_accuracy(db_path)
    out.append("<h3 class='subhead'>Accuracy by strike band "
                "<span class='small gray'>(closed bets only)</span></h3>")
    if not rows:
        out.append("<div class='empty'>No closed bets to break down by "
                    "strike yet.</div>")
    else:
        out.append("<table><thead><tr>"
                    "<th>Strike</th><th>Direction</th>"
                    "<th class='num'>Bets</th><th class='num'>Wins</th>"
                    "<th class='num'>Losses</th><th class='num'>Accuracy</th>"
                    "</tr></thead><tbody>")
        for r in rows:
            sl = r.get("floor_strike")
            sh = r.get("cap_strike")
            direction = r.get("direction") or "—"
            qstr = question_str(direction, sl, sh, display=display)
            acc = r["accuracy"]
            cls = ("green" if acc > 0.6 else
                   "yellow" if acc > 0.5 else
                   "red")
            out.append(
                f"<tr><td>{html.escape(qstr)}</td>"
                f"<td>{html.escape(direction)}</td>"
                f"<td class='num'>{r['n']}</td>"
                f"<td class='num green'>{r['wins']}</td>"
                f"<td class='num red'>{r['losses']}</td>"
                f"<td class='num {cls}'>{acc*100:.0f}%</td></tr>"
            )
        out.append("</tbody></table>")

    # ── Predicted vs realized EV ────────────────────────────────────
    # The "did the model's edge survive contact with the market?" check.
    # For each predicted-EV bucket: did realized ¢/contract roughly
    # track the predicted EV? If yes, the edge is real. If realized
    # systematically lags predicted, fees + slippage are eating the
    # edge. If realized is negative where predicted was positive,
    # the model is mis-calibrated (anti-edge).
    _render_ev_realized_table(out, fetch_ev_realized_buckets(db_path))

    # ── Hedge effectiveness audit ───────────────────────────────────
    # "Did the hedge_monitor's profit-lock / stop-loss exits actually
    # net make money vs. just holding to settlement?" Empty when no
    # hedge events have fired yet.
    _render_hedge_audit(out, fetch_hedge_audit(db_path))

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
                                 display: dict | None = None,
                                 contract_is_closed: bool = False) -> None:
    """Renders the 'Current prediction' card row.

    Lives at the top of the Watchlist section now (per user request) — it
    fits there better than under the Model section because it's the
    immediate context for reading the watchlist rows below it.

    Number formatting follows the bot's display config so unemployment
    shows "189K" / "+1K" instead of "$189000.00" / "+1000.00".

    When `contract_is_closed` is True (the current event has already
    settled and there's no live market to forecast for), every value
    cell is dashed out — a stale model snapshot from the previous
    contract isn't a forecast for the next one.
    """
    if not model:
        return
    display = display or {}
    # No subsec wrapper — match the Summary's plain `<div class='row'>`
    # structure so the bottom-of-cards → top-of-h3 spacing collapses
    # naturally and the Watchlist "Active bet" h3 sits at the same
    # offset Summary's "Active bets" h3 does.
    prob_up = float(model.get("prob_up") or 0)
    change = float(model.get("median_change") or 0)
    q05 = model.get("quantile_05")
    q95 = model.get("quantile_95")
    if contract_is_closed:
        cur_str = pred_str = chg_str = prob_str = q05_str = q95_str = "—"
    else:
        cur_str = html.escape(
            fmt_underlying(model.get("current_gas_price"), display))
        pred_str = html.escape(
            fmt_underlying(model.get("median_price"), display))
        chg_str = html.escape(_fmt_signed_underlying(change, display))
        prob_str = f"{prob_up:.0%}"
        q05_str = (html.escape(fmt_underlying(q05, display))
                   if q05 is not None else "—")
        q95_str = (html.escape(fmt_underlying(q95, display))
                   if q95 is not None else "—")
    # Cadence-aware labels driven by the bot's display config:
    #   • underlying_label gives the "Current X" card a bot-specific name
    #     ("Retail gas price" vs "Initial jobless claims" vs "Last
    #     realized Core CPI MoM").
    #   • prediction_period_label fills "Predicted ___" with the right
    #     cadence ("next week" / "next month") so monthly bots don't read
    #     as if they were weekly. Defaults preserve existing weekly bots.
    cur_label = (display.get("underlying_label") or "Current price") if display else "Current price"
    period_label = (display.get("prediction_period_label") or "next week") if display else "next week"
    out.append("<div class='row compact'>")
    out.append(f"<div class='card'><div class='label'>{html.escape(cur_label)}</div>"
               f"<div class='value'>{cur_str}</div></div>")
    out.append(f"<div class='card'><div class='label'>Predicted {html.escape(period_label)}</div>"
               f"<div class='value'>{pred_str}</div></div>")
    out.append(f"<div class='card'><div class='label'>Median change</div>"
               f"<div class='value'>{chg_str}</div></div>")
    out.append(f"<div class='card'><div class='label'>P(price goes up)</div>"
               f"<div class='value'>{prob_str}</div></div>")
    out.append(f"<div class='card'><div class='label'>Lower 5%</div>"
               f"<div class='value'>{q05_str}</div></div>")
    out.append(f"<div class='card'><div class='label'>Upper 95%</div>"
               f"<div class='value'>{q95_str}</div></div>")
    out.append("</div>")  # /row


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
                            prob_history: List[dict] | None = None,
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
    # Chart data source — prefer the probability series (one point per
    # bot poll, y-axis pinned 0..100¢) so the line reflects how the
    # tracked ticker's implied probability moved over time. Falls back
    # to the Kalshi implied-underlying when there's no probability
    # history yet (e.g. a freshly-registered bot with empty
    # market_views).
    prob_history = prob_history or []
    use_prob = bool(prob_history)
    if use_prob:
        chart_history = prob_history
        chart_display = {
            "divisor": 1.0,
            "underlying_decimals": 0,
            "underlying_unit": "%",
            "unit_position": "suffix",
        }
    else:
        chart_history = kalshi_history or []
        chart_display = display

    current: float | None = None
    earliest_value: float | None = None
    if chart_history:
        for r in reversed(chart_history):
            v = r.get("value")
            if v is None:
                continue
            try:
                current = float(v)
                break
            except (TypeError, ValueError):
                continue
        for r in chart_history:
            v = r.get("value")
            if v is None:
                continue
            try:
                earliest_value = float(v)
                break
            except (TypeError, ValueError):
                continue
    if not use_prob:
        # Fallbacks only meaningful for the implied-underlying view —
        # the probability series has no equivalent of model.current_gas_price.
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
    # No live contract — dash out the forecast + change indicator. Two
    # cases collapse to the same display: (a) the current event has
    # already settled (close_ts is in the past), or (b) Kalshi returned
    # no active markets in this series at all (between events, or the
    # fetch errored). In either case the "current forecast" isn't
    # forecasting anything — Kalshi's last printed price is just the
    # settlement value of an expired market, and a fresh series might
    # not have started yet.
    now_ts = datetime.now(timezone.utc).timestamp()
    contract_is_closed = (
        contract_close_ts is None
        or contract_close_ts <= now_ts
    )
    if contract_is_closed:
        current_str = "—"
        change_body = "—"
        change_cls = ""
        value_change = None
    else:
        current_str = fmt_underlying(current, chart_display)
        # Format the raw delta in the bot's native units, then strip the
        # leading sign (the arrow already conveys direction).
        if value_change is None:
            change_body = "—"
            change_cls = ""
        else:
            signed = _fmt_signed_underlying(value_change, chart_display)
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

    # Header layout (per user request): top-left is the volume of the
    # contract the chart line represents (the ATM market) — the
    # forecast value + arrow/change indicator that used to live here
    # were removed because they duplicate the information the chart's
    # right-edge already conveys visually. Top-right keeps the
    # time-to-close on the soonest market.
    contract_volume: float | None = None
    if atm_market:
        v = (atm_market.get("volume_fp")
             if atm_market.get("volume_fp") is not None
             else atm_market.get("volume"))
        if v is not None:
            try:
                contract_volume = float(v)
            except (TypeError, ValueError):
                contract_volume = None
    if contract_volume is None and total_volume:
        # No specific chart contract — show the visible watchlist's
        # aggregate Kalshi volume instead. Keeps the panel populated
        # for bots where the ATM lookup didn't return a market (e.g.
        # JSON-source bots with no per-ticker volume field).
        contract_volume = float(total_volume)
    if contract_volume is None:
        volume_str = "—"
    elif contract_volume >= 1e6:
        volume_str = f"{contract_volume/1e6:.2f}M"
    elif contract_volume >= 1e3:
        volume_str = f"{contract_volume/1e3:.1f}K"
    else:
        volume_str = f"{int(contract_volume):,}"

    out.append("<div class='wl-hero'>")
    out.append("<div class='wl-hero-top'>")
    out.append("<div class='wl-hero-stats'>")
    # Static volume display — no hover-swap behaviour (the chart
    # hover JS only swaps elements when it finds them; the legacy
    # .wl-hero-price / .wl-hero-change selectors are gone so the JS
    # cleanly no-ops on the swap step).
    out.append(
        f"<div class='wl-hero-volume'>"
        f"<span class='wl-hero-volume-text'>{html.escape(volume_str)}</span>"
        f"<span class='wl-hero-volume-label'>volume</span>"
        f"</div>"
    )
    out.append("</div>")  # /wl-hero-stats
    out.append(f"<div class='wl-hero-mtc'>"
               f"<span class='label'>Closes in</span> "
               f"<span class='value'>{time_left_str(soonest_mtc)}</span>"
               f"</div>")
    out.append("</div>")  # /wl-hero-top

    # Probability mode: pin y-axis to 0..100 (the value range a Kalshi
    # binary contract can ever take) and reference the active bet's
    # entry probability (entry_price_cents) — which lives on the same
    # 0..100 scale. Otherwise auto-scale as before and reference the
    # active bet's strike value.
    if use_prob:
        if latest_active is not None:
            try:
                reference_strike = float(
                    latest_active.get("entry_price_cents"))
                strike_is_active = True
            except (TypeError, ValueError):
                reference_strike = None
                strike_is_active = False
        else:
            reference_strike = None
            strike_is_active = False
        strike_side = active_side
        y_pin_min, y_pin_max = 0.0, 100.0
    else:
        reference_strike = active_strike
        strike_side = active_side
        strike_is_active = active_strike is not None
        y_pin_min, y_pin_max = None, None

    # Chart plots the chosen series (probability when present, else
    # Kalshi's implied-underlying forecast). svg_kalshi_chart handles
    # the <2 datapoint case internally.
    out.append(svg_kalshi_chart(
        chart_history, chart_display,
        reference_strike=reference_strike,
        strike_side=strike_side,
        strike_is_active_bet=strike_is_active,
        contract_open_ts=contract_open_ts,
        contract_close_ts=contract_close_ts,
        total_volume=total_volume,
        y_min=y_pin_min,
        y_max=y_pin_max,
    ))
    out.append("</div>")


def _render_watchlist(out: List[str], watchlist: List[dict],
                      model: dict | None,
                      underlying_history: List[dict] | None = None,
                      display: dict | None = None,
                      latest_active: dict | None = None,
                      bot_active_bets: List[dict] | None = None,
                      kalshi_history: List[dict] | None = None,
                      prob_history: List[dict] | None = None,
                      atm_market: dict | None = None,
                      contract_open_ts: float | None = None,
                      contract_close_ts: float | None = None,
                      event_title: str | None = None,
                      threshold_source: dict | None = None,
                      edge_cfg: dict | None = None,
                      validator_cfg: dict | None = None,
                      risk_caps: dict | None = None,
                      hedge_cfg: dict | None = None,
                      available_bots: List[dict] | None = None,
                      current_bot: str = "",
                      period_key: str = "all") -> None:
    out.append("<div class='section'><h2>"
               "Watchlist — model vs market</h2>"
               "<div class='body'>")
    # Bot dropdown moved above the tab bar (per user request) so it
    # applies uniformly across tabs.

    # Current-prediction card row (Current price, Predicted next week,
    # etc.) sits between the bot dropdown and the Active bet so the
    # model's view comes right after the bot selector. Dash every
    # value when there's no live contract — either the event has
    # closed (close_ts in the past) or no Kalshi markets were
    # returned at all (between events). The model snapshot is for
    # an event that no longer exists in either case.
    contract_is_closed = (
        contract_close_ts is None
        or contract_close_ts <= datetime.now(timezone.utc).timestamp()
    )
    _render_current_prediction(out, model, display=display,
                                 contract_is_closed=contract_is_closed)

    # Buy-criteria reference button — rendered as a small circle-i info
    # icon inline with the Active-bet h3 so it sits next to the
    # section title (compact, doesn't take a row of its own). Click
    # opens the same shared modal with the full rules.
    rules_payload = json.dumps({
        "edge": edge_cfg or {},
        "validators": validator_cfg or {},
        "risk": risk_caps or {},
        "hedge": hedge_cfg or {},
        "_source": threshold_source or {"source": "fallback",
                                          "captured_at": None,
                                          "missing_keys": []},
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
    # Inline style only adds flex layout for the h3 + button row; the
    # default .subhead margin-top (16px) collapses with the row's
    # margin-bottom (14px) above to give exactly Summary's rhythm.
    # Active-bets section header. ``Active bet`` (singular) used to
    # render only the most recent open position — but a bot can hold
    # multiple positions concurrently (NBA picks one game per night
    # but holds the YES and NO sides on different games' tickers).
    # Now we render the full list in the same shared table the home
    # summary uses, so the count here matches the per-bot row count
    # in the cross-bot summary.
    bets = list(bot_active_bets or [])
    # Backwards compat: when only `latest_active` is plumbed in (older
    # callers), fall back to a single-bet list. The new caller always
    # passes ``bot_active_bets``.
    if not bets and latest_active:
        bets = [latest_active]
    # Map of every held ticker to its position record. Used by the
    # strike-ladder verdict cell so EVERY held strike gets the
    # HOLDING badge — not just the most-recently-opened one (the
    # previous single-`latest_active` lookup left other concurrently-
    # held strikes still rendering "BUY YES", which is what prompted
    # this fix).
    held_by_ticker = {b.get("ticker"): b for b in bets if b.get("ticker")}
    n_bets = len(bets)
    label = ("Active bets" if n_bets > 1
              else "Active bet")
    count_suffix = (f" <span class='small gray'>({n_bets})</span>"
                     if n_bets > 1 else "")
    out.append(
        "<h3 class='subhead' "
        "style='display:flex;align-items:center;gap:8px;'>"
        f"{label}{count_suffix} {rules_icon_html}</h3>"
    )
    if bets:
        enriched_rows: List[dict] = []
        for ab in bets:
            enriched = dict(ab)
            wl_match = next(
                (w for w in (watchlist or [])
                 if w.get("ticker") == ab.get("ticker")),
                None,
            )
            if wl_match:
                enriched.setdefault("floor_strike", wl_match.get("strike_low"))
                enriched.setdefault("cap_strike", wl_match.get("strike_high"))
                enriched.setdefault("minutes_to_close",
                                      wl_match.get("minutes_to_close"))
                if enriched.get("mark_yes_ask") is None:
                    enriched["mark_yes_ask"] = wl_match.get("yes_ask_cents")
                if enriched.get("mark_no_ask") is None:
                    enriched["mark_no_ask"] = wl_match.get("no_ask_cents")
            enriched["_display"] = display or {}
            enriched_rows.append(enriched)
        # Most-recently opened first (consistent with the home table).
        enriched_rows.sort(key=lambda r: r.get("opened_at", ""), reverse=True)
        # Pass the watchlist + event title + sport-bot flag through so
        # the active-bets row mirrors the title and side text of the
        # ticker table directly underneath. Wrap in a dedicated scroll
        # container that mirrors the strike-ladder's styling (sticky
        # header, soft border) but with a lighter section-grey
        # background so the table contrasts the chart panel sitting
        # right above it.
        out.append("<div class='watchlist-active-scroll'>")
        _render_active_bets_table(
            out, enriched_rows, show_bot=False,
            chart_link=True, hedge_cfg=hedge_cfg,
            watchlist=watchlist,
            event_title=event_title,
            is_sport_bot=(current_bot in
                          {"nba", "tennis", "table-tennis", "darts"}),
            display=display)
        out.append("</div>")
    else:
        out.append("<div class='empty'>No active bets right now.</div>")

    # ── Hero header + chart (Kalshi-style) ────────────────────────────────
    # Top-line metrics for the underlying the bot tracks: current value,
    # % change vs the start of the chart window, total Kalshi volume
    # across the watchlist, and time-to-close on the soonest market.
    # Chart pulls candlesticks live from Kalshi when configured.
    _render_watchlist_hero(out, watchlist, model,
                           underlying_history or [],
                           display or {}, latest_active,
                           kalshi_history=kalshi_history,
                           prob_history=prob_history or [],
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
        # Net-of-fee EV. The Kalshi entry fee (ceil(0.07 × p × (1−p))
        # per contract) is charged at open; settlement at 0¢ / 100¢ has
        # no exit fee, so on a held-to-settle bet the only deduction is
        # the entry fee. Per-$1-contract figure: divide cents by 100.
        fee_yes_d = kalshi_fee_cents(ya, 1) / 100.0 if ya is not None else 0.0
        fee_no_d = kalshi_fee_cents(na, 1) / 100.0 if na is not None else 0.0
        ev_yes = ((p_yes_blend - be_yes) - half_spread_d - fee_yes_d
                  if p_yes_blend is not None and be_yes is not None else None)
        ev_no = (((1.0 - p_yes_blend) - be_no) - half_spread_d - fee_no_d
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
    # zero open interest aren't tradeable and clutter the table. Rows
    # that set ``_skip_oi_filter`` (e.g. billboard markets that may
    # have null Kalshi-side OI early in the chart week but are still
    # the correct surface to show on the dashboard) opt out.
    watchlist = [r for r in watchlist
                 if r.get("_skip_oi_filter")
                 or (r.get("open_interest") or 0) > 0]
    # Sort: sport bots (one row per game / match) have no strike axis,
    # so order by actionability — BUY-eligible verdicts first, then by
    # |best EV| descending — mirroring the tennis-specific table the
    # standard renderer is replacing. Non-sport bots keep the strike
    # ascending sort that drives the natural ladder layout.
    is_sport_bot = current_bot in {"nba", "tennis", "table-tennis", "darts"}
    if is_sport_bot:
        def _sport_sort_key(r: dict) -> Tuple[int, float]:
            v = r.get("bot_verdict") or "SKIP"
            actionable = 0 if v in ("BUY_YES", "BUY_NO") else 1
            ev = r.get("_best_ev")
            try:
                ev_mag = -abs(float(ev)) if ev is not None else 0.0
            except (TypeError, ValueError):
                ev_mag = 0.0
            return (actionable, ev_mag)
        watchlist = sorted(watchlist, key=_sport_sort_key)
    else:
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
    # Sport bots (NBA + tennis-shape) show Title + Side: the Title
    # carries Kalshi's published YES question ("Will MIN win the
    # SAS vs MIN game?") and Side carries the team / player being
    # bet on. Non-sport bots (gas / CPI / jobless) keep the
    # strike-band Question column too. is_sport_bot was already
    # computed above to drive the sport-specific row sort.
    if is_sport_bot:
        head_cols = (
            "<th title='Kalshi-published contract title — the "
            "YES question shown on the market page.'>Title</th>"
            "<th title='Who the bot is betting will win.'>Side</th>"
        )
    else:
        head_cols = (
            "<th title='Kalshi-published contract title — the "
            "YES question shown on the market page.'>Title</th>"
            "<th>Question</th>"
        )
    out.append("<div class='watchlist-scroll'>"
               "<table><thead><tr>"
               "<th>Ticker</th>"
               f"{head_cols}"
               "<th class='num' title='Open interest — total contracts currently held open across all traders on this strike.'>Total contracts</th>"
               # My % sits to the left of Kalshi %. Both columns (plus
               # Edge + EV) render their YES value stacked on top in
               # green and their NO value on the bottom in red — the
               # row-cell renderer (_stacked() below) emits the
               # .cell-stack td that the CSS prints two rows tall.
               # The legacy "yes | no" sub-label is dropped from the
               # header now that vertical position + colour convey
               # the side.
               "<th class='num' title='Bot model probability — YES on top (green), NO on bottom (red).'>My %</th>"
               "<th class='num' title='Kalshi market price — YES on top (green), NO on bottom (red). Each side&apos;s implied probability that side wins.'>Kalshi %</th>"
               "<th class='num' title='Edge = my probability − Kalshi price, per side. YES on top (green), NO on bottom (red).'>Edge</th>"
               "<th class='num' title='Expected value per $1 contract, per side, net of half-spread and the Kalshi entry fee. YES on top (green), NO on bottom (red).'>EV</th>"
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
        # Derive missing side from the other when only one ask is
        # quoted — render as a plain number (no "~" prefix) so the
        # cell parses as a real percentage. The derivation is exact
        # for binary contracts (YES + NO must sum to 100¢), so the
        # tilde was just adding noise.
        if ya_c is not None:
            kyes_str = f"{ya_c}%"
        elif na_c is not None:
            kyes_str = f"{100 - na_c}%"
        else:
            kyes_str = "—"
        if na_c is not None:
            kno_str = f"{na_c}%"
        elif ya_c is not None:
            kno_str = f"{100 - ya_c}%"
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

        # My YES / My NO render in default white. Row-level dimming
        # via row-suspect handles "this strike isn't a buy" — once
        # the row is actionable (white), BOTH probabilities render
        # at full opacity so the user can read the model's view of
        # each side cleanly.
        ev_yes_v = v.get("_ev_yes")
        ev_no_v = v.get("_ev_no")
        bot_verdict_pre = v.get("bot_verdict", "SKIP")
        my_yes_cls = ""
        my_no_cls = ""

        # ── Verdict — two states only ──────────────────────────────────
        # Rules:
        #   HOLDING YES / HOLDING NO — bot has an open position on this
        #     strike. Wins over the model's current view so the row
        #     reflects what was actually done, not a contradictory
        #     fresh recommendation. Critical for consistency with the
        #     "Active bet" table above — without this, a row we bought
        #     YES on can show a different state once the market moves.
        #   SKIP — every other row. The model's recommendation (BUY
        #     YES / BUY NO / hold off / blocked-by-gate) shows up in
        #     the Edge / EV / tooltip columns; the Verdict column
        #     itself just reports "have we taken this position or
        #     not". The prior BUY YES / BUY NO / WATCH verdicts were
        #     retired per user request to keep the column to two
        #     stable states.
        held_bet = held_by_ticker.get(ticker)
        is_bought = held_bet is not None
        bought_side = ((held_bet.get("side") or "").upper()
                       if held_bet else "")
        bot_verdict = v.get("bot_verdict", "SKIP")
        reason = v.get("rejection_reason") or ""
        best_ev_v = v.get("_best_ev")
        best_side_v = v.get("_best_side")
        tt = f" title='{html.escape(reason)}'" if reason else ""
        if is_bought and bought_side in ("YES", "NO"):
            # HOLDING badge keeps its YES/NO colouring (the badge pill
            # tints itself — not the surrounding row, which now reads
            # in plain white). Tooltip surfaces entry price + the
            # model's current take so the user can audit "is the
            # model still on board with this position?"
            held_cls = "badge-yes" if bought_side == "YES" else "badge-no"
            entry_c = held_bet.get("entry_price_cents")
            entry_part = f" @ {entry_c}c" if entry_c is not None else ""
            model_part = ""
            if best_ev_v is not None and best_side_v in ("YES", "NO"):
                _ev_sign = "+" if best_ev_v > 0 else "−"
                model_part = (f" · model now: {best_side_v} "
                              f"(EV {_ev_sign}${abs(best_ev_v):.2f})")
            held_tt = (f"You are holding {bought_side}{entry_part}"
                       f"{model_part}")
            badge = (f"<span class='badge {held_cls}' "
                     f"title='{html.escape(held_tt)}'>"
                     f"HOLDING {bought_side}</span>")
        else:
            # Tooltip carries the model's recommendation when there
            # is one, so the user can still see "model would buy YES,
            # EV $0.05" on hover even though the cell says SKIP.
            skip_tt = reason
            if best_ev_v is not None and best_side_v in ("YES", "NO"):
                _ev_sign = "+" if best_ev_v > 0 else "−"
                rec = (f"model favours {best_side_v} "
                       f"(EV {_ev_sign}${abs(best_ev_v):.2f})")
                skip_tt = (f"{rec} · {reason}" if reason else rec)
            tt_attr = (f" title='{html.escape(skip_tt)}'"
                       if skip_tt else "")
            badge = f"<span class='badge badge-skip'{tt_attr}>SKIP</span>"
        # A row is a "good buy opportunity" when the bot would actually
        # take a position on it: BUY_YES/BUY_NO verdict + positive EV
        # + no validator flags. Rows that don't clear all three get
        # greyed out so the user sees only actionable rows in colour.
        is_buyable = (
            bot_verdict_pre in ("BUY_YES", "BUY_NO")
            and best_ev_v is not None and best_ev_v > 0
            and not flags
        )
        classes: List[str] = []
        title_attr = ""
        if is_bought:
            classes.append("row-bought")
            classes.append("bought-yes" if bought_side == "YES"
                           else "bought-no" if bought_side == "NO"
                           else "")
            entry_c = held_bet.get("entry_price_cents")
            contracts = held_bet.get("contracts")
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
        else:
            # Per user request: only HOLDING rows get full-bright
            # white text. Every other row — buyable or not — renders
            # dimmed via .row-suspect so the holdings stand out at
            # a glance against the rest of the watchlist.
            classes.append("row-suspect")
            if flags:
                reason = "Validator flags: " + ", ".join(flags)
            elif best_ev_v is None or best_ev_v <= 0:
                reason = "No positive edge"
            elif not is_buyable:
                reason = "Bot verdict not actionable"
            elif best_side_v in ("YES", "NO"):
                _ev_sign = "+" if (best_ev_v or 0) > 0 else "−"
                reason = (f"Model favours {best_side_v} "
                            f"(EV {_ev_sign}${abs(best_ev_v or 0):.2f}) — "
                            f"no position held")
            else:
                reason = "No position held"
            title_attr = (" title='" + html.escape(reason) + "'")
        row_cls = (f" class='{' '.join(classes)}'" if classes else "") + title_attr

        # Pre-format EV cells. Zero or missing values render as a plain
        # "0" instead of the signed "+$0.00" or "—" dash — both convey
        # the same thing ("no actionable edge") and "0" reads cleaner
        # across a dense table.
        def _ev_cell(ev: float | None) -> tuple[str, str]:
            if ev is None:
                return "0", "gray"
            if round(float(ev), 2) == 0:
                return "0", "gray"
            cls_, _ = _ev_status(ev)
            sign = "+" if ev > 0 else "−"
            return f"{sign}${abs(ev):.2f}", cls_
        ev_yes_str, ev_yes_cls = _ev_cell(ev_yes_v)
        ev_no_str, ev_no_cls = _ev_cell(ev_no_v)

        # Edge cells — model probability for the side minus Kalshi's
        # ask price for the same side. Positive = bot's model disagrees
        # with Kalshi in that side's favour. Half-spread is NOT
        # subtracted here (that's what the EV column is for); Edge is
        # the raw model-vs-market gap so the user can read the bot's
        # underlying view independent of liquidity cost.
        def _edge(p: float | None, ask_c: int | None) -> float | None:
            if p is None or ask_c is None:
                return None
            return float(p) - (int(ask_c) / 100.0)
        edge_yes_v = _edge(p, ya_c)
        edge_no_v = _edge((1.0 - float(p)) if p is not None else None,
                           na_c)
        def _edge_cell(e: float | None) -> tuple[str, str]:
            if e is None:
                return "0", "gray"
            pp = e * 100.0
            if round(pp) == 0:
                return "0", "gray"
            cls_ = ("green" if e >= 0.05 else
                    "yellow" if e > 0 else
                    "red" if e <= -0.02 else "gray")
            return f"{pp:+.0f}%", cls_
        edge_yes_str, edge_yes_cls = _edge_cell(edge_yes_v)
        edge_no_str, edge_no_cls = _edge_cell(edge_no_v)

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
        # Pass the row's strike value through ``data-strike`` so the
        # JS row-click hook can draw a horizontal threshold line on the
        # chart at this market's strike level (non-sport bots) or at the
        # ticker's YES ask price (sport bots, where strike isn't a
        # meaningful concept).
        sl = v.get("strike_low")
        sh = v.get("strike_high")
        try:
            strike_attr = f" data-strike='{float(sl):.6f}'" if sl is not None else ""
        except (TypeError, ValueError):
            strike_attr = ""
        try:
            yes_attr = f" data-yes-prob='{int(ya_c) / 100.0:.4f}'" if ya_c is not None else ""
        except (TypeError, ValueError):
            yes_attr = ""
        # Sport bots: Title + Side. Non-sport: Title + Question.
        # ``watchlist_title_use_event`` overrides the per-market title
        # with the event-level one (used by the unemployment bot, where
        # every row of the table is the same Initial-Claims week and
        # the Kalshi event title — "Initial jobless claims for the
        # week ending May 9, 2026" — is what the user wants in the
        # Title column instead of the per-strike "200K" repetition).
        if (display or {}).get("watchlist_title_use_event") and event_title:
            title_text = event_title
        else:
            title_text = v.get("title") or ""
        if is_sport_bot:
            # Tennis-shape rows pre-fill _yes_label / _no_label with the
            # player names (the ticker doesn't carry a parseable tricode
            # the way KXNBAGAME does). Prefer those when set; fall back
            # to the NBA tricode parser for KXNBAGAME tickers.
            yes_team = v.get("_yes_label") or _side_tricode_from_ticker(
                ticker, "YES")
            opp_team = v.get("_no_label") or _side_tricode_from_ticker(
                ticker, "NO")
            if yes_team:
                side_cell = (
                    f"<td><strong>{html.escape(str(yes_team))}</strong>"
                    f"<br><span class='small gray'>vs "
                    f"{html.escape(str(opp_team))}</span></td>"
                )
            else:
                side_cell = f"<td>{html.escape(qstr)}</td>"
            middle_cells = (
                f"<td>{html.escape(title_text)}</td>"
                f"{side_cell}"
            )
        else:
            middle_cells = (
                f"<td>{html.escape(title_text)}</td>"
                f"<td>{html.escape(qstr)}</td>"
            )
        # User-requested layout: YES on top in green, NO on bottom in
        # red — across every side-paired column (My %, Kalshi %, Edge,
        # EV). Replaces the previous horizontal "yes | no" rendering.
        # The side is conveyed by vertical position + colour; we drop
        # the per-value green/yellow/red EV-magnitude tinting since
        # the side colour now dominates the cell.
        def _stacked(yes_val: str, no_val: str,
                       field: str, extra_tt: str = "") -> str:
            return (
                f"<td class='num cell-stack' "
                f"data-field='{field}'{extra_tt}>"
                f"<div class='side-yes green' data-side='yes'>{yes_val}</div>"
                f"<div class='side-no red' data-side='no'>{no_val}</div>"
                f"</td>"
            )
        kalshi_cell = _stacked(kyes_str, kno_str, "kalshi")
        my_cell     = _stacked(my_yes_str, my_no_str, "my",
                                  extra_tt=(my_yes_tt or my_no_tt))
        edge_cell   = _stacked(edge_yes_str, edge_no_str, "edge")
        ev_cell     = _stacked(ev_yes_str, ev_no_str, "ev")

        out.append(f"<tr{row_cls} data-ticker='{tt_esc}'{strike_attr}{yes_attr}>"
                   f"<td class='mono'>{ticker_cell}</td>"
                   f"{middle_cells}"
                   f"<td class='num' data-field='oi'>{oi_str}</td>"
                   f"{my_cell}"
                   f"{kalshi_cell}"
                   f"{edge_cell}"
                   f"{ev_cell}"
                   f"<td data-field='verdict'>{badge}</td></tr>")
    out.append("</tbody></table></div>")
    # Append the row-click JS hook so clicks on a watchlist row draw a
    # horizontal threshold line on the hero chart at the row's value.
    out.append(_WATCHLIST_ROW_CLICK_JS)
    out.append("</div></div>")


# Vanilla-JS hook for the Kalshi watchlist tables. Each watchlist row
# carries ``data-ticker`` plus (for non-sport bots) ``data-strike`` and
# (sport / NBA) ``data-yes-prob``. On click:
#
#   * Highlight the selected row.
#   * Find the existing chart's SVG and overlay a horizontal dashed
#     line at the row's strike value (non-sport: strike on the
#     underlying-value Y axis) or at the YES ask probability (sport:
#     plotted on a 0..1 secondary axis). The line replaces any prior
#     overlay so each click "moves" the threshold rather than stacking.
#
# The chart-coordinate math is delegated to a per-chart ``data-y-min``
# / ``data-y-max`` pair the chart renderer stamps onto its SVG. When
# the chart isn't tagged, the JS bails quietly so it never breaks the
# page. The hero-chart implementation in ``_render_watchlist_hero``
# emits these attributes alongside the existing polyline.
_WATCHLIST_ROW_CLICK_JS = """
<script>
(function() {
  // Both the strike-ladder table (id='watchlist-tbody') and the per-bot
  // active-bets table (id='wl-active-tbody', stamped only on the
  // watchlist tab via ``chart_link=True``) feed the same hero chart.
  // Listing them together lets clicks in either table draw the same
  // overlay line, with rows in both tables clearing each other's
  // selection (so the user always sees one active selection).
  const tbodies = Array.from(document.querySelectorAll(
    \"tbody[data-chart-link], tbody#watchlist-tbody\"
  ));
  if (!tbodies.length) return;

  function findChart() {
    // Look for the kalshi-history hero chart's SVG. It carries
    // ``data-chart='wl-hero'`` so we can find it without grabbing
    // any stray SVG (the favicon is one). Returns null when the
    // page renders the empty-frame placeholder.
    return document.querySelector(\"svg[data-chart='wl-hero']\");
  }

  function clearOverlay(svg) {
    svg.querySelectorAll('.row-overlay').forEach(n => n.remove());
  }

  function drawOverlay(svg, label, color) {
    // Read the chart's plotted Y range from the SVG's data attrs
    // and draw a horizontal line at the requested data value.
    const rangeAttr = svg.getAttribute('data-y-range');
    if (!rangeAttr) return;
    const [yMin, yMax, yPad, padT, padL, padR] = rangeAttr.split(',').map(parseFloat);
    if (!Number.isFinite(yMin) || !Number.isFinite(yMax) || yMax === yMin) return;
    const value = label.value;
    if (!Number.isFinite(value)) return;
    if (value < yMin || value > yMax) return;
    const w = svg.viewBox && svg.viewBox.baseVal && svg.viewBox.baseVal.width || 760;
    const h = svg.viewBox && svg.viewBox.baseVal && svg.viewBox.baseVal.height || 220;
    const innerH = h - padT - yPad;
    const y = padT + (1 - (value - yMin) / (yMax - yMin)) * innerH;
    const xL = padL, xR = w - padR;

    const ns = 'http://www.w3.org/2000/svg';
    const line = document.createElementNS(ns, 'line');
    line.setAttribute('class', 'row-overlay');
    line.setAttribute('x1', xL); line.setAttribute('x2', xR);
    line.setAttribute('y1', y); line.setAttribute('y2', y);
    line.setAttribute('stroke', color);
    line.setAttribute('stroke-width', '1.5');
    line.setAttribute('stroke-dasharray', '6,4');
    svg.appendChild(line);

    const text = document.createElementNS(ns, 'text');
    text.setAttribute('class', 'row-overlay');
    text.setAttribute('x', xR - 4);
    text.setAttribute('y', y - 4);
    text.setAttribute('fill', color);
    text.setAttribute('text-anchor', 'end');
    text.setAttribute('font-size', '11');
    text.setAttribute('font-weight', '600');
    text.appendChild(document.createTextNode(label.text));
    svg.appendChild(text);
  }

  function setSelected(activeTr) {
    // Clear selection on every chart-linked row across all tables so
    // only one row is highlighted at a time.
    tbodies.forEach(function (tb) {
      tb.querySelectorAll('tr').forEach(function (r) {
        r.classList.toggle('row-selected', r === activeTr);
      });
    });
  }

  function onClick(ev) {
    const tr = ev.target.closest('tr');
    if (!tr) return;
    // The criteria-button has its own modal handler — don't hijack it.
    if (ev.target.closest('.criteria-btn')) return;
    if (!tr.dataset || !tr.dataset.ticker) return;
    setSelected(tr);
    const svg = findChart();
    if (!svg) return;
    clearOverlay(svg);
    // Strike overlay removed per user request. Sport bots can still
    // stamp data-yes-prob (0..1); those rows are no-ops on the
    // probability-axis hero chart since the range check rejects
    // values < yMin or > yMax.
    const yesProb = parseFloat(tr.dataset.yesProb);
    if (Number.isFinite(yesProb)) {
      drawOverlay(svg, {
        value: yesProb,
        text: 'YES ' + (yesProb * 100).toFixed(0) + '%',
      }, '#58a6ff');
    }
  }

  tbodies.forEach(function (tb) { tb.addEventListener('click', onClick); });
})();
</script>
<style>
#watchlist-tbody tr,
#wl-active-tbody tr { cursor: pointer; }
#watchlist-tbody tr.row-selected td,
#wl-active-tbody tr.row-selected td { background: #1f2630 !important; }
#watchlist-tbody tr:hover td,
#wl-active-tbody tr:hover td { background: #1c222b; }
/* Held-position rows share the same neutral grey hover / selected
   tint as every other row — the per-row colouring was retired in
   favour of the HOLDING badge in the Verdict column. */
</style>
"""


def _render_contract_rules(out: List[str], watchlist: List[dict],
                           current_bot: str,
                           contract_close_ts: float | None = None) -> None:
    """Section 7 — actual Kalshi resolution rule, one paragraph.

    All KXAAAGASW contracts share the same template (only the strike
    differs). We pick the first market with a populated rules_primary
    and render that single paragraph verbatim. The strike count and
    range are noted so the user knows it applies across the series.

    No info-button popup here — Kalshi's "View full rules" web-UI text
    isn't exposed via their public API (the API only returns
    rules_primary, which is what's shown verbatim below). Scraping
    the Kalshi UI would be fragile + against their TOS, so we just
    show the rules_primary and accept that it's the short summary.

    When the current event has already closed, the empty-state copy
    swaps from "rules not cached yet" to "There are no rules" — once
    the contract has settled there's no upcoming bot tick that will
    fill the gap, so promising one would be misleading.
    """
    out.append("<div class='section'><h2>6 · Kalshi rules — "
               "how the market resolves</h2>"
               "<div class='body rules'>")

    # Find a representative rules_primary string.
    primary = ""
    for v in watchlist:
        rp = (v.get("rules_primary") or "").strip()
        if rp:
            primary = rp
            break

    if not primary:
        # "No live contract" covers both cases: the current event has
        # already settled, OR Kalshi returned no active markets at all
        # (between events). The "next bot tick will populate it"
        # promise only makes sense while a contract exists.
        contract_is_closed = (
            contract_close_ts is None
            or contract_close_ts <= datetime.now(timezone.utc).timestamp()
        )
        if contract_is_closed:
            out.append("<div class='empty'>There are no rules.</div>")
        else:
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
               f"{html.escape(primary)}</p>")
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
                if tab_key not in {"home", "watchlist", "models",
                                    "history", "seasons"}:
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
                if bot.get("dashboard_type") == "tennis":
                    from . import tennis as _tennis
                    payload_wl = _tennis.load_watchlist(
                        bot.get("watchlist_json_path"))
                    watchlist = _tennis.build_standard_watchlist_rows(payload_wl)
                    bot_active_bets = _tennis.active_bets_for_rollup(
                        bot.get("sim_state_path"),
                        watchlist_path=bot.get("watchlist_json_path"),
                    )
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
                    bot_active_bets = []
                    latest_active = None
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
                        and bot.get("dashboard_type") not in ("tennis",
                                                              "billboard")):
                    from . import kalshi_client
                    # Sport series like KXNBAGAME have many concurrent
                    # open events (one per game on the slate). Narrowing
                    # to the most-imminent event hides the rest of the
                    # slate from the watchlist; flag those bots so the
                    # client returns every market with a future close.
                    all_open_events = bot.get("key") in {
                        "nba", "tennis", "table-tennis", "darts",
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
                        and bot.get("dashboard_type") not in ("tennis",
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

                # Global cross-bot fetches (these power the Summary section
                # which is identical regardless of which bot is selected).
                global_summary = fetch_global_summary(self.bots,
                                                       period_days=period_days)
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
                for b in self.bots:
                    # Tennis bot doesn't have a sim.db, but it does have a
                    # metrics.json / coefficients.json. Synthesize a model
                    # dict for the card grid so the tennis bot shows up
                    # alongside the Kalshi bots on the home page.
                    if b.get("dashboard_type") in ("tennis", "survivor", "billboard"):
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
                        # active-bets table.
                        if b.get("dashboard_type") == "tennis":
                            for ab in _tennis.active_bets_for_rollup(
                                b.get("sim_state_path"),
                                watchlist_path=b.get("watchlist_json_path"),
                            ):
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
                        # Closed paper bets into the cross-bot history
                        # so hedge exits + natural settles surface on
                        # the History tab. Same row shape the standard
                        # ``fetch_bet_history`` produces.
                        for h in adapter.closed_positions_for_rollup(
                            b.get("sim_state_path"), limit=50,
                        ):
                            h["_bot_name"] = b["name"]
                            h["_bot_key"] = b["key"]
                            h["_dashboard_type"] = b.get("dashboard_type") or "standard"
                            h["_display"] = b.get("display") or {}
                            global_history.append(h)
                        continue
                    if b.get("dashboard_type") and b["dashboard_type"] != "standard":
                        continue
                    for ab in fetch_active_bets_with_marks(b["db_path"]):
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
                    for h in fetch_bet_history(b["db_path"], limit=50):
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
                # the active-bet table per request. Tennis-shape bots
                # have no sim.db; pull their closed paper-bet rollup
                # from the tennis adapter instead.
                if bot.get("dashboard_type") == "tennis":
                    from . import tennis as _tennis
                    bot_closed_positions = _tennis.closed_positions_for_rollup(
                        bot.get("sim_state_path"), limit=100,
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
                 bot_hedge_cfg, threshold_source) = resolve_bot_thresholds(
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
                    threshold_source=threshold_source,
                    available_bots=self.bots,
                    current_bot=bot["key"],
                    period_key=period_key,
                    tab_key=tab_key,
                    bot_models=bot_models,
                    model_view=model_view,
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
                if bot.get("dashboard_type") == "tennis":
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
                    )
                    payload_dict["bot"] = bot["key"]
                    payload_dict["type"] = "billboard"
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
          edge_cfg: dict, validator_cfg: dict, hedge_cfg: dict) -> None:
    Handler.bots = bots
    Handler.risk_caps = risk_caps
    Handler.edge_cfg = edge_cfg
    Handler.validator_cfg = validator_cfg
    Handler.hedge_cfg = hedge_cfg
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
    # Tennis odds snapshotter. Every 60s walks each tennis-shape
    # bot's watchlist.json and appends per-match price snapshots so
    # the in-game model has a velocity / volatility / divergence
    # time-series to read — same shape NBA gets from market_views.
    from .in_game import tennis_snapshotter
    tennis_snapshotter.start_daemon(bots)
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

    # Bot registry comes from the dashboard YAML. Each entry's "available"
    # flag reflects whether the bot's sim.db exists on disk — selecting an
    # unavailable bot in the dropdown shows a friendly stub.
    bots: list[dict] = []
    for b in cfg.bots:
        if b.dashboard_type == "tennis":
            # Tennis bot is "available" if the watchlist JSON exists. The
            # tennis-forecast cron writes it on every refresh; an empty
            # rows list still counts as available (renders empty state).
            available = bool(b.watchlist_json_path
                             and Path(b.watchlist_json_path).exists())
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
            "series_ticker": b.series_ticker,
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

    host = args.host or cfg.host
    port = args.port or cfg.port
    serve(host, port, bots, risk_caps, edge_cfg, validator_cfg, hedge_cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
