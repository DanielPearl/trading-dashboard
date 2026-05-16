"""Bitcoin Live Forecast dashboard adapter.

Routes the bot's ``dashboard_type: bitcoin`` requests to the right page
under ``trading_dashboard.pages``. Also exposes the cross-bot rollup
helpers (``summary_for_rollup``, ``active_bets_for_rollup``,
``closed_positions_for_rollup``) so the home page's cards include the
Bitcoin bot's totals.

Keeping this thin so the per-page HTML lives where the user expects
(``pages/bitcoin_watchlist.py`` + ``pages/bitcoin_performance.py``).
"""
from __future__ import annotations

import logging
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Dict, List, Optional, Tuple

log = logging.getLogger("dashboard.bitcoin_live_forecast")


BITCOIN_TABS: List[Tuple[str, str]] = [
    ("home",        "Home"),
    ("watchlist",   "Bitcoin Watchlist"),
    ("performance", "Bitcoin Performance"),
    ("history",     "History"),
]


def _ro_conn(db_path: str | Path) -> Optional[sqlite3.Connection]:
    p = Path(db_path)
    if not p.exists():
        return None
    try:
        c = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
        c.row_factory = sqlite3.Row
        return c
    except sqlite3.OperationalError:
        return None


def _rows(db_path: str | Path, sql: str, params: tuple = ()) -> List[dict]:
    c = _ro_conn(db_path)
    if c is None:
        return []
    try:
        with closing(c):
            return [dict(r) for r in c.execute(sql, params).fetchall()]
    except sqlite3.DatabaseError as exc:
        log.warning("query failed: %s", exc)
        return []


# ----------------------------------------------------------------------- #
# Cross-bot rollup adapters (same shape as the tennis adapter's helpers)
# ----------------------------------------------------------------------- #

def summary_for_rollup(db_path: str | Path) -> Dict:
    """Bot-level numbers used by the home page's cross-bot summary."""
    rows = _rows(
        db_path,
        "SELECT realized_pnl_cents, entry_price_cents, contracts,"
        " status, entry_at, exit_at"
        " FROM btc_paper_trades",
    )
    open_count = sum(1 for r in rows if r.get("status") == "open")
    closed = [r for r in rows if r.get("status") == "closed"]
    pnls = [int(r.get("realized_pnl_cents") or 0) for r in closed]
    wins = sum(1 for p in pnls if p > 0)
    losses = sum(1 for p in pnls if p < 0)
    exposure = sum(
        int(r.get("entry_price_cents") or 0) * int(r.get("contracts") or 0)
        for r in rows if r.get("status") == "open"
    )
    potential_gain = sum(
        (100 - int(r.get("entry_price_cents") or 0)) * int(r.get("contracts") or 0)
        for r in rows if r.get("status") == "open"
    )
    realized = sum(pnls)
    return {
        "total_bets": len(rows),
        "open_count": open_count,
        "exposure_cents": exposure,
        "closed_count": len(closed),
        "realized_pnl_cents": realized,
        "wins_lifetime": wins,
        "losses_lifetime": losses,
        "avg_win_cents": int(round(sum(p for p in pnls if p > 0)
                                    / max(wins, 1))) if wins else 0,
        "avg_loss_cents": int(round(sum(p for p in pnls if p < 0)
                                     / max(losses, 1))) if losses else 0,
        "biggest_win_cents": max(pnls) if pnls else 0,
        "biggest_loss_cents": min(pnls) if pnls else 0,
        "this_week_pnl_cents": realized,   # paper-only; whole history ~= recent
        "this_week_wins": wins,
        "this_week_losses": losses,
        "bets_today": sum(1 for r in rows
                          if (r.get("entry_at") or "")[:10]
                          == _today_str()),
        # Period filters aren't honored here yet — paper P&L doesn't
        # have enough volume for meaningful period scoping. Use the
        # lifetime totals for both.
        "period_bets_made": len(rows),
        "period_net_pnl_cents": realized,
        "period_wins": wins,
        "period_losses": losses,
        "period_money_spent_cents": sum(
            int(r.get("entry_price_cents") or 0) * int(r.get("contracts") or 0)
            for r in rows
        ),
        "period_money_gained_cents": sum(
            int(r.get("entry_price_cents") or 0) * int(r.get("contracts") or 0)
            + int(r.get("realized_pnl_cents") or 0)
            for r in closed
        ),
        "potential_gain_cents": potential_gain,
    }


def _today_str() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def active_bets_for_rollup(db_path: str | Path) -> List[dict]:
    """Bitcoin bot's open paper positions in the dashboard's row shape."""
    rows = _rows(
        db_path,
        "SELECT t.position_id AS id, t.ticker, t.side, t.contracts,"
        " t.entry_price_cents, t.entry_at, t.entry_threshold AS strike_low,"
        " t.entry_threshold AS strike_high, t.entry_direction AS direction,"
        " t.entry_signal, t.entry_confidence, t.entry_model_prob,"
        " t.entry_kalshi_prob, t.entry_minutes_to_expiry,"
        " m.yes_ask_cents AS mark_yes_ask, m.no_ask_cents AS mark_no_ask,"
        " m.yes_bid_cents AS mark_yes_bid, m.mid_cents AS mark_mid,"
        " m.updated_at AS mark_updated_at"
        " FROM btc_paper_trades t"
        " LEFT JOIN position_marks m ON m.position_id = t.position_id"
        " WHERE t.status = 'open'"
        " ORDER BY t.entry_at DESC",
    )
    for r in rows:
        r["opened_at"] = r.get("entry_at")
        r["bot_key"] = "bitcoin-live-forecast"
    return rows


def closed_positions_for_rollup(db_path: str | Path,
                                limit: int = 100) -> List[dict]:
    rows = _rows(
        db_path,
        "SELECT t.position_id AS id, t.ticker, t.side,"
        " t.entry_price_cents, t.exit_price_cents, t.contracts,"
        " t.realized_pnl_cents, t.entry_at AS opened_at,"
        " t.exit_at AS exited_at, t.exit_reason, t.entry_threshold,"
        " t.entry_direction"
        " FROM btc_paper_trades t WHERE t.status = 'closed'"
        " ORDER BY t.exit_at DESC LIMIT ?",
        (limit,),
    )
    for r in rows:
        r["bot_key"] = "bitcoin-live-forecast"
    return rows


# ----------------------------------------------------------------------- #
# Page dispatch
# ----------------------------------------------------------------------- #

def render_page(*, db_path: str | Path,
                available_bots: List[dict],
                current_bot_key: str,
                tab_key: str = "watchlist") -> str:
    """Dispatch into the Bitcoin Watchlist / Performance page renderers.

    Falls back to the Watchlist page for unknown ``tab_key`` values so
    deep links never 404.
    """
    from .pages import bitcoin_performance, bitcoin_watchlist
    tab = tab_key if tab_key in {k for k, _ in BITCOIN_TABS} else "watchlist"
    if tab == "performance":
        return bitcoin_performance.render(
            db_path=db_path,
            available_bots=available_bots,
            current_bot_key=current_bot_key,
            tab_key=tab,
        )
    # "home" and "history" route back to the main dashboard's standard
    # pages by re-rendering the Watchlist (the dashboard's URL handler
    # passes them through to the standard render path before reaching
    # us, but include them here for safety).
    return bitcoin_watchlist.render(
        db_path=db_path,
        available_bots=available_bots,
        current_bot_key=current_bot_key,
        tab_key=tab,
    )
