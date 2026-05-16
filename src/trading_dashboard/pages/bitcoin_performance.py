"""Bitcoin Performance page.

Aggregations of the BTC bot's simulated trade history:

  - cumulative + daily P&L
  - win rate, average return per trade, best / worst
  - signal accuracy & performance by signal type (BUY_YES vs BUY_NO)
  - performance by time-to-expiry bucket
  - the recent paper-trade history table
"""
from __future__ import annotations

import html
import logging
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Dict, List, Optional

from . import _bitcoin_common as common

log = logging.getLogger("dashboard.bitcoin_performance")


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


def _q(db_path: str | Path, sql: str, params: tuple = ()) -> List[dict]:
    c = _ro_conn(db_path)
    if c is None:
        return []
    try:
        with closing(c):
            return [dict(r) for r in c.execute(sql, params).fetchall()]
    except sqlite3.DatabaseError as exc:
        log.warning("query failed: %s", exc)
        return []


def _summary(db_path: str | Path) -> Dict[str, object]:
    rows = _q(
        db_path,
        "SELECT realized_pnl_cents FROM btc_paper_trades WHERE status='closed'",
    )
    pnls = [int(r["realized_pnl_cents"] or 0) for r in rows]
    wins = sum(1 for p in pnls if p > 0)
    losses = sum(1 for p in pnls if p < 0)
    return {
        "trades": len(pnls),
        "wins": wins,
        "losses": losses,
        "cumulative_pnl_cents": sum(pnls),
        "win_rate": (wins / (wins + losses)) if (wins + losses) else None,
        "avg_return_cents": (sum(pnls) / len(pnls)) if pnls else None,
        "best_trade_cents": max(pnls) if pnls else None,
        "worst_trade_cents": min(pnls) if pnls else None,
    }


def _daily_pnl(db_path: str | Path, days: int = 14) -> List[dict]:
    return _q(
        db_path,
        "SELECT date(exit_at) AS day, COUNT(*) trades,"
        " SUM(CASE WHEN realized_pnl_cents>0 THEN 1 ELSE 0 END) wins,"
        " SUM(CASE WHEN realized_pnl_cents<0 THEN 1 ELSE 0 END) losses,"
        " COALESCE(SUM(realized_pnl_cents),0) pnl_cents"
        " FROM btc_paper_trades WHERE status='closed' AND exit_at IS NOT NULL"
        " GROUP BY date(exit_at) ORDER BY day DESC LIMIT ?",
        (days,),
    )


def _by_signal(db_path: str | Path) -> List[dict]:
    return _q(
        db_path,
        "SELECT entry_signal AS signal,"
        " COUNT(*) trades,"
        " SUM(CASE WHEN realized_pnl_cents>0 THEN 1 ELSE 0 END) wins,"
        " SUM(CASE WHEN realized_pnl_cents<0 THEN 1 ELSE 0 END) losses,"
        " COALESCE(SUM(realized_pnl_cents),0) pnl_cents,"
        " COALESCE(AVG(realized_pnl_cents),0) avg_pnl_cents"
        " FROM btc_paper_trades WHERE status='closed'"
        " GROUP BY entry_signal ORDER BY trades DESC",
    )


def _by_time_bucket(db_path: str | Path) -> List[dict]:
    return _q(
        db_path,
        "SELECT"
        " CASE"
        "   WHEN entry_minutes_to_expiry < 15 THEN '0-15m'"
        "   WHEN entry_minutes_to_expiry < 60 THEN '15-60m'"
        "   WHEN entry_minutes_to_expiry < 240 THEN '1-4h'"
        "   WHEN entry_minutes_to_expiry < 1440 THEN '4-24h'"
        "   ELSE '24h+'"
        " END AS bucket,"
        " COUNT(*) trades,"
        " SUM(CASE WHEN realized_pnl_cents>0 THEN 1 ELSE 0 END) wins,"
        " SUM(CASE WHEN realized_pnl_cents<0 THEN 1 ELSE 0 END) losses,"
        " COALESCE(SUM(realized_pnl_cents),0) pnl_cents"
        " FROM btc_paper_trades WHERE status='closed'"
        " GROUP BY bucket"
        " ORDER BY"
        "  CASE bucket"
        "   WHEN '0-15m' THEN 0"
        "   WHEN '15-60m' THEN 1"
        "   WHEN '1-4h' THEN 2"
        "   WHEN '4-24h' THEN 3"
        "   ELSE 4"
        "  END",
    )


def _recent(db_path: str | Path, limit: int = 50) -> List[dict]:
    return _q(
        db_path,
        "SELECT * FROM btc_paper_trades ORDER BY"
        " COALESCE(exit_at, entry_at) DESC LIMIT ?",
        (limit,),
    )


# ----------------------------------------------------------------------- #
# Renderers
# ----------------------------------------------------------------------- #

def _render_summary_cards(out: List[str], s: dict) -> None:
    pnl = s.get("cumulative_pnl_cents") or 0
    out.append("<div class='row compact'>")
    out.append(
        "<div class='card'>"
        "<div class='label'>Closed paper trades</div>"
        f"<div class='value'>{int(s.get('trades') or 0)}</div>"
        f"<div class='label muted'>"
        f"{int(s.get('wins') or 0)}–{int(s.get('losses') or 0)}</div>"
        "</div>"
    )
    out.append(
        "<div class='card'>"
        "<div class='label'>Cumulative simulated P&amp;L</div>"
        f"<div class='value {common.pnl_class(pnl)}'>"
        f"{common.fmt_signed_dollars(pnl)}</div>"
        f"<div class='label muted'>win rate "
        f"{common.fmt_pct(s.get('win_rate'), 0)}</div>"
        "</div>"
    )
    out.append(
        "<div class='card'>"
        "<div class='label'>Avg return / trade</div>"
        f"<div class='value {common.pnl_class(s.get('avg_return_cents'))}'>"
        f"{common.fmt_signed_dollars(s.get('avg_return_cents'))}</div>"
        f"<div class='label muted'>best "
        f"{common.fmt_signed_dollars(s.get('best_trade_cents'))} · "
        f"worst {common.fmt_signed_dollars(s.get('worst_trade_cents'))}</div>"
        "</div>"
    )
    out.append("</div>")


def _render_daily_table(out: List[str], rows: List[dict]) -> None:
    out.append("<div class='section'><h2>Daily simulated P&amp;L</h2>"
               "<div class='body'>")
    if not rows:
        out.append("<div class='empty'>No closed trades yet.</div>"
                   "</div></div>")
        return
    # Compute cumulative — daily rows come oldest-last from _daily_pnl
    # (we ORDER DESC there), so rebuild a chronological cumulative.
    chrono = list(reversed(rows))
    cum = 0
    for row in chrono:
        cum += int(row.get("pnl_cents") or 0)
        row["cumulative_pnl_cents"] = cum
    rows_for_display = list(reversed(chrono))
    out.append("<table><thead><tr>"
               "<th>Day</th>"
               "<th class='num'>Trades</th>"
               "<th class='num'>W</th>"
               "<th class='num'>L</th>"
               "<th class='num'>Daily P&amp;L</th>"
               "<th class='num'>Cumulative</th>"
               "</tr></thead><tbody>")
    for r in rows_for_display:
        pnl_c = int(r.get("pnl_cents") or 0)
        cum_c = int(r.get("cumulative_pnl_cents") or 0)
        out.append(
            "<tr>"
            f"<td>{html.escape(str(r.get('day') or '—'))}</td>"
            f"<td class='num'>{int(r.get('trades') or 0)}</td>"
            f"<td class='num pos'>{int(r.get('wins') or 0)}</td>"
            f"<td class='num neg'>{int(r.get('losses') or 0)}</td>"
            f"<td class='num {common.pnl_class(pnl_c)}'>"
            f"{common.fmt_signed_dollars(pnl_c)}</td>"
            f"<td class='num {common.pnl_class(cum_c)}'>"
            f"{common.fmt_signed_dollars(cum_c)}</td>"
            "</tr>"
        )
    out.append("</tbody></table></div></div>")


def _render_by_signal(out: List[str], rows: List[dict]) -> None:
    out.append("<div class='section'><h2>Performance by signal type</h2>"
               "<div class='body'>")
    if not rows:
        out.append("<div class='empty'>No closed trades yet.</div>"
                   "</div></div>")
        return
    out.append("<table><thead><tr>"
               "<th>Signal</th><th class='num'>Trades</th>"
               "<th class='num'>Wins</th><th class='num'>Losses</th>"
               "<th class='num'>Win rate</th>"
               "<th class='num'>Total P&amp;L</th>"
               "<th class='num'>Avg / trade</th>"
               "</tr></thead><tbody>")
    for r in rows:
        wins = int(r.get("wins") or 0)
        losses = int(r.get("losses") or 0)
        decided = wins + losses
        win_rate = (wins / decided) if decided else None
        pnl = int(r.get("pnl_cents") or 0)
        avg = float(r.get("avg_pnl_cents") or 0.0)
        out.append(
            "<tr>"
            f"<td>{common.signal_pill(r.get('signal'))}</td>"
            f"<td class='num'>{int(r.get('trades') or 0)}</td>"
            f"<td class='num pos'>{wins}</td>"
            f"<td class='num neg'>{losses}</td>"
            f"<td class='num'>{common.fmt_pct(win_rate, 0)}</td>"
            f"<td class='num {common.pnl_class(pnl)}'>"
            f"{common.fmt_signed_dollars(pnl)}</td>"
            f"<td class='num {common.pnl_class(avg)}'>"
            f"{common.fmt_signed_dollars(avg)}</td>"
            "</tr>"
        )
    out.append("</tbody></table></div></div>")


def _render_by_time_bucket(out: List[str], rows: List[dict]) -> None:
    out.append("<div class='section'><h2>Performance by time to expiry</h2>"
               "<div class='body'>")
    if not rows:
        out.append("<div class='empty'>No closed trades yet.</div>"
                   "</div></div>")
        return
    out.append("<table><thead><tr>"
               "<th>Time to expiry at entry</th>"
               "<th class='num'>Trades</th>"
               "<th class='num'>Wins</th>"
               "<th class='num'>Losses</th>"
               "<th class='num'>Win rate</th>"
               "<th class='num'>P&amp;L</th>"
               "</tr></thead><tbody>")
    for r in rows:
        wins = int(r.get("wins") or 0)
        losses = int(r.get("losses") or 0)
        decided = wins + losses
        win_rate = (wins / decided) if decided else None
        pnl = int(r.get("pnl_cents") or 0)
        out.append(
            "<tr>"
            f"<td>{html.escape(str(r.get('bucket') or '—'))}</td>"
            f"<td class='num'>{int(r.get('trades') or 0)}</td>"
            f"<td class='num pos'>{wins}</td>"
            f"<td class='num neg'>{losses}</td>"
            f"<td class='num'>{common.fmt_pct(win_rate, 0)}</td>"
            f"<td class='num {common.pnl_class(pnl)}'>"
            f"{common.fmt_signed_dollars(pnl)}</td>"
            "</tr>"
        )
    out.append("</tbody></table></div></div>")


def _render_recent(out: List[str], rows: List[dict]) -> None:
    out.append("<div class='section'><h2>Paper trade history</h2>"
               "<div class='body'>")
    if not rows:
        out.append("<div class='empty'>No paper trades yet.</div>"
                   "</div></div>")
        return
    out.append("<table><thead><tr>"
               "<th>Status</th><th>Ticker</th><th>Side</th>"
               "<th class='num'>Qty</th>"
               "<th class='num'>Entry</th>"
               "<th class='num'>Exit</th>"
               "<th class='num'>P&amp;L</th>"
               "<th>Signal</th>"
               "<th class='num'>Edge</th>"
               "<th class='num'>BTC @ entry</th>"
               "<th>Exit reason</th>"
               "<th>Opened</th><th>Closed</th>"
               "</tr></thead><tbody>")
    for r in rows:
        status = (r.get("status") or "").lower()
        status_cls = "pill green" if status == "closed" else "pill yellow"
        side = (r.get("side") or "").lower()
        side_cls = "side-yes" if side == "yes" else "side-no"
        pnl = r.get("realized_pnl_cents")
        out.append(
            "<tr>"
            f"<td><span class='{status_cls}'>{html.escape(status.upper())}</span></td>"
            f"<td class='mono'>{html.escape(r.get('ticker') or '')}</td>"
            f"<td><span class='{side_cls}'>{html.escape(side.upper())}</span></td>"
            f"<td class='num'>{int(r.get('contracts') or 0)}</td>"
            f"<td class='num'>{common.fmt_cents(r.get('entry_price_cents'))}</td>"
            f"<td class='num'>{common.fmt_cents(r.get('exit_price_cents'))}</td>"
            f"<td class='num {common.pnl_class(pnl)}'>"
            f"{common.fmt_signed_dollars(pnl)}</td>"
            f"<td>{common.signal_pill(r.get('entry_signal'))}</td>"
            f"<td class='num'>{common.fmt_pct(r.get('entry_edge'), 1)}</td>"
            f"<td class='num'>{common.fmt_btc(r.get('entry_btc_price'))}</td>"
            f"<td>{html.escape(r.get('exit_reason') or '—')}</td>"
            f"<td class='small gray'>{common.fmt_ts(r.get('entry_at'))}</td>"
            f"<td class='small gray'>{common.fmt_ts(r.get('exit_at'))}</td>"
            "</tr>"
        )
    out.append("</tbody></table></div></div>")


def render(*, db_path, available_bots: List[dict],
           current_bot_key: str, tab_key: str = "performance") -> str:
    out: List[str] = []
    common.render_chrome(
        out, title="Bitcoin Performance",
        active_tab="performance", available_bots=available_bots,
        current_bot_key=current_bot_key,
    )
    summary = _summary(db_path)
    if (summary.get("trades") or 0) == 0:
        common.empty_state_card(
            out,
            "No simulated trades closed yet. "
            "The Watchlist tab shows live scoring before any trades fire.",
        )
        common.render_footer(out)
        return "".join(out)
    _render_summary_cards(out, summary)
    _render_daily_table(out, _daily_pnl(db_path))
    _render_by_signal(out, _by_signal(db_path))
    _render_by_time_bucket(out, _by_time_bucket(db_path))
    _render_recent(out, _recent(db_path))
    common.render_footer(out)
    return "".join(out)
