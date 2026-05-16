"""Bitcoin Watchlist page.

One row per active Kalshi BTC contract — the columns the spec lists:

    contract · expiry · BTC price · threshold · direction · distance to
    threshold · YES bid · YES ask · NO bid · NO ask · implied prob ·
    model prob · edge · signal · confidence · liquidity · open paper
    position · unrealized P&L · last updated

Reads from the BTC bot's ``data/sim.db`` (read-only) via the helpers
in ``model_card`` (re-exported from the bot package onto the dashboard
host's PYTHONPATH).
"""
from __future__ import annotations

import html
import logging
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import List, Optional

from . import _bitcoin_common as common

log = logging.getLogger("dashboard.bitcoin_watchlist")


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


def _model_card_payload(db_path: str | Path) -> dict:
    rows = _q(
        db_path,
        "SELECT * FROM model_snapshots ORDER BY id DESC LIMIT 1",
    )
    snap = rows[0] if rows else {}
    price_rows = _q(
        db_path,
        "SELECT * FROM btc_price_snapshots ORDER BY id DESC LIMIT 1",
    )
    price = price_rows[0] if price_rows else {}
    return {
        "btc_price": snap.get("current_gas_price") or price.get("price_usd"),
        "price_age_seconds": snap.get("price_age_seconds"),
        "price_source": snap.get("price_source") or price.get("source"),
        "contracts_scanned": snap.get("contracts_scanned"),
        "strongest_signal": snap.get("strongest_signal"),
        "strongest_signal_ticker": snap.get("strongest_signal_ticker"),
        "strongest_confidence": snap.get("strongest_confidence"),
        "strongest_edge": snap.get("strongest_edge"),
        "strongest_model_prob": snap.get("strongest_model_prob"),
        "strongest_kalshi_prob": snap.get("strongest_kalshi_prob"),
        "open_positions": snap.get("open_positions") or 0,
        "daily_realized_pnl_cents": snap.get("daily_realized_pnl_cents") or 0,
        "win_rate": snap.get("win_rate") or 0.0,
        "last_updated": snap.get("captured_at"),
    }


def _watchlist_rows(db_path: str | Path, limit: int = 80) -> List[dict]:
    return _q(
        db_path,
        """
        WITH last_feat AS (
            SELECT f.* FROM btc_contract_features f
            JOIN (
                SELECT ticker, MAX(id) max_id
                FROM btc_contract_features GROUP BY ticker
            ) lf ON f.id = lf.max_id
        ),
        last_sig AS (
            SELECT s.* FROM btc_signals s
            JOIN (
                SELECT ticker, MAX(id) max_id
                FROM btc_signals GROUP BY ticker
            ) ls ON s.id = ls.max_id
        ),
        last_market AS (
            SELECT m.* FROM kalshi_btc_market_snapshots m
            JOIN (
                SELECT ticker, MAX(id) max_id
                FROM kalshi_btc_market_snapshots GROUP BY ticker
            ) lm ON m.id = lm.max_id
        ),
        open_pos AS (
            SELECT * FROM btc_paper_trades WHERE status = 'open'
        )
        SELECT
            f.ticker, f.captured_at, f.btc_price, f.threshold,
            f.threshold_high, f.direction, f.minutes_to_expiry,
            f.distance_to_threshold, f.pct_distance_to_threshold,
            f.yes_bid_cents, f.yes_ask_cents, f.no_bid_cents,
            f.no_ask_cents, f.kalshi_implied_yes_prob, f.fair_prob_yes,
            f.edge_yes, f.edge_no, f.liquidity_score,
            s.signal, s.confidence, s.edge AS signal_edge,
            s.reason AS signal_reason, s.expected_value_cents,
            s.validator_block_reason,
            lm.title AS market_title, lm.close_time,
            op.position_id AS open_position_id, op.side AS open_side,
            op.entry_price_cents AS open_entry_cents,
            op.contracts AS open_contracts,
            (SELECT m.mid_cents FROM position_marks m
                WHERE m.position_id = op.position_id) AS open_mark_cents
        FROM last_feat f
        LEFT JOIN last_sig s ON s.ticker = f.ticker
        LEFT JOIN last_market lm ON lm.ticker = f.ticker
        LEFT JOIN open_pos op ON op.ticker = f.ticker
        ORDER BY
            CASE s.signal
                WHEN 'BUY_YES' THEN 0
                WHEN 'BUY_NO' THEN 0
                WHEN 'HOLD' THEN 1
                ELSE 2
            END,
            ABS(COALESCE(s.edge, 0)) DESC,
            f.minutes_to_expiry ASC
        LIMIT ?
        """,
        (limit,),
    )


def _direction_pill(direction: Optional[str]) -> str:
    d = (direction or "").lower()
    label_map = {
        "above": ("green", "ABOVE"),
        "below": ("red", "BELOW"),
        "range": ("yellow", "RANGE"),
        "exact": ("gray", "EXACT"),
    }
    cls, label = label_map.get(d, ("gray", (direction or "—").upper()))
    return f"<span class='pill {cls}'>{html.escape(label)}</span>"


def _unrealized_pnl_cents(row: dict) -> Optional[int]:
    """Mark-minus-entry × size for the open paper leg, if any."""
    pid = row.get("open_position_id")
    if pid is None:
        return None
    side = (row.get("open_side") or "").lower()
    entry = row.get("open_entry_cents")
    contracts = row.get("open_contracts") or 0
    mark: Optional[float] = None
    if side == "yes":
        if row.get("yes_bid_cents") is not None:
            mark = row["yes_bid_cents"]
        elif row.get("no_ask_cents") is not None:
            mark = 100 - row["no_ask_cents"]
    elif side == "no":
        if row.get("no_bid_cents") is not None:
            mark = row["no_bid_cents"]
        elif row.get("yes_ask_cents") is not None:
            mark = 100 - row["yes_ask_cents"]
    if mark is None or entry is None:
        return None
    try:
        return int((float(mark) - float(entry)) * float(contracts))
    except (TypeError, ValueError):
        return None


def _render_hero(out: List[str], card: dict) -> None:
    """Top stats row — current BTC price + open paper positions + strongest signal."""
    out.append("<div class='row compact'>")

    btc = card.get("btc_price")
    age = card.get("price_age_seconds")
    age_text = ""
    if age is not None:
        try:
            age_text = f"updated {float(age):.0f}s ago"
        except (TypeError, ValueError):
            age_text = ""
    out.append(
        "<div class='card'>"
        "<div class='label'>BTC / USD spot</div>"
        f"<div class='value'>{common.fmt_btc(btc)}</div>"
        f"<div class='label muted'>{html.escape(age_text)} · "
        f"{html.escape(card.get('price_source') or '—')}</div>"
        "</div>"
    )
    out.append(
        "<div class='card'>"
        "<div class='label'>Contracts scanned</div>"
        f"<div class='value'>{int(card.get('contracts_scanned') or 0)}</div>"
        f"<div class='label muted'>Open paper: "
        f"{int(card.get('open_positions') or 0)}</div>"
        "</div>"
    )
    strongest = card.get("strongest_signal")
    edge = card.get("strongest_edge")
    out.append(
        "<div class='card'>"
        "<div class='label'>Strongest signal</div>"
        f"<div class='value'>{common.signal_pill(strongest)}</div>"
        f"<div class='label muted'>edge "
        f"{common.fmt_pct(edge, 1) if edge is not None else '—'} · "
        f"conf {common.fmt_pct(card.get('strongest_confidence'), 0)}</div>"
        "</div>"
    )
    pnl = card.get("daily_realized_pnl_cents") or 0
    out.append(
        "<div class='card'>"
        "<div class='label'>Today simulated P&amp;L</div>"
        f"<div class='value {common.pnl_class(pnl)}'>"
        f"{common.fmt_signed_dollars(pnl)}</div>"
        f"<div class='label muted'>win rate "
        f"{common.fmt_pct(card.get('win_rate'), 0)}</div>"
        "</div>"
    )
    out.append("</div>")


def _render_watchlist_table(out: List[str], rows: List[dict]) -> None:
    out.append("<div class='section'><h2>Bitcoin Watchlist</h2><div class='body'>")
    if not rows:
        out.append("<div class='empty'>"
                   "No Kalshi BTC contracts captured yet. "
                   "If the bot is running it should fill in within a tick or two."
                   "</div></div></div>")
        return
    out.append(
        "<table><thead><tr>"
        "<th>Contract</th>"
        "<th>Expiry</th>"
        "<th class='num' title='BTC/USD spot at last feature row'>BTC</th>"
        "<th class='num'>Threshold</th>"
        "<th>Direction</th>"
        "<th class='num' title='BTC price − threshold'>Distance</th>"
        "<th class='num'>YES bid</th>"
        "<th class='num'>YES ask</th>"
        "<th class='num'>NO bid</th>"
        "<th class='num'>NO ask</th>"
        "<th class='num' title='Mid-of-book in probability units'>Implied</th>"
        "<th class='num' title='Bot fair-probability estimate'>Model</th>"
        "<th class='num' title='Edge on the favored side'>Edge</th>"
        "<th>Signal</th>"
        "<th class='num'>Conf</th>"
        "<th class='num' title='0 = empty book / 1 = tight + deep'>Liq</th>"
        "<th>Open paper</th>"
        "<th class='num'>Unreal. P&amp;L</th>"
        "<th>Updated</th>"
        "</tr></thead><tbody>"
    )
    for r in rows:
        signal = r.get("signal")
        edge = r.get("signal_edge")
        edge_cell = (f"<span class='{common.pnl_class(edge or 0)}'>"
                     f"{common.fmt_pct(edge, 1) if edge is not None else '—'}"
                     "</span>")
        unrealized = _unrealized_pnl_cents(r)
        open_side = r.get("open_side")
        open_cell = "—"
        if open_side:
            side_cls = "side-yes" if open_side == "yes" else "side-no"
            open_cell = (
                f"<span class='{side_cls}'>{html.escape(open_side.upper())}</span>"
                f" × {int(r.get('open_contracts') or 0)}"
                f" @ {common.fmt_cents(r.get('open_entry_cents'))}"
            )
        out.append("<tr>")
        out.append(
            f"<td class='mono'>"
            f"<div>{html.escape(r.get('ticker') or '')}</div>"
            f"<div class='small gray btc-question' title='"
            f"{html.escape(r.get('market_title') or '')}'>"
            f"{html.escape(r.get('market_title') or '')}</div>"
            f"</td>"
        )
        out.append(
            f"<td>{common.fmt_minutes(r.get('minutes_to_expiry'))}"
            f"<div class='small gray'>"
            f"{html.escape(common.fmt_ts(r.get('close_time'))[:16])}</div>"
            f"</td>"
        )
        out.append(f"<td class='num'>{common.fmt_btc(r.get('btc_price'))}</td>")
        threshold_label = common.fmt_btc(r.get("threshold"))
        if r.get("threshold_high") is not None:
            threshold_label += f" – {common.fmt_btc(r['threshold_high'])}"
        out.append(f"<td class='num'>{threshold_label}</td>")
        out.append(f"<td>{_direction_pill(r.get('direction'))}</td>")
        out.append(f"<td class='num'>"
                   f"{common.fmt_distance(r.get('distance_to_threshold'))}"
                   f"</td>")
        out.append(f"<td class='num'>{common.fmt_cents(r.get('yes_bid_cents'))}</td>")
        out.append(f"<td class='num'>{common.fmt_cents(r.get('yes_ask_cents'))}</td>")
        out.append(f"<td class='num'>{common.fmt_cents(r.get('no_bid_cents'))}</td>")
        out.append(f"<td class='num'>{common.fmt_cents(r.get('no_ask_cents'))}</td>")
        out.append(f"<td class='num'>"
                   f"{common.fmt_pct(r.get('kalshi_implied_yes_prob'), 0)}</td>")
        out.append(f"<td class='num'>"
                   f"{common.fmt_pct(r.get('fair_prob_yes'), 0)}</td>")
        out.append(f"<td class='num'>{edge_cell}</td>")
        out.append(f"<td>{common.signal_pill(signal)}</td>")
        out.append(f"<td class='num'>{common.fmt_pct(r.get('confidence'), 0)}</td>")
        liq = r.get("liquidity_score")
        liq_text = f"{float(liq):.2f}" if liq is not None else "—"
        out.append(f"<td class='num'>{liq_text}</td>")
        out.append(f"<td>{open_cell}</td>")
        out.append(f"<td class='num {common.pnl_class(unrealized)}'>"
                   f"{common.fmt_signed_dollars(unrealized)}</td>")
        out.append(f"<td class='small gray'>{common.fmt_ts(r.get('captured_at'))}</td>")
        out.append("</tr>")
    out.append("</tbody></table></div></div>")


def render(*, db_path, available_bots: List[dict],
           current_bot_key: str, tab_key: str = "watchlist") -> str:
    out: List[str] = []
    common.render_chrome(
        out, title="Bitcoin Watchlist",
        active_tab="watchlist", available_bots=available_bots,
        current_bot_key=current_bot_key,
    )
    card = _model_card_payload(db_path)
    rows = _watchlist_rows(db_path, limit=80)
    if not card.get("btc_price") and not rows:
        common.empty_state_card(
            out,
            "Bitcoin Live Forecast is configured but no data yet. "
            "Wait one poll cycle (~45s) after the service starts.",
        )
        common.render_footer(out)
        return "".join(out)
    _render_hero(out, card)
    _render_watchlist_table(out, rows)
    common.render_footer(out)
    return "".join(out)
