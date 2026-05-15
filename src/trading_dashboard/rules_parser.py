"""Rules Parser dashboard view.

Read-only viewer over the Rules Parser bot's SQLite DB
(``rules_intel.db``). Surfaces:

  - the homepage card (model_summary_for_card) — simulated win-rate
    and realised P&L
  - the per-bot Watchlist page — current simulated positions, recent
    signals, ambiguity warnings, settled P&L, source reliability
  - rollup adapters that feed the cross-bot active-bets + history
    panes on the dashboard home

The dashboard process never imports the trader; it opens the DB with
``mode=ro`` so an accidental write from the viewer is impossible.
"""
from __future__ import annotations

import html
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("dashboard.rules_parser")


# Tabs shown in the watchlist page. Same shape as the other adapters'
# tab lists so the rendering helpers can reuse the navigation idiom.
RULES_PARSER_TABS: List[Tuple[str, str]] = [
    ("home",      "Home"),
    ("watchlist", "Watchlist"),
    ("history",   "History"),
]


# --------------------------------------------------------------------------- #
# Read helpers — open the DB read-only so the dashboard can't corrupt it
# even by accident. Every function tolerates a missing DB and returns a
# safe empty default rather than raising.
# --------------------------------------------------------------------------- #

def _connect_ro(db_path: str | Path) -> Optional[sqlite3.Connection]:
    p = Path(db_path)
    if not p.exists():
        return None
    try:
        c = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
        c.row_factory = sqlite3.Row
        return c
    except sqlite3.OperationalError:
        return None


def _rows(c: sqlite3.Connection, sql: str,
          params: tuple = ()) -> List[Dict[str, Any]]:
    return [dict(r) for r in c.execute(sql, params).fetchall()]


def _scalar(c: sqlite3.Connection, sql: str,
            params: tuple = (), default: int = 0) -> int:
    row = c.execute(sql, params).fetchone()
    return int(row[0]) if row and row[0] is not None else default


def load_signals(db_path: str | Path, limit: int = 200,
                 status: Optional[str] = None) -> List[Dict[str, Any]]:
    """Latest signals joined with their contract + matched news item."""
    c = _connect_ro(db_path)
    if c is None:
        return []
    try:
        where = ""
        params: tuple = ()
        if status:
            where = "WHERE s.status = ?"
            params = (status,)
        sql = f"""
            SELECT
                s.id                AS signal_id,
                s.ticker            AS ticker,
                s.confidence        AS confidence,
                s.source_reliability AS source_reliability,
                s.clause_match_score AS clause_match_score,
                s.expected_resolution AS expected_resolution,
                s.suggested_action   AS suggested_action,
                s.reasoning          AS reasoning,
                s.yes_price_cents_at_detect AS yes_at_detect,
                s.no_price_cents_at_detect  AS no_at_detect,
                s.status             AS status,
                s.created_at         AS created_at,
                c.title              AS title,
                c.subtitle           AS subtitle,
                c.yes_bid_cents      AS yes_bid_cents,
                c.yes_ask_cents      AS yes_ask_cents,
                c.no_bid_cents       AS no_bid_cents,
                c.no_ask_cents       AS no_ask_cents,
                c.last_price_cents   AS last_price_cents,
                c.status             AS contract_status,
                c.event_ticker       AS event_ticker,
                c.series_ticker      AS series_ticker,
                cl.text              AS clause_text,
                cl.clause_type       AS clause_type,
                n.url                AS news_url,
                n.title              AS news_title,
                n.summary            AS news_summary,
                n.published_at       AS news_published_at,
                src.name             AS source_name
            FROM signals s
            LEFT JOIN contracts c     ON c.ticker = s.ticker
            LEFT JOIN rule_clauses cl ON cl.id = s.clause_id
            LEFT JOIN news_items  n   ON n.id = s.news_item_id
            LEFT JOIN sources     src ON src.id = n.source_id
            {where}
            ORDER BY s.created_at DESC
            LIMIT ?
        """
        return _rows(c, sql, params + (limit,))
    finally:
        c.close()


def load_open_positions(db_path: str | Path) -> List[Dict[str, Any]]:
    c = _connect_ro(db_path)
    if c is None:
        return []
    try:
        return _rows(c, """
            SELECT
                p.id                AS position_id,
                p.signal_id         AS signal_id,
                p.ticker            AS ticker,
                p.side              AS side,
                p.contracts         AS contracts,
                p.entry_price_cents AS entry_price_cents,
                p.entry_cost_cents  AS entry_cost_cents,
                p.expected_resolution AS expected_resolution,
                p.confidence        AS confidence,
                p.mark_price_cents  AS mark_price_cents,
                p.mark_pnl_cents    AS mark_pnl_cents,
                p.mark_updated_at   AS mark_updated_at,
                p.opened_at         AS opened_at,
                c.title             AS title,
                c.subtitle          AS subtitle,
                c.status            AS contract_status,
                c.close_time        AS close_time,
                c.expiration_time   AS expiration_time,
                s.reasoning         AS reasoning,
                s.clause_match_score AS clause_match_score,
                s.source_reliability AS source_reliability
            FROM simulated_positions p
            LEFT JOIN contracts c ON c.ticker = p.ticker
            LEFT JOIN signals   s ON s.id     = p.signal_id
            WHERE p.status = 'open'
            ORDER BY p.opened_at DESC
        """)
    finally:
        c.close()


def load_closed_positions(db_path: str | Path,
                          limit: int = 100) -> List[Dict[str, Any]]:
    c = _connect_ro(db_path)
    if c is None:
        return []
    try:
        return _rows(c, """
            SELECT
                p.id                AS position_id,
                p.signal_id         AS signal_id,
                p.ticker            AS ticker,
                p.side              AS side,
                p.contracts         AS contracts,
                p.entry_price_cents AS entry_price_cents,
                p.entry_cost_cents  AS entry_cost_cents,
                p.expected_resolution AS expected_resolution,
                p.confidence        AS confidence,
                p.settled_outcome   AS settled_outcome,
                p.pnl_cents         AS pnl_cents,
                p.status            AS status,
                p.opened_at         AS opened_at,
                p.settled_at        AS settled_at,
                c.title             AS title
            FROM simulated_positions p
            LEFT JOIN contracts c ON c.ticker = p.ticker
            WHERE p.status IN ('settled', 'voided')
            ORDER BY p.settled_at DESC
            LIMIT ?
        """, (limit,))
    finally:
        c.close()


def load_summary(db_path: str | Path) -> Dict[str, Any]:
    """Snapshot of bot state used by the homepage card + Home tab."""
    base = {
        "contracts": 0,
        "event_based_contracts": 0,
        "clauses": 0,
        "news_items_24h": 0,
        "signals_total": 0,
        "signals_24h": 0,
        "signals_flagged": 0,
        "signals_traded": 0,
        "open_positions": 0,
        "open_exposure_cents": 0,
        "open_mark_pnl_cents": 0,
        "settled_positions": 0,
        "wins": 0,
        "losses": 0,
        "voids": 0,
        "realised_pnl_cents": 0,
        "sources": 0,
        "available": False,
    }
    c = _connect_ro(db_path)
    if c is None:
        return base
    try:
        base["available"] = True
        base["contracts"] = _scalar(c, "SELECT COUNT(*) FROM contracts")
        base["event_based_contracts"] = _scalar(
            c, "SELECT COUNT(*) FROM contracts WHERE is_event_based = 1")
        base["clauses"] = _scalar(c, "SELECT COUNT(*) FROM rule_clauses")
        base["sources"] = _scalar(c, "SELECT COUNT(*) FROM sources")
        base["news_items_24h"] = _scalar(c,
            "SELECT COUNT(*) FROM news_items "
            "WHERE fetched_at >= datetime('now', '-1 days')")
        base["signals_total"]   = _scalar(c, "SELECT COUNT(*) FROM signals")
        base["signals_24h"] = _scalar(c,
            "SELECT COUNT(*) FROM signals "
            "WHERE created_at >= datetime('now', '-1 days')")
        base["signals_flagged"] = _scalar(c,
            "SELECT COUNT(*) FROM signals WHERE status = 'flagged'")
        base["signals_traded"]  = _scalar(c,
            "SELECT COUNT(*) FROM signals WHERE status = 'traded'")
        # Position aggregates
        try:
            r = c.execute(
                """
                SELECT COUNT(*) AS n,
                       COALESCE(SUM(entry_cost_cents), 0) AS exposure,
                       COALESCE(SUM(mark_pnl_cents), 0)   AS mark
                FROM simulated_positions WHERE status='open'
                """).fetchone()
            base["open_positions"]     = int(r["n"] or 0)
            base["open_exposure_cents"] = int(r["exposure"] or 0)
            base["open_mark_pnl_cents"] = int(r["mark"] or 0)
            r = c.execute(
                """
                SELECT
                    COUNT(*) AS n,
                    SUM(CASE WHEN pnl_cents > 0 THEN 1 ELSE 0 END) AS wins,
                    SUM(CASE WHEN pnl_cents <= 0 AND status='settled'
                              THEN 1 ELSE 0 END) AS losses,
                    SUM(CASE WHEN status='voided' THEN 1 ELSE 0 END) AS voids,
                    COALESCE(SUM(pnl_cents), 0) AS realised
                FROM simulated_positions
                WHERE status IN ('settled','voided')
                """).fetchone()
            base["settled_positions"] = int(r["n"] or 0)
            base["wins"]   = int(r["wins"]   or 0)
            base["losses"] = int(r["losses"] or 0)
            base["voids"]  = int(r["voids"]  or 0)
            base["realised_pnl_cents"] = int(r["realised"] or 0)
        except sqlite3.OperationalError:
            # Older DB without simulated_positions table — just leave
            # the zero defaults. Migration happens on next runner boot.
            pass
        return base
    finally:
        c.close()


def load_sources(db_path: str | Path) -> List[Dict[str, Any]]:
    c = _connect_ro(db_path)
    if c is None:
        return []
    try:
        return _rows(c, """
            SELECT name, url, kind, reliability,
                   success_count, error_count,
                   last_fetched_at, last_error
            FROM sources
            ORDER BY reliability DESC, name
        """)
    finally:
        c.close()


# --------------------------------------------------------------------------- #
# Adapters used by the main dashboard (homepage card + cross-bot rollup).
# Signatures mirror the tennis / survivor adapters so dashboard.py can
# dispatch on dashboard_type without bespoke argument plumbing.
# --------------------------------------------------------------------------- #

def model_summary_for_card(db_path: str | Path) -> Dict[str, Any]:
    """Synthesise the dict the homepage card renderer reads.

    The card template surfaces ``actual_wins`` / ``actual_losses`` and
    a ``period_net_pnl_cents`` rollup value. The other fields
    (classifier_accuracy, training_f1, …) are model-quality metrics
    that don't apply to a rules-arbitrage bot — leave them None and
    the card formatter renders "—".

    ``feature_count`` is repurposed to show *contracts watched* —
    the visual is "n features = n inputs the bot considers" which
    maps to "n contracts we're parsing rules for".
    """
    s = load_summary(db_path)
    return {
        "classifier_accuracy": None,
        "training_f1":         None,
        "training_precision":  None,
        "training_roc_auc":    None,
        "training_recall":     None,
        "feature_count":       s["event_based_contracts"] or s["contracts"],
        "actual_wins":   s["wins"],
        "actual_losses": s["losses"],
    }


def closed_positions_for_rollup(db_path: str | Path,
                                limit: int = 50) -> List[Dict[str, Any]]:
    """Settled simulated positions reshaped to the cross-bot history row
    shape (matches what ``fetch_bet_history`` returns for standard bots).
    Used by the dashboard's History tab so Rules Parser settled trades
    show up alongside the other bots' closed bets.
    """
    raw = load_closed_positions(db_path, limit=limit)
    out: List[Dict[str, Any]] = []
    for p in raw:
        out.append({
            "ticker":          p["ticker"],
            "title":           p.get("title") or p["ticker"],
            "side":            (p.get("side") or "").upper(),
            "contracts":       p.get("contracts") or 0,
            "entry_price_cents": p.get("entry_price_cents") or 0,
            "exit_price_cents":  100 if p.get("settled_outcome") == "YES"
                                 else (0 if p.get("settled_outcome") == "NO"
                                       else None),
            "pnl_cents":       p.get("pnl_cents") or 0,
            "opened_at":       p.get("opened_at") or "",
            "exited_at":       p.get("settled_at") or "",
            "exit_reason":     "settled" if p.get("status") == "settled"
                                else "void",
        })
    return out


def active_bets_for_rollup(db_path: str | Path) -> List[Dict[str, Any]]:
    """Open simulated positions reshaped for the cross-bot active-bets
    table on the dashboard home. Mirrors the column set the standard
    renderer expects from ``fetch_active_bets_with_marks``.
    """
    raw = load_open_positions(db_path)
    out: List[Dict[str, Any]] = []
    for p in raw:
        side = (p.get("side") or "").upper()
        entry = p.get("entry_price_cents") or 0
        contracts = p.get("contracts") or 0
        mark = p.get("mark_price_cents")
        # Reframe into the rollup row shape — kalshi_ask is the price
        # our SIDE would have to pay if entered now, used by the
        # active-bets table to display current prob.
        out.append({
            "ticker":            p["ticker"],
            "title":             p.get("title") or p["ticker"],
            "side":              side,
            "contracts":         contracts,
            "entry_price_cents": entry,
            "opened_at":         p.get("opened_at") or "",
            "mark_mid":          mark,
            "mark_yes_ask":      p.get("yes_ask_cents"),
            "mark_yes_bid":      p.get("yes_bid_cents"),
            "mark_no_ask":       p.get("no_ask_cents"),
            "mark_no_bid":       p.get("no_bid_cents"),
            "minutes_to_close":  None,  # computed by caller from ticker
        })
    return out


# --------------------------------------------------------------------------- #
# Formatting helpers — kept tiny and local so we don't depend on the
# main dashboard's helpers that aren't exposed at module level.
# --------------------------------------------------------------------------- #

def _fmt_cents(c: Optional[int]) -> str:
    if c is None:
        return "—"
    return f"{c}¢"


def _fmt_signed_dollars(c: Optional[int]) -> str:
    if c is None:
        return "—"
    sign = "+" if c > 0 else ("−" if c < 0 else "")
    return f"{sign}${abs(c)/100:.2f}"


def _fmt_pct(v: Optional[float], decimals: int = 0) -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v)*100:.{decimals}f}%"
    except (TypeError, ValueError):
        return "—"


def _fmt_ts(s: Optional[str]) -> str:
    if not s:
        return "—"
    return s.replace("T", " ")[:19]


def _pnl_class(c: Optional[int]) -> str:
    if c is None:
        return "gray"
    if c > 0:
        return "green"
    if c < 0:
        return "red"
    return "gray"


def _confidence_class(c: Optional[float]) -> str:
    if c is None:
        return "gray"
    try:
        v = float(c)
    except (TypeError, ValueError):
        return "gray"
    if v >= 0.85:
        return "green"
    if v >= 0.50:
        return "yellow"
    return "red"


# Ambiguity is a UI signal we surface inline on each row. Heuristic:
# expected_resolution == 'UNCERTAIN', OR clause_match_score < 0.4, OR
# source_reliability < 0.5. Any one of these fires the warning pill.
def _ambiguity_warning(row: Dict[str, Any]) -> Optional[str]:
    reasons: List[str] = []
    if (row.get("expected_resolution") or "").upper() == "UNCERTAIN":
        reasons.append("rule outcome is UNCERTAIN")
    try:
        if float(row.get("clause_match_score") or 0) < 0.4:
            reasons.append("weak clause match")
    except (TypeError, ValueError):
        pass
    try:
        if float(row.get("source_reliability") or 0) < 0.5:
            reasons.append("low-reliability source")
    except (TypeError, ValueError):
        pass
    if not reasons:
        return None
    return "; ".join(reasons)


# --------------------------------------------------------------------------- #
# Page renderer                                                               #
# --------------------------------------------------------------------------- #

def render_page(*, db_path: str | Path,
                available_bots: List[dict],
                current_bot_key: str,
                tab_key: str = "watchlist") -> str:
    """Whole HTML page for the Rules Parser view.

    Layout mirrors the whale-watcher page so the navigation is one
    idiom across bots:

        Bot dropdown  ←  bot picker (jumps to a different bot's view)
        [Home] [Watchlist] [History]   ← top-level tabs
        [stats cards row]              ← bot-level summary
        Section: Current bets — simulated
        Section: Recent signals + rule interpretation
        Section: Sources reliability
        Section: Settled simulated trades (P&L)
    """
    from .dashboard import CSS, _favicon_link, _render_bot_filter

    summary = load_summary(db_path)
    open_positions  = load_open_positions(db_path)
    closed = load_closed_positions(db_path, limit=200)
    signals = load_signals(db_path, limit=100)
    sources = load_sources(db_path)

    active_tab = tab_key if tab_key in {k for k, _ in RULES_PARSER_TABS} else "watchlist"

    out: List[str] = []
    out.append("<!doctype html><html><head>")
    out.append("<meta charset='utf-8'>")
    out.append("<meta http-equiv='refresh' content='30'>")
    out.append("<title>Rules Parser — Kalshi simulation dashboard</title>")
    out.append(_favicon_link())
    out.append(f"<style>{CSS}</style>")
    out.append("<style>"
               ".side-yes { color:#3fb950; font-weight:600; }"
               ".side-no  { color:#f85149; font-weight:600; }"
               ".pos { color:#3fb950; }"
               ".neg { color:#f85149; }"
               ".gray { color:#8b949e; }"
               ".pill { display:inline-block; padding:2px 8px; "
                 "border-radius:10px; font-size:11px; font-weight:600; "
                 "font-variant-numeric: tabular-nums; }"
               ".pill.green { background: rgba(63,185,80,0.18); "
                 "color:#3fb950; border:1px solid rgba(63,185,80,0.35); }"
               ".pill.red { background: rgba(248,81,73,0.15); "
                 "color:#f85149; border:1px solid rgba(248,81,73,0.30); }"
               ".pill.yellow { background: rgba(212,153,0,0.18); "
                 "color:#d49900; border:1px solid rgba(212,153,0,0.35); }"
               ".pill.gray { background: rgba(139,148,158,0.15); "
                 "color:#8b949e; border:1px solid rgba(139,148,158,0.30); }"
               ".pill.action-buy_yes { background: rgba(63,185,80,0.18); "
                 "color:#3fb950; border:1px solid rgba(63,185,80,0.35); }"
               ".pill.action-buy_no  { background: rgba(248,81,73,0.15); "
                 "color:#f85149; border:1px solid rgba(248,81,73,0.30); }"
               ".pill.action-flag_only,.pill.action-hold { background:"
                 " rgba(139,148,158,0.15); color:#8b949e; border:1px solid"
                 " rgba(139,148,158,0.30); }"
               ".rules-question { max-width: 360px; overflow: hidden; "
                 "text-overflow: ellipsis; white-space: nowrap; "
                 "color: #c9d1d9; }"
               ".clause-text { max-width: 420px; color:#8b949e; "
                 "font-size:12px; line-height:1.4; }"
               ".news-link { color:#58a6ff; text-decoration:none; }"
               ".news-link:hover { text-decoration:underline; }"
               ".warn { color:#d49900; font-size:11px; }"
               "</style>")
    out.append("</head><body>")
    out.append("<h1>Kalshi simulation dashboard</h1>")
    out.append(
        "<div class='meta'>"
        f"Loaded {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%SZ')}"
        " · refreshes every 30s · DRY-RUN mode (no real orders)"
        "</div>"
    )

    main_tabs = [
        ("home",      "Home",      "?tab=home"),
        ("watchlist", "Watchlist", f"?tab=watchlist&bot={html.escape(current_bot_key)}"),
        ("models",    "Models",    f"?tab=models&bot={html.escape(current_bot_key)}"),
        ("history",   "History",   "?tab=history"),
    ]
    _render_bot_filter(out, available_bots, current_bot_key,
                       select_id="bot-select-top",
                       include_all_option=True,
                       tab_key="watchlist")
    out.append("<div class='tab-bar'>")
    # Watchlist is the active tab here — Home / History link back to
    # the main dashboard so navigation feels unified.
    for key, label, href in main_tabs:
        is_active = "watchlist" if active_tab not in {"history", "home"} else active_tab
        cls = "tab-pill" + (" tab-pill-active"
                              if key == is_active else "")
        out.append(f"<a class='{cls}' href='{href}'>{html.escape(label)}</a>")
    out.append("</div>")

    if not summary["available"]:
        out.append("<div class='section'><h2>Rules Parser</h2><div class='body'>")
        out.append("<div class='empty'>"
                   "Rules Parser is configured but its database is not "
                   "available on this host yet. Start the "
                   "<code>rules-parser</code> service to populate "
                   "<code>data/rules_intel.db</code>."
                   "</div>")
        out.append("</div></div>")
        out.append(_BOT_SELECT_NAVIGATE_JS)
        out.append("</body></html>")
        return "".join(out)

    _render_summary_cards(out, summary)
    _render_open_positions(out, open_positions)
    _render_recent_signals(out, signals)
    _render_sources(out, sources)
    _render_settled_positions(out, closed)

    out.append(_BOT_SELECT_NAVIGATE_JS)
    out.append("</body></html>")
    return "".join(out)


_BOT_SELECT_NAVIGATE_JS = """<script>
(function () {
  const sel = document.getElementById("bot-select-top");
  if (!sel) return;
  sel.addEventListener("change", function () {
    if (sel.value) window.location.href = sel.value;
  });
})();
</script>"""


# --------------------------------------------------------------------------- #
# Section renderers                                                           #
# --------------------------------------------------------------------------- #

def _render_summary_cards(out: List[str], s: Dict[str, Any]) -> None:
    realised = s["realised_pnl_cents"]
    mark = s["open_mark_pnl_cents"]
    win_rate = None
    decided = s["wins"] + s["losses"]
    if decided > 0:
        win_rate = s["wins"] / decided
    void_suffix = f" · {s['voids']} void" if s["voids"] else ""
    out.append("<div class='row compact'>")
    out.append(f"<div class='card'><div class='label'>Contracts watched</div>"
               f"<div class='value'>{s['contracts']}</div>"
               f"<div class='label'>"
               f"event-based: {s['event_based_contracts']}</div></div>")
    out.append(f"<div class='card'><div class='label'>Signals (24h)</div>"
               f"<div class='value'>{s['signals_24h']}</div>"
               f"<div class='label'>flagged: {s['signals_flagged']}</div></div>")
    out.append(f"<div class='card'><div class='label'>Open sim positions</div>"
               f"<div class='value'>{s['open_positions']}</div>"
               f"<div class='label'>mark: "
               f"<span class='{_pnl_class(mark)}'>"
               f"{_fmt_signed_dollars(mark)}</span></div></div>")
    out.append(f"<div class='card'><div class='label'>Settled simulated P&amp;L</div>"
               f"<div class='value {_pnl_class(realised)}'>"
               f"{_fmt_signed_dollars(realised)}</div>"
               f"<div class='label'>"
               f"win rate: {_fmt_pct(win_rate)} "
               f"({s['wins']}–{s['losses']}{void_suffix})"
               f"</div></div>")
    out.append("</div>")


def _render_open_positions(out: List[str],
                            positions: List[Dict[str, Any]]) -> None:
    out.append("<div class='section'><h2>Current bets — simulated</h2>"
               "<div class='body'>")
    if not positions:
        out.append("<div class='empty'>"
                   "No open simulated positions. "
                   "The Watchlist tab below shows every recent signal — "
                   "ones that clear every trading gate become positions here."
                   "</div></div></div>")
        return
    out.append(
        "<table><thead><tr>"
        "<th>Opened</th><th>Ticker</th><th>Title</th>"
        "<th class='num'>Side</th>"
        "<th class='num' title='Number of simulated contracts purchased'>Qty</th>"
        "<th class='num' title='Entry price in cents = implied probability'>Entry</th>"
        "<th class='num' title='Current mid for the side we hold'>Mark</th>"
        "<th class='num' title='Confidence the rules-arb signal assigned'>Conf</th>"
        "<th title='Outcome the rules engine expects'>Expected</th>"
        "<th class='num' title='Unrealised P&L (mark − entry)'>Unrealised P&amp;L</th>"
        "<th>Notes</th>"
        "</tr></thead><tbody>"
    )
    for p in positions:
        side = (p.get("side") or "").upper()
        side_cls = "side-yes" if side == "YES" else "side-no"
        mark = p.get("mark_price_cents")
        mark_pnl = p.get("mark_pnl_cents")
        warn = _ambiguity_warning(p)
        conf = p.get("confidence")
        conf_cls = _confidence_class(conf)
        warn_html = (f"<span class='warn'>⚠ {html.escape(warn)}</span>"
                     if warn else "")
        expected = (p.get("expected_resolution") or "—").upper()
        title_text = p.get("title") or "—"
        title_attr = p.get("title") or ""
        out.append(
            f"<tr>"
            f"<td>{html.escape(_fmt_ts(p.get('opened_at')))}</td>"
            f"<td class='mono'>{html.escape(p.get('ticker') or '')}</td>"
            f"<td class='rules-question' title='{html.escape(title_attr)}'>"
            f"{html.escape(title_text)}</td>"
            f"<td class='num'><span class='{side_cls}'>{side}</span></td>"
            f"<td class='num'>{p.get('contracts') or 0}</td>"
            f"<td class='num'>{_fmt_cents(p.get('entry_price_cents'))}</td>"
            f"<td class='num'>{_fmt_cents(mark)}</td>"
            f"<td class='num'><span class='pill {conf_cls}'>"
            f"{_fmt_pct(conf)}</span></td>"
            f"<td>{html.escape(expected)}</td>"
            f"<td class='num {_pnl_class(mark_pnl)}'>"
            f"{_fmt_signed_dollars(mark_pnl)}</td>"
            f"<td>{warn_html}</td>"
            f"</tr>"
        )
    out.append("</tbody></table></div></div>")


def _render_recent_signals(out: List[str],
                           signals: List[Dict[str, Any]]) -> None:
    out.append("<div class='section'><h2>Recent signals — rule interpretation</h2>"
               "<div class='body'>")
    out.append("<div class='small gray' style='margin-bottom:8px;'>"
               "Every (news × rule clause) match the scorer produced. "
               "A signal becomes a simulated position when it clears every "
               "trading gate (confidence, source reliability, clause match). "
               "Below-threshold signals stay in <code>monitoring</code> so "
               "they're auditable but never trade.</div>")
    if not signals:
        out.append("<div class='empty'>No signals captured yet.</div>"
                   "</div></div>")
        return
    out.append(
        "<table><thead><tr>"
        "<th>Detected</th><th>Ticker</th><th>Title</th>"
        "<th>Clause</th>"
        "<th class='num' title='Detector confidence — combines source "
        "reliability, clause match strength, and news recency'>Conf</th>"
        "<th class='num' title='How well the news item matched the clause "
        "keywords (0–1)'>Match</th>"
        "<th class='num' title='Historical reliability of the news source "
        "(0–1)'>Src rel</th>"
        "<th title='Outcome the rule logic implies'>Likely</th>"
        "<th title='What the gate would buy if it clears every threshold'>Action</th>"
        "<th>News</th>"
        "<th>Status</th>"
        "<th>Notes</th>"
        "</tr></thead><tbody>"
    )
    for s in signals:
        warn = _ambiguity_warning(s)
        clause_type = (s.get("clause_type") or "—").replace("_", " ")
        action = (s.get("suggested_action") or "—").lower()
        action_cls = f"pill action-{html.escape(action)}"
        action_label = action.replace("_", " ")
        conf = s.get("confidence")
        conf_cls = _confidence_class(conf)
        clause_text_full = s.get("clause_text") or ""
        clause_text_clip = clause_text_full[:140]
        expected = (s.get("expected_resolution") or "—").upper()
        status_text = (s.get("status") or "—").upper()
        title_text = s.get("title") or "—"
        title_attr = s.get("title") or ""
        warn_html = (f"<span class='warn'>⚠ {html.escape(warn)}</span>"
                     if warn else "")
        news_cell = "<span class='gray'>—</span>"
        if s.get("news_url"):
            label = (s.get("source_name") or "source")
            news_cell = (
                f"<a class='news-link' target='_blank' rel='noopener' "
                f"href='{html.escape(s['news_url'])}' "
                f"title='{html.escape(s.get('news_title') or '')}'>"
                f"{html.escape(label)}</a>"
            )
        status_pill_cls = {
            "monitoring": "pill gray",
            "flagged":    "pill yellow",
            "traded":     "pill green",
            "ignored":    "pill red",
        }.get((s.get("status") or "").lower(), "pill gray")
        out.append(
            f"<tr>"
            f"<td>{html.escape(_fmt_ts(s.get('created_at')))}</td>"
            f"<td class='mono'>{html.escape(s.get('ticker') or '')}</td>"
            f"<td class='rules-question' title='{html.escape(title_attr)}'>"
            f"{html.escape(title_text)}</td>"
            f"<td><span class='pill gray'>{html.escape(clause_type)}</span>"
            f"<div class='clause-text' title='{html.escape(clause_text_full)}'>"
            f"{html.escape(clause_text_clip)}</div></td>"
            f"<td class='num'><span class='pill {conf_cls}'>"
            f"{_fmt_pct(conf)}</span></td>"
            f"<td class='num'>{_fmt_pct(s.get('clause_match_score'))}</td>"
            f"<td class='num'>{_fmt_pct(s.get('source_reliability'))}</td>"
            f"<td>{html.escape(expected)}</td>"
            f"<td><span class='{action_cls}'>{html.escape(action_label)}</span></td>"
            f"<td>{news_cell}</td>"
            f"<td><span class='{status_pill_cls}'>"
            f"{html.escape(status_text)}</span></td>"
            f"<td>{warn_html}</td>"
            f"</tr>"
        )
    out.append("</tbody></table></div></div>")


def _render_sources(out: List[str], sources: List[Dict[str, Any]]) -> None:
    out.append("<div class='section'><h2>News sources — reliability</h2>"
               "<div class='body'>")
    if not sources:
        out.append("<div class='empty'>No sources configured yet — "
                   "add some to <code>config/config.yaml</code>.</div>"
                   "</div></div>")
        return
    out.append(
        "<table><thead><tr>"
        "<th>Source</th><th>Kind</th>"
        "<th class='num' title='Reliability score, updated from observed "
        "signal accuracy. 0.5 = neutral seed.'>Reliability</th>"
        "<th class='num'>Fetches OK</th>"
        "<th class='num'>Fetch errors</th>"
        "<th>Last fetched</th>"
        "<th>Last error</th>"
        "</tr></thead><tbody>"
    )
    for s in sources:
        rel = s.get("reliability")
        rel_cls = _confidence_class(rel)
        out.append(
            f"<tr>"
            f"<td>{html.escape(s.get('name') or '')}</td>"
            f"<td><span class='pill gray'>{html.escape(s.get('kind') or '')}</span></td>"
            f"<td class='num'><span class='pill {rel_cls}'>"
            f"{_fmt_pct(rel)}</span></td>"
            f"<td class='num'>{s.get('success_count') or 0}</td>"
            f"<td class='num'>{s.get('error_count') or 0}</td>"
            f"<td>{html.escape(_fmt_ts(s.get('last_fetched_at')))}</td>"
            f"<td class='gray small'>{html.escape((s.get('last_error') or '')[:120])}</td>"
            f"</tr>"
        )
    out.append("</tbody></table></div></div>")


def _render_settled_positions(out: List[str],
                              positions: List[Dict[str, Any]]) -> None:
    out.append("<div class='section'><h2>Settled simulated trades</h2>"
               "<div class='body'>")
    if not positions:
        out.append("<div class='empty'>No settled simulated trades yet — "
                   "positions remain open until their underlying contract "
                   "resolves.</div></div></div>")
        return
    pnl_total = sum((p.get("pnl_cents") or 0) for p in positions
                     if p.get("pnl_cents") is not None)
    out.append("<div class='small gray' style='margin-bottom:8px;'>"
               f"Realised P&amp;L across the last {len(positions)} "
               f"settled trades: "
               f"<span class='{_pnl_class(pnl_total)}'>"
               f"{_fmt_signed_dollars(pnl_total)}</span></div>")
    out.append(
        "<table><thead><tr>"
        "<th>Settled</th><th>Ticker</th><th>Title</th>"
        "<th class='num'>Side</th><th class='num'>Qty</th>"
        "<th class='num'>Entry</th>"
        "<th title='What the rules engine expected at entry'>Expected</th>"
        "<th title='Actual settled outcome'>Outcome</th>"
        "<th class='num'>Realised P&amp;L</th>"
        "</tr></thead><tbody>"
    )
    for p in positions:
        side = (p.get("side") or "").upper()
        side_cls = "side-yes" if side == "YES" else "side-no"
        outcome = (p.get("settled_outcome") or "—").upper()
        if outcome == "VOID":
            outcome_pill = "<span class='pill gray'>VOID</span>"
        elif outcome == (p.get("expected_resolution") or "").upper():
            outcome_pill = f"<span class='pill green'>{outcome}</span>"
        else:
            outcome_pill = f"<span class='pill red'>{outcome}</span>"
        pnl = p.get("pnl_cents")
        out.append(
            f"<tr>"
            f"<td>{html.escape(_fmt_ts(p.get('settled_at')))}</td>"
            f"<td class='mono'>{html.escape(p.get('ticker') or '')}</td>"
            f"<td class='rules-question' title='{html.escape(p.get('title') or '')}'>"
            f"{html.escape(p.get('title') or '—')}</td>"
            f"<td class='num'><span class='{side_cls}'>{side}</span></td>"
            f"<td class='num'>{p.get('contracts') or 0}</td>"
            f"<td class='num'>{_fmt_cents(p.get('entry_price_cents'))}</td>"
            f"<td>{html.escape((p.get('expected_resolution') or '—').upper())}</td>"
            f"<td>{outcome_pill}</td>"
            f"<td class='num {_pnl_class(pnl)}'>"
            f"{_fmt_signed_dollars(pnl)}</td>"
            f"</tr>"
        )
    out.append("</tbody></table></div></div>")
