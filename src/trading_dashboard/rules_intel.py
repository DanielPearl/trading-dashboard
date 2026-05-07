"""Dashboard view for the rules-parser pipeline.

Why a separate module: dashboard.py is already 4k+ lines, and this view
only depends on the rules-parser SQLite DB — keeping it isolated means
the trading dashboard process does NOT import any of the rules-parser
trader code (so it can never accidentally place an order).

The view reads from ``rules_intel_db_path`` in the dashboard YAML
(``rules_intel:`` section) and renders:

    - a header strip with summary counts (contracts watched, signals
      flagged, signals traded, sources, avg reliability)
    - a filter bar (status: all / monitoring / flagged / traded)
    - a table of the 100 most recent signals matching the filter

Every row links to:
    - the contract on Kalshi (``KALSHI_MARKET_URL_FMT``)
    - the news article (the source URL we scraped)
    - a "view rules" modal showing the full rules_primary text
      (clipped if > 4 KB so a giant CFTC-style ruleset doesn't blow up
      the DOM)

The route handler in ``dashboard.py`` calls ``render_section()`` to get
the HTML and ``snapshot()`` to get the JSON payload for /api/rules-intel.
"""
from __future__ import annotations

import html
import json
import logging
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger("dashboard.rules_intel")

# Public Kalshi market page URL. ?ticker is appended.
KALSHI_MARKET_URL_FMT = "https://kalshi.com/markets/{ticker}"


# --------------------------------------------------------------------------- #
# Read helpers — we duplicate the small set of read queries we need so this
# module doesn't import the rules_parser package (which lives in a sibling
# repo and may not be on PYTHONPATH at dashboard render time).
# --------------------------------------------------------------------------- #

def _conn(db_path: str) -> Optional[sqlite3.Connection]:
    if not Path(db_path).exists():
        return None
    try:
        c = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        c.row_factory = sqlite3.Row
        return c
    except sqlite3.Error as e:
        log.warning("rules_intel: failed to open %s read-only: %s",
                    db_path, e)
        return None


def fetch_summary(db_path: str) -> Dict[str, Any]:
    """Return aggregate counts for the header strip."""
    empty = {"contracts": 0, "clauses": 0, "news_items": 0,
             "sources": 0, "avg_reliability": 0.0,
             "signals_total": 0, "signals_monitoring": 0,
             "signals_flagged": 0, "signals_traded": 0,
             "signals_ignored": 0, "available": False}
    c = _conn(db_path)
    if c is None:
        return empty
    out = dict(empty)
    out["available"] = True
    try:
        with closing(c) as cn:
            for col, sql in [
                ("contracts", "SELECT COUNT(*) FROM contracts"),
                ("clauses", "SELECT COUNT(*) FROM rule_clauses"),
                ("news_items", "SELECT COUNT(*) FROM news_items"),
                ("sources", "SELECT COUNT(*) FROM sources"),
                ("signals_total", "SELECT COUNT(*) FROM signals"),
                ("signals_monitoring",
                 "SELECT COUNT(*) FROM signals WHERE status='monitoring'"),
                ("signals_flagged",
                 "SELECT COUNT(*) FROM signals WHERE status='flagged'"),
                ("signals_traded",
                 "SELECT COUNT(*) FROM signals WHERE status='traded'"),
                ("signals_ignored",
                 "SELECT COUNT(*) FROM signals WHERE status='ignored'"),
            ]:
                row = cn.execute(sql).fetchone()
                out[col] = int(row[0] or 0)
            row = cn.execute(
                "SELECT AVG(reliability) FROM sources").fetchone()
            out["avg_reliability"] = float(row[0] or 0.0)
    except sqlite3.Error as e:
        log.warning("rules_intel.fetch_summary failed: %s", e)
    return out


def fetch_signals(db_path: str, *, status: str = "all",
                  limit: int = 100) -> List[Dict[str, Any]]:
    c = _conn(db_path)
    if c is None:
        return []
    where = ""
    params: tuple = ()
    if status and status != "all":
        where = "WHERE s.status = ?"
        params = (status,)
    sql = f"""
        SELECT
            s.id            AS signal_id,
            s.ticker        AS ticker,
            s.confidence    AS confidence,
            s.source_reliability AS source_reliability,
            s.clause_match_score AS clause_match_score,
            s.expected_resolution AS expected_resolution,
            s.suggested_action AS suggested_action,
            s.reasoning     AS reasoning,
            s.yes_price_cents_at_detect AS yes_at_detect,
            s.no_price_cents_at_detect  AS no_at_detect,
            s.status        AS status,
            s.created_at    AS created_at,
            c.title         AS title,
            c.subtitle      AS subtitle,
            c.yes_sub_title AS yes_sub_title,
            c.yes_bid_cents AS yes_bid_cents,
            c.yes_ask_cents AS yes_ask_cents,
            c.no_bid_cents  AS no_bid_cents,
            c.no_ask_cents  AS no_ask_cents,
            c.last_price_cents AS last_price_cents,
            c.event_ticker  AS event_ticker,
            c.series_ticker AS series_ticker,
            c.rules_primary AS rules_primary,
            cl.text         AS clause_text,
            cl.clause_type  AS clause_type,
            n.url           AS news_url,
            n.title         AS news_title,
            n.summary       AS news_summary,
            n.published_at  AS news_published_at,
            src.name        AS source_name
        FROM signals s
        LEFT JOIN contracts c     ON c.ticker = s.ticker
        LEFT JOIN rule_clauses cl ON cl.id = s.clause_id
        LEFT JOIN news_items  n   ON n.id = s.news_item_id
        LEFT JOIN sources    src  ON src.id = n.source_id
        {where}
        ORDER BY s.created_at DESC
        LIMIT ?
    """
    try:
        with closing(c) as cn:
            rows = cn.execute(sql, params + (limit,)).fetchall()
        return [dict(r) for r in rows]
    except sqlite3.Error as e:
        log.warning("rules_intel.fetch_signals failed: %s", e)
        return []


def fetch_watchlist(db_path: str, *, limit: int = 200) -> List[Dict[str, Any]]:
    """One row per contract that meets rules-parser criteria.

    "Meets criteria" = open contract with at least one parsed rules
    clause. The row carries a rollup of clause counts, the most recent
    signal (if any), and the current YES/NO ask so the caller can build
    a discrepancy statement.

    Sort order:
      1. signal-bearing contracts first, by most-recent-signal time
         (so a freshly fired signal is at the top)
      2. then contracts with high-impact clauses (cancellation, injury,
         postponement, force_majeure) on more clauses first
      3. then alphabetical by ticker for stability

    The query is one big JOIN so the dashboard renders this in a single
    DB round-trip; SQLite handles 100s of rows here trivially.
    """
    c = _conn(db_path)
    if c is None:
        return []
    sql = """
    WITH clause_rollup AS (
        SELECT
            ticker,
            COUNT(*)                                                    AS n_clauses,
            SUM(clause_type = 'cancellation')                           AS n_cancel,
            SUM(clause_type = 'postponement')                           AS n_postpone,
            SUM(clause_type = 'injury')                                 AS n_injury,
            SUM(clause_type = 'weather')                                AS n_weather,
            SUM(clause_type = 'force_majeure')                          AS n_force,
            SUM(clause_type = 'government_release')                     AS n_govt,
            SUM(clause_type = 'official_announcement')                  AS n_announce,
            SUM(clause_type = 'tie_void')                               AS n_void,
            SUM(clause_type = 'settlement_source')                      AS n_source,
            SUM(clause_type = 'deadline')                               AS n_deadline,
            -- Risk score: weighted count of clauses where a
            -- triggering event would flip settlement. Includes
            -- low-weight credit for government_release and
            -- settlement_source clauses, since for econ markets
            -- (CPI, jobless, NFP, EIA storage) the official data
            -- release IS the triggering event — they're a legitimate
            -- arb surface even without explicit cancellation language.
            (SUM(clause_type = 'cancellation') * 3
             + SUM(clause_type = 'postponement') * 2
             + SUM(clause_type = 'injury') * 2
             + SUM(clause_type = 'force_majeure') * 3
             + SUM(clause_type = 'weather') * 1
             + SUM(clause_type = 'tie_void') * 2
             + SUM(clause_type = 'government_release') * 2
             + SUM(clause_type = 'settlement_source') * 1
             + SUM(clause_type = 'scoring_rule') * 1)                   AS risk_weight
        FROM rule_clauses
        GROUP BY ticker
    ),
    latest_signal AS (
        -- One row per ticker with the most recent signal's fields.
        SELECT s.*
        FROM signals s
        JOIN (
            SELECT ticker, MAX(created_at) AS mx
            FROM signals GROUP BY ticker
        ) m ON m.ticker = s.ticker AND m.mx = s.created_at
    )
    SELECT
        c.ticker            AS ticker,
        c.event_ticker      AS event_ticker,
        c.series_ticker     AS series_ticker,
        c.title             AS title,
        c.subtitle          AS subtitle,
        c.yes_sub_title     AS yes_sub_title,
        c.status            AS status,
        c.close_time        AS close_time,
        c.yes_bid_cents     AS yes_bid_cents,
        c.yes_ask_cents     AS yes_ask_cents,
        c.no_bid_cents      AS no_bid_cents,
        c.no_ask_cents      AS no_ask_cents,
        c.last_price_cents  AS last_price_cents,
        c.last_seen_at      AS last_seen_at,
        cr.n_clauses        AS n_clauses,
        cr.n_cancel         AS n_cancel,
        cr.n_postpone       AS n_postpone,
        cr.n_injury         AS n_injury,
        cr.n_weather        AS n_weather,
        cr.n_force          AS n_force,
        cr.n_govt           AS n_govt,
        cr.n_announce       AS n_announce,
        cr.n_void           AS n_void,
        cr.n_source         AS n_source,
        cr.n_deadline       AS n_deadline,
        cr.risk_weight      AS risk_weight,
        ls.id               AS signal_id,
        ls.confidence       AS signal_confidence,
        ls.expected_resolution AS signal_expected,
        ls.suggested_action AS signal_action,
        ls.status           AS signal_status,
        ls.created_at       AS signal_created_at,
        ls.reasoning        AS signal_reasoning,
        ls.clause_id        AS signal_clause_id,
        ls.news_item_id     AS signal_news_id
    FROM contracts c
    JOIN clause_rollup cr ON cr.ticker = c.ticker
    LEFT JOIN latest_signal ls ON ls.ticker = c.ticker
    -- Kalshi reports the live-market status as 'active'; 'open' is the
    -- query-string filter value, not the field value. Accept both so
    -- the filter remains correct if Kalshi unifies the namespaces.
    WHERE c.status IN ('active', 'open')
      -- Restrict to event-based contracts (sports games, awards,
      -- elections, launches, court rulings, weather events).
      -- Statistical-release contracts (CPI, jobless, NFP, EIA storage)
      -- settle on schedule on official numbers — there is no news-event
      -- arb surface for the rules-parser bot to act on.
      -- The is_event_based flag is set per-contract by
      -- discovery.is_event_based_market() during ingestion.
      -- We tolerate NULL (older rows pre-migration) by treating
      -- unflagged rows as event-based to avoid hiding the operator's
      -- existing watchlist mid-rollout.
      AND COALESCE(c.is_event_based, 1) = 1
      -- "Meets arbitrage criteria" = at least one directionally
      -- impactful clause (risk_weight >= 1) OR a live signal. Pure
      -- settlement-source / deadline clauses alone don't qualify
      -- since there is no event-surface for the rules-arb thesis.
      AND (cr.risk_weight >= 1 OR ls.id IS NOT NULL)
    ORDER BY
        CASE WHEN ls.id IS NULL THEN 1 ELSE 0 END,
        ls.created_at DESC,
        cr.risk_weight DESC,
        c.ticker
    LIMIT ?
    """
    try:
        with closing(c) as cn:
            rows = cn.execute(sql, (limit,)).fetchall()
        return [dict(r) for r in rows]
    except sqlite3.Error as e:
        log.warning("rules_intel.fetch_watchlist failed: %s", e)
        return []


def fetch_sources(db_path: str) -> List[Dict[str, Any]]:
    c = _conn(db_path)
    if c is None:
        return []
    try:
        with closing(c) as cn:
            return [dict(r) for r in cn.execute(
                "SELECT * FROM sources ORDER BY reliability DESC, name"
            ).fetchall()]
    except sqlite3.Error as e:
        log.warning("rules_intel.fetch_sources failed: %s", e)
        return []


# --------------------------------------------------------------------------- #
# Render
# --------------------------------------------------------------------------- #

# Embedded so the host page's CSS doesn't need a separate stylesheet.
# Keeps the same dark palette as the rest of the dashboard.
RULES_INTEL_CSS = """
.ri-section { padding: 16px 0; }
.ri-meta { color: #8b949e; font-size: 12px; margin-bottom: 12px; }
.ri-cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 8px; margin-bottom: 16px; }
.ri-card { background: #161b22; border: 1px solid #30363d; border-radius: 6px;
    padding: 10px 12px; }
.ri-card .lbl { color: #8b949e; font-size: 11px; text-transform: uppercase;
    letter-spacing: .5px; }
.ri-card .val { color: #f0f6fc; font-size: 20px; font-weight: 600; }
.ri-filter { display: flex; gap: 6px; margin-bottom: 12px; }
.ri-filter a { padding: 4px 10px; border-radius: 12px; background: #21262d;
    border: 1px solid #30363d; color: #c9d1d9; font-size: 12px;
    text-decoration: none; }
.ri-filter a.active { background: #30363d; border-color: #58a6ff; color: #fff; }
.ri-table { width: 100%; border-collapse: collapse; font-size: 12px; }
.ri-table th, .ri-table td { padding: 6px 8px; border-bottom: 1px solid #21262d;
    text-align: left; vertical-align: top; }
.ri-table th { color: #8b949e; font-weight: 500; font-size: 11px;
    text-transform: uppercase; }
.ri-table td.t-ticker { font-family: ui-monospace, monospace; color: #58a6ff; }
.ri-table td.t-conf { font-weight: 600; }
.ri-table td.t-conf.high { color: #3fb950; }
.ri-table td.t-conf.med { color: #d29922; }
.ri-table td.t-conf.low { color: #8b949e; }
.ri-table .pill { display: inline-block; padding: 1px 8px; border-radius: 10px;
    font-size: 10px; text-transform: uppercase; }
.ri-table .pill-monitoring { background: #21262d; color: #8b949e; }
.ri-table .pill-flagged    { background: #4d2d00; color: #ffb74d; }
.ri-table .pill-traded     { background: #033a16; color: #3fb950; }
.ri-table .pill-ignored    { background: #2d2d2d; color: #6e7681; }
.ri-table .pill-yes        { background: #033a16; color: #3fb950; }
.ri-table .pill-no         { background: #4d0c0c; color: #f85149; }
.ri-table .pill-uncertain  { background: #3a3033; color: #d29922; }
.ri-empty { color: #8b949e; font-style: italic; padding: 24px;
    text-align: center; background: #0d1117; border: 1px dashed #30363d;
    border-radius: 6px; }
.ri-disabled { color: #d29922; padding: 16px; background: #21262d;
    border: 1px solid #30363d; border-radius: 6px; }
.ri-rules-modal { max-height: 60vh; overflow-y: auto; white-space: pre-wrap;
    font-family: ui-monospace, monospace; font-size: 12px; }
/* Watchlist-specific bits — borrows shape from whale.py's main table */
.ri-clause-tags { display: flex; flex-wrap: wrap; gap: 4px; }
.ri-tag { display: inline-block; padding: 1px 6px; border-radius: 8px;
    font-size: 10px; background: #21262d; color: #8b949e;
    border: 1px solid #30363d; }
.ri-tag.high { background: #4d2d00; color: #ffb74d; border-color: #5f3700; }
.ri-table td.t-disc { max-width: 360px; }
.ri-disc { color: #c9d1d9; }
.ri-disc.edge-pos { color: #3fb950; }
.ri-disc.edge-neg { color: #f85149; }
.ri-disc.edge-flat { color: #8b949e; }
.ri-disc .edge-pp { font-weight: 600; margin-left: 6px; }
.ri-subhead { color: #c9d1d9; font-size: 13px; font-weight: 600;
    margin: 18px 0 6px 0; }
.ri-sublead { color: #8b949e; font-size: 12px; margin-bottom: 8px; }
"""


# Map clause-type column name -> short display tag. Order matters here:
# tags are emitted in this order for readability across rows. The "high"
# subset gets an amber background to flag the structurally risky clauses.
_CLAUSE_TAGS = [
    ("n_cancel",   "cancel",   True),
    ("n_force",    "force-maj", True),
    ("n_postpone", "postpone", True),
    ("n_injury",   "injury",   True),
    ("n_void",     "void/tie", True),
    ("n_weather",  "weather",  False),
    ("n_govt",     "gov-rel",  False),
    ("n_announce", "announce", False),
    ("n_source",   "src-rule", False),
    ("n_deadline", "deadline", False),
]


def discrepancy(yes_ask: Optional[int], no_ask: Optional[int],
                expected: Optional[str], confidence: Optional[float],
                risk_weight: int, has_signal: bool
                ) -> Dict[str, Any]:
    """Return a small dict the renderer turns into a discrepancy cell.

    Two regimes:

      1) Has signal — compare the signal's directional view to the
         current Kalshi price. Returned ``edge_pp`` is signed:
         positive means OUR view says the chosen side is more likely
         than the market currently prices; negative means already
         priced in. Above ±5pp we render a colored cell.

      2) No signal yet — describe the *passive* watch state instead.
         For high-risk-weight contracts we say the parser is sitting
         on N triggering clauses awaiting an event; for low risk
         we just say "no clauses with directional impact".

    Centralising the logic here means the API endpoint and the HTML
    table render the same wording.
    """
    out: Dict[str, Any] = {
        "edge_pp": None,
        "edge_class": "edge-flat",
        "headline": "—",
        "detail": "",
    }
    if not has_signal or expected is None or confidence is None:
        if risk_weight >= 5:
            out["headline"] = (
                f"High event-surface (risk={risk_weight}) — awaiting "
                "triggering news"
            )
            out["detail"] = (
                "Cancellation / postponement / injury / force-majeure "
                "clauses parsed; a matching news event would flip this "
                "to a directional signal."
            )
        elif risk_weight >= 2:
            out["headline"] = (
                f"Event-driven settlement (risk={risk_weight}) — "
                "tracking source feeds"
            )
            out["detail"] = (
                "Contract settles on an official release (BLS / EIA / "
                "league box-score). Discrepancy is between the next "
                "release print vs the current Kalshi-implied number."
            )
        elif risk_weight >= 1:
            out["headline"] = (
                f"Light clause coverage (risk={risk_weight}) — passive watch"
            )
            out["detail"] = (
                "Only minor settlement clauses parsed; rules-arb "
                "opportunities here are unlikely."
            )
        else:
            out["headline"] = (
                "No directional clauses parsed — outside arb surface"
            )
            out["detail"] = (
                "Pure deadline / settlement-source rules only."
            )
        return out

    # ── Has-signal regime ─────────────────────────────────────────
    if expected == "UNCERTAIN":
        out["headline"] = (
            "Rules flag possible void/postpone — non-directional"
        )
        out["detail"] = (
            "Trader gates this to flag-only; cannot be converted "
            "to a buy in either direction."
        )
        return out

    our_pct = max(0.0, min(1.0, float(confidence))) * 100.0
    if expected == "YES":
        market_pct = yes_ask if yes_ask is not None else None
        side_label = "YES"
    elif expected == "NO":
        market_pct = no_ask if no_ask is not None else None
        side_label = "NO"
    else:
        out["headline"] = f"Unknown signal direction ({expected})"
        return out

    if market_pct is None:
        out["headline"] = (
            f"Signal → {side_label} {our_pct:.0f}% confidence; "
            "market ask unavailable"
        )
        return out

    edge = our_pct - float(market_pct)
    out["edge_pp"] = round(edge, 1)
    if edge > 5:
        out["edge_class"] = "edge-pos"
        out["headline"] = (
            f"Signal puts {side_label} at {our_pct:.0f}% but market "
            f"only {market_pct}¢ — {side_label} underpriced"
        )
    elif edge < -5:
        out["edge_class"] = "edge-neg"
        out["headline"] = (
            f"Signal puts {side_label} at {our_pct:.0f}% — already "
            f"priced in by market ({market_pct}¢ ask)"
        )
    else:
        out["edge_class"] = "edge-flat"
        out["headline"] = (
            f"Signal aligned with market — both ~{market_pct}¢ on {side_label}"
        )
    out["detail"] = (
        f"market_ask={market_pct}¢ on {side_label}; "
        f"signal_confidence={confidence:.2f}; "
        f"edge={edge:+.1f}pp"
    )
    return out


def _conf_bucket(conf: Optional[float]) -> str:
    if conf is None:
        return "low"
    if conf >= 0.85:
        return "high"
    if conf >= 0.6:
        return "med"
    return "low"


def _fmt_cents(c: Optional[int]) -> str:
    if c is None:
        return "—"
    return f"{int(c)}¢"


def _fmt_pct(p: Optional[float]) -> str:
    if p is None:
        return "—"
    return f"{p * 100:.0f}%"


def _fmt_ago(ts: str | None) -> str:
    if not ts:
        return "—"
    import datetime as dt
    try:
        t = dt.datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        try:
            t = dt.datetime.strptime(ts[:19], "%Y-%m-%d %H:%M:%S"
                                       ).replace(tzinfo=dt.timezone.utc)
        except ValueError:
            return ts
    if t.tzinfo is None:
        t = t.replace(tzinfo=dt.timezone.utc)
    now = dt.datetime.now(dt.timezone.utc)
    age_s = (now - t).total_seconds()
    if age_s < 60:
        return f"{int(age_s)}s ago"
    if age_s < 3600:
        return f"{int(age_s // 60)}m ago"
    if age_s < 86400:
        return f"{int(age_s // 3600)}h ago"
    return f"{int(age_s // 86400)}d ago"


def _render_rules_watchlist(out: List[str], db_path: str) -> None:
    """Render the "Arbitrage watchlist" table.

    One row per contract that meets the rules-parser arbitrage criteria
    (see fetch_watchlist for the SQL filter). Borrows the whale-watcher
    main-table shape — ticker, question, prices, signal direction, and a
    discrepancy statement that's a full English sentence explaining how
    the parser's view disagrees with the current Kalshi market price.

    The discrepancy column is the headline output of this section: it's
    the user-facing translation of "we have a rules-arb opportunity
    here" into something that's safe to read at a glance. Numbers are
    shown alongside (edge in pp) but the sentence is the load-bearing
    field.
    """
    rows = fetch_watchlist(db_path, limit=200)
    out.append("<h3 class='ri-subhead'>Arbitrage watchlist — contracts meeting "
                "rules-parser criteria</h3>")
    out.append(
        "<div class='ri-sublead'>Open contracts whose parsed rules expose an "
        "event surface (cancellation, postponement, injury, weather, "
        "force-majeure, void clauses) — i.e. the rules permit a settlement "
        "flip if a triggering event lands. Sorted by recency of the last "
        "scored signal, then by clause-risk weight.</div>"
    )
    if not rows:
        out.append(
            "<div class='ri-empty'>No contracts currently meet the "
            "arbitrage criteria. The parser ingests + parses every "
            "watched series each ingest cycle; if you just started the "
            "service this typically populates within ~10 min.</div>"
        )
        return

    out.append("<table class='ri-table'><thead><tr>"
               "<th>Ticker</th>"
               "<th>Title</th>"
               "<th>YES / NO ask</th>"
               "<th>Clauses</th>"
               "<th>Signal</th>"
               "<th>Discrepancy statement</th>"
               "<th>Edge</th>"
               "<th>Last seen</th>"
               "<th>Links</th>"
               "</tr></thead><tbody>")

    for r in rows:
        ticker = r["ticker"] or "—"
        title = (r.get("title") or "")[:90]
        yes_ask = r.get("yes_ask_cents")
        no_ask  = r.get("no_ask_cents")

        # Clause tags — skip zeros, mark high-impact ones with .high.
        tags_html_parts: List[str] = []
        for col, label, is_high in _CLAUSE_TAGS:
            n = int(r.get(col) or 0)
            if n <= 0:
                continue
            cls = "ri-tag high" if is_high else "ri-tag"
            tags_html_parts.append(
                f"<span class='{cls}' title='{n} {label} clause(s)'>"
                f"{html.escape(label)}·{n}</span>"
            )
        tags_html = ("<div class='ri-clause-tags'>"
                     + "".join(tags_html_parts) + "</div>"
                     if tags_html_parts else "<span class='small gray'>—</span>")

        # Signal cell — direction pill + confidence + age. Empty when
        # the contract is on the watchlist but no news has fired a
        # signal yet (the common case for high-risk-but-quiet markets).
        has_signal = r.get("signal_id") is not None
        signal_cell: str
        expected = r.get("signal_expected")
        confidence = r.get("signal_confidence")
        if has_signal and expected:
            pill_cls = {
                "YES": "pill-yes", "NO": "pill-no",
                "UNCERTAIN": "pill-uncertain",
            }.get(expected, "pill-uncertain")
            sig_age = _fmt_ago(r.get("signal_created_at"))
            conf_pct = (f"{confidence * 100:.0f}%" if confidence is not None
                         else "—")
            signal_cell = (
                f"<span class='pill {pill_cls}'>{html.escape(expected)}</span>"
                f" <span class='small gray'>{conf_pct} · {html.escape(sig_age)}"
                f"</span>"
            )
        else:
            signal_cell = "<span class='small gray'>watching — no event yet</span>"

        disc = discrepancy(
            yes_ask=yes_ask, no_ask=no_ask,
            expected=expected, confidence=confidence,
            risk_weight=int(r.get("risk_weight") or 0),
            has_signal=has_signal,
        )
        edge_pp = disc.get("edge_pp")
        edge_cell: str
        if edge_pp is None:
            edge_cell = "—"
        else:
            sign = "+" if edge_pp > 0 else ""
            edge_cell = (
                f"<span class='ri-disc {disc['edge_class']}'>"
                f"<span class='edge-pp'>{sign}{edge_pp:.1f}pp</span></span>"
            )

        kalshi_url = KALSHI_MARKET_URL_FMT.format(ticker=ticker)
        # Rules text for the modal. Same blob the signals table uses.
        rules_blob = ""  # supplied by inline data attr below if needed.

        out.append(
            "<tr>"
            f"<td class='t-ticker'>"
            f"<a href='{html.escape(kalshi_url)}' target='_blank' rel='noopener'>"
            f"{html.escape(ticker)}</a></td>"
            f"<td>{html.escape(title)}</td>"
            f"<td>{_fmt_cents(yes_ask)} / {_fmt_cents(no_ask)}</td>"
            f"<td>{tags_html}</td>"
            f"<td>{signal_cell}</td>"
            f"<td class='t-disc'>"
            f"<div class='ri-disc {disc['edge_class']}' "
            f"title='{html.escape(disc.get('detail') or '')}'>"
            f"{html.escape(disc['headline'])}</div></td>"
            f"<td>{edge_cell}</td>"
            f"<td>{html.escape(_fmt_ago(r.get('last_seen_at')))}</td>"
            f"<td>"
            f"<a href='{html.escape(kalshi_url)}' target='_blank' rel='noopener'>"
            f"kalshi</a>"
            f"</td>"
            "</tr>"
        )
    out.append("</tbody></table>")


def render_section(db_path: str, *, status_filter: str = "all",
                   current_bot: str = "") -> str:
    """Build the HTML for the Rules Intel tab.

    The caller (Handler.do_GET in dashboard.py) wraps this in the
    standard <div class='tab-panel'> shell, so we only emit the inner
    content.
    """
    out: List[str] = []
    out.append(f"<style>{RULES_INTEL_CSS}</style>")
    out.append("<div class='section'><h2>Rules Intelligence — Event Signals</h2>"
               "<div class='body ri-section'>")
    out.append(
        "<div class='ri-meta'>Cross-source rules-arbitrage layer. "
        "News and government releases are matched against parsed "
        "settlement clauses; signals that clear the confidence + "
        "reliability + clause-match thresholds are flagged for review. "
        "<strong>Live trading is disabled by default</strong>; signals "
        "in <span class='pill pill-traded'>traded</span> status with "
        "kind=<code>dryrun</code> in the audit log are simulated only."
        "</div>"
    )

    summary = fetch_summary(db_path)
    if not summary["available"]:
        out.append(
            "<div class='ri-disabled'>Rules-parser DB not found at "
            f"<code>{html.escape(db_path)}</code>. Start the "
            "<code>kalshi-rules-parser</code> service or set "
            "<code>rules_intel.db_path</code> in the dashboard YAML.</div>"
        )
        out.append("</div></div>")
        return "".join(out)

    # ── summary cards ─────────────────────────────────────────────
    out.append("<div class='ri-cards'>")
    cards = [
        ("Contracts watched", summary["contracts"]),
        ("Rule clauses",      summary["clauses"]),
        ("News items",        summary["news_items"]),
        ("Sources",           summary["sources"]),
        ("Avg source reliab.", _fmt_pct(summary["avg_reliability"])),
        ("Signals — flagged", summary["signals_flagged"]),
        ("Signals — traded",  summary["signals_traded"]),
        ("Signals — total",   summary["signals_total"]),
    ]
    for lbl, val in cards:
        out.append(
            f"<div class='ri-card'><div class='lbl'>{html.escape(lbl)}</div>"
            f"<div class='val'>{html.escape(str(val))}</div></div>"
        )
    out.append("</div>")

    # ── Watchlist — every contract that meets criteria ─────────────
    # One row per open contract that has at least one parsed clause.
    # Borrows the whale-watcher table shape: ticker, question, prices,
    # latest signal, and a discrepancy statement comparing the
    # rules+news view to the current Kalshi market price.
    _render_rules_watchlist(out, db_path)

    # ── filter bar ────────────────────────────────────────────────
    out.append("<div class='ri-filter'>")
    for key, label in [
        ("all", "All"),
        ("flagged", "Flagged"),
        ("monitoring", "Monitoring"),
        ("traded", "Traded"),
        ("ignored", "Ignored"),
    ]:
        cls = "active" if key == status_filter else ""
        href = (f"?tab=rules&rstatus={html.escape(key)}"
                + (f"&bot={html.escape(current_bot)}" if current_bot else ""))
        out.append(
            f"<a class='{cls}' href='{href}'>{html.escape(label)}</a>"
        )
    out.append("</div>")

    # ── signals table ─────────────────────────────────────────────
    signals = fetch_signals(db_path, status=status_filter, limit=200)
    if not signals:
        out.append(
            f"<div class='ri-empty'>No signals in status "
            f"<strong>{html.escape(status_filter)}</strong> yet. "
            "Rules-parser runs every "
            "<code>scrape_interval_seconds</code>; first signals "
            "typically appear within ~5 min of starting the service.</div>"
        )
        out.append("</div></div>")
        return "".join(out)

    out.append("<table class='ri-table'>")
    out.append(
        "<thead><tr>"
        "<th>Ticker</th>"
        "<th>Title</th>"
        "<th>YES / NO</th>"
        "<th>Clause</th>"
        "<th>Detected event</th>"
        "<th>Source</th>"
        "<th>Conf.</th>"
        "<th>Impact</th>"
        "<th>Action</th>"
        "<th>Status</th>"
        "<th>When</th>"
        "<th>Links</th>"
        "</tr></thead><tbody>"
    )

    for s in signals:
        ticker = s["ticker"] or "—"
        title = (s["title"] or "")[:80]
        yes_ask = s["yes_ask_cents"] if s.get("yes_ask_cents") is not None else s.get("yes_at_detect")
        no_ask  = s["no_ask_cents"]  if s.get("no_ask_cents")  is not None else s.get("no_at_detect")
        conf = s.get("confidence")
        conf_cls = _conf_bucket(conf)
        clause_type = s.get("clause_type") or "—"
        clause_text = (s.get("clause_text") or "")[:160]
        news_title = (s.get("news_title") or "")[:120]
        source_name = s.get("source_name") or "—"
        impact = s.get("expected_resolution") or "UNCERTAIN"
        impact_pill = {
            "YES": "pill-yes", "NO": "pill-no", "UNCERTAIN": "pill-uncertain",
        }.get(impact, "pill-uncertain")
        action = s.get("suggested_action") or "—"
        status_pill = {
            "monitoring": "pill-monitoring",
            "flagged":    "pill-flagged",
            "traded":     "pill-traded",
            "ignored":    "pill-ignored",
        }.get(s.get("status") or "monitoring", "pill-monitoring")

        kalshi_url = KALSHI_MARKET_URL_FMT.format(ticker=ticker)
        news_url = s.get("news_url") or ""
        rules_text = s.get("rules_primary") or ""
        # Provide the rules text as a data attribute so a small click-
        # handler can expand it inline. Capped at 4 KB to avoid a
        # 200-row table doubling the page size.
        rules_blob = html.escape(rules_text[:4000])

        out.append("<tr>")
        out.append(
            f"<td class='t-ticker'>"
            f"<a href='{html.escape(kalshi_url)}' target='_blank' "
            f"rel='noopener'>{html.escape(ticker)}</a></td>"
        )
        out.append(f"<td>{html.escape(title)}</td>")
        out.append(
            f"<td>{_fmt_cents(yes_ask)} / {_fmt_cents(no_ask)}</td>"
        )
        out.append(
            f"<td><div><strong>{html.escape(clause_type)}</strong></div>"
            f"<div class='small gray'>{html.escape(clause_text)}</div></td>"
        )
        out.append(
            f"<td><div>{html.escape(news_title)}</div>"
            f"<div class='small gray'>{html.escape(_fmt_ago(s.get('news_published_at')))}"
            f"</div></td>"
        )
        out.append(f"<td>{html.escape(source_name)} "
                   f"({_fmt_pct(s.get('source_reliability'))})</td>")
        out.append(
            f"<td class='t-conf {conf_cls}'>{_fmt_pct(conf)}</td>"
        )
        out.append(
            f"<td><span class='pill {impact_pill}'>"
            f"{html.escape(impact)}</span></td>"
        )
        out.append(f"<td>{html.escape(action)}</td>")
        out.append(
            f"<td><span class='pill {status_pill}'>"
            f"{html.escape(s.get('status') or '—')}</span></td>"
        )
        out.append(f"<td>{html.escape(_fmt_ago(s.get('created_at')))}</td>")
        out.append(
            "<td>"
            f"<a href='{html.escape(kalshi_url)}' target='_blank' rel='noopener'>kalshi</a>"
        )
        if news_url:
            out.append(
                f" · <a href='{html.escape(news_url)}' target='_blank' "
                f"rel='noopener'>source</a>"
            )
        out.append(
            f" · <a href='#' class='ri-rules-link' "
            f"data-rules='{rules_blob}' "
            f"data-ticker='{html.escape(ticker)}'>rules</a>"
        )
        out.append("</td></tr>")

    out.append("</tbody></table>")

    # Tiny inline script to toggle a rules-modal. We re-use the
    # criteria-modal scaffolding the dashboard already has at the
    # bottom of every page, so this is just an event hook.
    out.append("""
<script>
(function(){
  document.querySelectorAll('.ri-rules-link').forEach(function(a){
    a.addEventListener('click', function(ev){
      ev.preventDefault();
      var ticker = a.getAttribute('data-ticker') || '';
      var rules = a.getAttribute('data-rules') || '';
      var overlay = document.getElementById('criteria-overlay');
      var modal   = document.getElementById('criteria-modal');
      var head    = document.getElementById('criteria-modal-ticker');
      var body    = document.getElementById('criteria-modal-body');
      if (!modal) return;
      modal.querySelector('h3').textContent = 'Settlement rules';
      if (head) head.textContent = ticker;
      if (body) body.innerHTML =
        "<div class='ri-rules-modal'>" + rules + "</div>";
      overlay.hidden = false; modal.hidden = false;
    });
  });
})();
</script>
""")

    out.append("</div></div>")
    return "".join(out)


def snapshot(db_path: str, *, status_filter: str = "all",
             limit: int = 100) -> Dict[str, Any]:
    """JSON payload returned by the /api/rules-intel endpoint.

    The ``watchlist`` array is one row per contract that meets the
    rules-parser arbitrage criteria (see fetch_watchlist). Each row
    carries an embedded ``discrepancy`` object so a non-HTML consumer
    (slackbot, notebook) doesn't have to reimplement the comparison.
    """
    watchlist = fetch_watchlist(db_path, limit=200)
    for r in watchlist:
        r["discrepancy"] = discrepancy(
            yes_ask=r.get("yes_ask_cents"),
            no_ask=r.get("no_ask_cents"),
            expected=r.get("signal_expected"),
            confidence=r.get("signal_confidence"),
            risk_weight=int(r.get("risk_weight") or 0),
            has_signal=r.get("signal_id") is not None,
        )
    return {
        "summary": fetch_summary(db_path),
        "watchlist": watchlist,
        "signals": fetch_signals(db_path, status=status_filter, limit=limit),
        "sources": fetch_sources(db_path),
    }
