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
"""


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
    """JSON payload returned by the /api/rules-intel endpoint."""
    return {
        "summary": fetch_summary(db_path),
        "signals": fetch_signals(db_path, status=status_filter, limit=limit),
        "sources": fetch_sources(db_path),
    }
