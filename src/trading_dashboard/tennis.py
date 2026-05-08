"""Tennis-forecast (Baseline Break) dashboard view.

Different shape than the gas-bot-style page:

  - Source is JSON (the watchlist file written by the tennis-forecast
    project's ``src/dashboard/export_watchlist.py``), not SQLite.
  - There are no Kalshi tickers / strikes / hedges. The "watchlist"
    is one row per upcoming-or-live tennis match, with the model's
    pre-match probability, the live-adjusted probability, the
    market-implied probability, and the resulting edge / EV / signal.

Reuses the standard dashboard's CSS + page chrome (title, tab bar,
.section/.body blocks) so the tennis page is visually indistinguishable
from the Kalshi-bot pages — it just renders tennis-shaped data.

Tab structure mirrors the standard renderer's three-tab bar:

  Home      → ``/``        (the cross-bot home — true website home)
  Watchlist → tennis page  (the only tennis-specific tab)
  History   → ``/?tab=history`` (cross-bot history)

So clicking Home or History on the tennis page takes the user out of
the tennis context and into the cross-bot dashboard. The tennis page
itself only renders the watchlist content.
"""
from __future__ import annotations

import html
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

log = logging.getLogger("dashboard.tennis")

_LABEL_COLORS = {
    "STRONG_EDGE":         "#3fb950",
    "SMALL_EDGE":          "#56d364",
    "MARKET_OVERREACTION": "#e3b341",
    "WATCH":               "#58a6ff",
    "AVOID_VOLATILE":      "#d29922",
    "INJURY_RISK":         "#f85149",
    "NO_TRADE":            "#8b949e",
}


# --------------------------------------------------------------------------- #
# Data loaders                                                                #
# --------------------------------------------------------------------------- #

def load_watchlist(path: str | None) -> Dict[str, Any]:
    if not path:
        return {"generated_at": None, "rows": []}
    p = Path(path)
    if not p.exists():
        return {"generated_at": None, "rows": []}
    try:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {"generated_at": None, "rows": []}


def load_metrics(metrics_path: str | None) -> Dict[str, Any]:
    if not metrics_path:
        return {}
    p = Path(metrics_path)
    if not p.exists():
        return {}
    try:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def load_coefficients(coefs_path: str | None) -> Dict[str, Any]:
    if not coefs_path:
        return {}
    p = Path(coefs_path)
    if not p.exists():
        return {}
    try:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def load_sim_state(sim_state_path: str | None) -> Dict[str, Any]:
    """Read the paper-trade simulator's state file. Tolerates missing
    file (returns an empty stub so renderers show 'no positions yet')."""
    empty = {
        "open_positions": [], "closed_positions": [],
        "stats": {"open_count": 0, "total_closed": 0, "wins": 0, "losses": 0,
                   "total_realized_pnl": 0.0, "total_unrealized_pnl": 0.0,
                   "total_staked": 0.0, "win_rate": None, "roi": None},
    }
    if not sim_state_path:
        return empty
    p = Path(sim_state_path)
    if not p.exists():
        return empty
    try:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return empty
    for k, v in empty.items():
        data.setdefault(k, v)
    for k, v in empty["stats"].items():
        data["stats"].setdefault(k, v)
    return data


def model_summary_for_card(metrics_path: str | None,
                            sim_state_path: str | None = None) -> Dict[str, Any]:
    """Return a dict shaped like ``fetch_latest_model``'s output so the
    cross-bot card grid renders the tennis bot with the same eight
    cells as every other card (Accuracy / F1 / Precision / ROC AUC /
    Recall / Features / Actual win % / Gain / loss).
    """
    metrics = load_metrics(metrics_path)
    if not metrics:
        return {}
    blended = metrics.get("blended") or metrics.get("ensemble") or {}
    sim = load_sim_state(sim_state_path) if sim_state_path else {}
    stats = (sim or {}).get("stats") or {}
    return {
        "classifier_accuracy": blended.get("accuracy"),
        "training_brier": blended.get("brier"),
        "training_log_loss": blended.get("log_loss"),
        "training_f1": blended.get("f1"),
        "training_precision": blended.get("precision"),
        "training_recall": blended.get("recall"),
        "training_roc_auc": blended.get("roc_auc"),
        "feature_count": 12,
        "actual_wins": int(stats.get("wins", 0) or 0),
        "actual_losses": int(stats.get("losses", 0) or 0),
    }


def summary_for_rollup(sim_state_path: str | None) -> Dict[str, Any]:
    """Tennis summary in the shape the cross-bot rollup expects.
    Cents conversion: tennis stake is dollars (1.0 = $1) → ×100 for cents.
    """
    s = load_sim_state(sim_state_path)
    stats = s.get("stats") or {}
    open_positions = s.get("open_positions") or []
    closed = s.get("closed_positions") or []
    money_spent_cents = int(round(sum(
        float(c.get("stake", 0)) * 100.0 for c in closed
    )))
    money_gained_cents = 0
    for c in closed:
        stake = float(c.get("stake", 0))
        pnl = float(c.get("realized_pnl", 0))
        money_gained_cents += int(round((stake + pnl) * 100.0))
    realized_pnl_cents = money_gained_cents - money_spent_cents
    potential_gain_cents = int(round(sum(
        (1.0 - float(p.get("entry_market_prob", 0.5))) * float(p.get("stake", 0)) * 100.0
        for p in open_positions
    )))
    return {
        "open_count": len(open_positions),
        "period_bets_made": int(stats.get("total_closed", 0)) + len(open_positions),
        "period_net_pnl_cents": realized_pnl_cents,
        "period_wins": int(stats.get("wins", 0)),
        "period_losses": int(stats.get("losses", 0)),
        "period_money_spent_cents": money_spent_cents,
        "period_money_gained_cents": money_gained_cents,
        "potential_gain_cents": potential_gain_cents,
        "total_bets": int(stats.get("total_closed", 0)) + len(open_positions),
        "realized_pnl_cents": realized_pnl_cents,
        "wins_lifetime": int(stats.get("wins", 0)),
        "losses_lifetime": int(stats.get("losses", 0)),
    }


# --------------------------------------------------------------------------- #
# Formatters                                                                  #
# --------------------------------------------------------------------------- #

def _fmt_pct(v, decimals: int = 1) -> str:
    if v is None: return "—"
    try: return f"{float(v) * 100:.{decimals}f}%"
    except (TypeError, ValueError): return "—"


def _fmt_signed_pp(v) -> str:
    if v is None: return "—"
    try: return f"{float(v) * 100:+.1f}pp"
    except (TypeError, ValueError): return "—"


def _fmt_signed_ev(v) -> str:
    if v is None: return "—"
    try: return f"{float(v):+.3f}"
    except (TypeError, ValueError): return "—"


def _fmt_signed_dollars(cents: float | int) -> str:
    cents = int(round(float(cents)))
    sign = "+" if cents >= 0 else "−"
    return f"{sign}${abs(cents) / 100:.2f}"


def _label_pill(label: str) -> str:
    color = _LABEL_COLORS.get(label, "#8b949e")
    return (f"<span class='pill' style='background:{color}22;color:{color};"
            f"border:1px solid {color}55'>{html.escape(label)}</span>")


def _last_updated_age(generated_at: str | None) -> str:
    if not generated_at:
        return "never"
    try:
        ts = datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
        delta = (datetime.now(timezone.utc) - ts).total_seconds()
        if delta < 60: return f"{int(delta)}s ago"
        if delta < 3600: return f"{int(delta // 60)}m {int(delta % 60)}s ago"
        return f"{int(delta // 3600)}h {int((delta % 3600) // 60)}m ago"
    except (TypeError, ValueError):
        return "—"


def _summary_stats(rows: List[dict]) -> Dict[str, Any]:
    if not rows:
        return {"total": 0, "live": 0, "actionable": 0, "avg_confidence": 0.0,
                "max_edge_pp": 0.0}
    total = len(rows)
    actionable = sum(1 for r in rows if r.get("recommended_action") in
                     ("STRONG_EDGE", "SMALL_EDGE", "MARKET_OVERREACTION"))
    live = sum(1 for r in rows
               if (r.get("current_score") or "0-0") not in ("0-0", "—"))
    confs = [float(r.get("confidence_score") or 0) for r in rows]
    edges = [abs(float(r.get("edge_a") or 0)) for r in rows]
    return {"total": total, "live": live, "actionable": actionable,
            "avg_confidence": sum(confs) / max(1, len(confs)),
            "max_edge_pp": max(edges) * 100 if edges else 0.0}


# --------------------------------------------------------------------------- #
# Sections                                                                    #
# --------------------------------------------------------------------------- #

def _render_tab_bar(active: str = "watchlist") -> str:
    """Three-tab bar matching the standard renderer's chrome.

    Home → / (cross-bot home), Watchlist → tennis (active),
    History → /?tab=history (cross-bot history). All three tabs are
    full-page navigations because the tennis page is rendered by a
    different code path than the standard one — we don't try to share
    the standard renderer's JS panel-toggling here.
    """
    tabs = [
        ("home", "Home", "/"),
        ("watchlist", "Watchlist", "?bot=tennis&tab=watchlist"),
        ("history", "History", "/?tab=history"),
    ]
    out = ["<div class='tab-bar'>"]
    for k, label, href in tabs:
        cls = "tab-pill" + (" tab-pill-active" if k == active else "")
        out.append(
            f"<a class='{cls}' data-tab='{html.escape(k)}' "
            f"href='{html.escape(href)}'>{html.escape(label)}</a>"
        )
    out.append("</div>")
    return "".join(out)


def _render_bot_dropdown(available_bots: List[dict], current_bot_key: str
                          ) -> str:
    """Same bot-filter dropdown the standard renderer uses."""
    if not available_bots:
        return ""
    out = ["<div class='bot-filter-bar'>",
           "<label class='filter-label' for='tennis-bot-select'>Bot</label>",
           "<select id='tennis-bot-select' class='bot-select' "
           "onchange='if(this.value)window.location=this.value'>"]
    for b in available_bots:
        key = b.get("key", "")
        name = b.get("name", key)
        sel = " selected" if key == current_bot_key else ""
        out.append(
            f"<option value='?bot={html.escape(key)}'{sel}>"
            f"{html.escape(name)}</option>"
        )
    out.append("</select></div>")
    return "".join(out)


def _render_current_prediction(metrics: dict, sim_state: dict) -> str:
    """Tennis equivalent of the standard 'Current prediction' card row.

    Six small cards, same layout as the gas / NBA / CPI watchlists.
    Surfaces the metrics the user grades the forecaster on (Accuracy /
    F1 / ROC AUC / Brier) plus the live trading state (open paper
    bets / realized P&L).
    """
    blended = metrics.get("blended") or {}
    stats = sim_state.get("stats") or {}
    realized_cents = int(round(float(stats.get("total_realized_pnl") or 0) * 100))
    pnl_cls = ("green" if realized_cents > 0
                else "red" if realized_cents < 0 else "gray")
    win_rate = stats.get("win_rate")
    win_rate_str = "—" if win_rate is None else f"{float(win_rate) * 100:.0f}%"
    cards = [
        ("Accuracy",         _fmt_pct(blended.get("accuracy"), 1), ""),
        ("F1",               _fmt_pct(blended.get("f1"), 1), ""),
        ("ROC AUC",          _fmt_pct(blended.get("roc_auc"), 1), ""),
        ("Brier",            f"{blended.get('brier', 0):.3f}" if blended.get("brier") is not None else "—", "lower better"),
        ("Open paper bets",  str(int(stats.get("open_count") or 0)), ""),
        ("Realized P&L",     _fmt_signed_dollars(realized_cents), pnl_cls),
    ]
    out = ["<div class='row compact'>"]
    for label, value, sub in cards:
        # ``sub`` here doubles as a CSS class for the Realized P&L cell
        # (green / red) and as a small caption for the Brier cell. We
        # detect colors by membership and split the rendering.
        if sub in ("green", "red", "gray"):
            value_cls = f"value {sub}"
            sub_html = ""
        else:
            value_cls = "value"
            sub_html = f"<div class='small gray'>{html.escape(sub)}</div>" if sub else ""
        out.append(f"<div class='card'><div class='label'>{html.escape(label)}</div>"
                   f"<div class='{value_cls}'>{html.escape(value) if isinstance(value, str) else value}</div>"
                   f"{sub_html}</div>")
    out.append("</div>")
    return "".join(out)


def _render_active_paper_bets(sim_state: dict) -> str:
    """Tennis equivalent of the standard 'Active bet' table."""
    open_positions = sim_state.get("open_positions") or []
    if not open_positions:
        return "<div class='empty'>No active paper bets right now.</div>"
    out = ["<table>",
           "<thead><tr>"
           "<th>Match</th><th>Side</th><th>Entry</th>"
           "<th>Mark</th><th>Live model</th>"
           "<th>Unrealized</th><th>Label</th><th>Opened</th>"
           "</tr></thead><tbody>"]
    for p in sorted(open_positions, key=lambda r: r.get("opened_at", ""),
                    reverse=True):
        unr = float(p.get("unrealized_pnl") or 0.0)
        unr_cls = "green" if unr > 0 else ("red" if unr < 0 else "gray")
        out.append(
            "<tr>"
            f"<td><strong>{html.escape(str(p.get('player_a','')))}</strong>"
            f" vs {html.escape(str(p.get('player_b','')))}<br>"
            f"<span class='small gray'>{html.escape(str(p.get('tournament','')))} · "
            f"{html.escape(str(p.get('surface','')))}</span></td>"
            f"<td><strong>{html.escape(str(p.get('side_player','')))}</strong></td>"
            f"<td>{_fmt_pct(p.get('entry_market_prob'), 1)}</td>"
            f"<td>{_fmt_pct(p.get('current_market_prob'), 1)}</td>"
            f"<td>{_fmt_pct(p.get('current_model_prob'), 1)}</td>"
            f"<td class='{unr_cls}'>{unr:+.3f}</td>"
            f"<td>{_label_pill(str(p.get('label_at_open', '')))}</td>"
            f"<td class='small gray'>{html.escape(str(p.get('opened_at',''))[:19])}</td>"
            "</tr>"
        )
    out.append("</tbody></table>")
    return "".join(out)


def _render_watchlist_table(payload: dict) -> str:
    rows = payload.get("rows") or []
    rows_sorted = sorted(
        rows,
        key=lambda r: (
            -1 if r.get("recommended_action") in
            ("STRONG_EDGE", "MARKET_OVERREACTION") else
            -0.5 if r.get("recommended_action") == "SMALL_EDGE" else 0,
            -abs(float(r.get("edge_a") or 0)),
        ),
    )
    if not rows_sorted:
        return ("<div class='empty'>No matches yet — run "
                "<code>scripts/run_daily_prematch.py</code>.</div>")

    out = ["<table>",
           "<thead><tr>"
           "<th>Match</th><th>Tournament</th><th>Surface</th>"
           "<th>Score</th><th>Market</th><th>Pre-match</th>"
           "<th>Live</th><th>Edge</th><th>EV</th>"
           "<th>Conf</th><th>Vol</th><th>Risk</th>"
           "<th>Signal</th>"
           "</tr></thead><tbody>"]
    for r in rows_sorted:
        edge_a = r.get("edge_a")
        edge_cls = ("green" if (edge_a or 0) > 0
                    else "red" if (edge_a or 0) < 0 else "gray")
        match_html = (
            f"<strong>{html.escape(str(r.get('player_a', '')))}</strong>"
            f" vs {html.escape(str(r.get('player_b', '')))}<br>"
            f"<span class='small gray'>"
            f"{html.escape(str(r.get('round_label', '')))}</span>"
        )
        injury_html = ('<span class="red">⚠ injury</span>'
                       if r.get("injury_news_flag") else '—')
        out.append(
            "<tr>"
            f"<td>{match_html}</td>"
            f"<td>{html.escape(str(r.get('tournament', '')))}</td>"
            f"<td>{html.escape(str(r.get('surface', '')))}</td>"
            f"<td>{html.escape(str(r.get('current_score') or '—'))}</td>"
            f"<td>{_fmt_pct(r.get('market_prob_a'))}</td>"
            f"<td>{_fmt_pct(r.get('pre_match_prob_a'))}</td>"
            f"<td>{_fmt_pct(r.get('live_prob_a'))}</td>"
            f"<td class='{edge_cls}'>{_fmt_signed_pp(edge_a)}</td>"
            f"<td>{_fmt_signed_ev(r.get('ev_a'))}</td>"
            f"<td>{_fmt_pct(r.get('confidence_score'))}</td>"
            f"<td>{_fmt_pct(r.get('volatility_score'))}</td>"
            f"<td>{injury_html}</td>"
            f"<td>{_label_pill(str(r.get('recommended_action', 'NO_TRADE')))}<br>"
            f"<span class='small gray'>"
            f"{html.escape(str(r.get('reason_for_signal', '')))}</span></td>"
            "</tr>"
        )
    out.append("</tbody></table>")
    return "".join(out)


def _render_recent_settles(sim_state: dict, limit: int = 25) -> str:
    closed = list(sim_state.get("closed_positions") or [])
    if not closed:
        return "<div class='empty'>No settled paper bets yet — wait for a match to complete.</div>"
    closed.sort(key=lambda c: c.get("closed_at", ""), reverse=True)
    out = ["<table>",
           "<thead><tr>"
           "<th>Match</th><th>Side</th><th>Entry</th>"
           "<th>Result</th><th>Realized P&amp;L</th><th>Closed</th>"
           "</tr></thead><tbody>"]
    for c in closed[:limit]:
        won = c.get("won")
        result_html = ("<span class='green'>WIN</span>" if won
                        else "<span class='red'>LOSS</span>")
        pnl = float(c.get("realized_pnl", 0))
        pnl_cls = "green" if pnl > 0 else ("red" if pnl < 0 else "gray")
        out.append(
            "<tr>"
            f"<td><strong>{html.escape(str(c.get('player_a','')))}</strong>"
            f" vs {html.escape(str(c.get('player_b','')))}<br>"
            f"<span class='small gray'>{html.escape(str(c.get('tournament','')))} · "
            f"{html.escape(str(c.get('surface','')))}</span></td>"
            f"<td><strong>{html.escape(str(c.get('side_player','')))}</strong></td>"
            f"<td>{_fmt_pct(c.get('entry_market_prob'), 1)}</td>"
            f"<td>{result_html}</td>"
            f"<td class='{pnl_cls}'>{pnl:+.3f}</td>"
            f"<td class='small gray'>{html.escape(str(c.get('closed_at',''))[:19])}</td>"
            "</tr>"
        )
    out.append("</tbody></table>")
    return "".join(out)


def _render_model_card_section(metrics: dict, coefficients: dict) -> str:
    """Tennis 'Model card' — equivalent to NBA's 'Contract rules' panel
    at the bottom of the watchlist page. Surfaces the model's
    component breakdown and the (interpretable) logistic coefficients."""
    blended = metrics.get("blended") or {}
    elo_only = metrics.get("elo_only") or {}
    ens = metrics.get("ensemble") or {}

    out: List[str] = []
    out.append("<p class='small gray'>Pre-match probability blends a "
               "logistic regression on Elo (overall + surface) with a "
               "calibrated boosted ensemble. Live adjustment is a "
               "transparent rules layer (score-state, serve %, momentum, "
               "tiebreak / decider / medical flags). Signals only fire "
               "when model and market disagree by more than the configured "
               "edge floor.</p>")
    out.append("<h3 class='subhead'>Component breakdown</h3>")
    out.append("<table><thead><tr>"
               "<th>Component</th><th>Accuracy</th><th>Brier</th>"
               "<th>Log loss</th></tr></thead><tbody>")
    for name, mm in [("Elo-only logistic", elo_only),
                      ("GBT ensemble", ens),
                      ("Blended (live)", blended)]:
        if not isinstance(mm.get("brier"), (int, float)):
            out.append(f"<tr><td>{html.escape(name)}</td><td>—</td><td>—</td><td>—</td></tr>")
            continue
        out.append(
            f"<tr><td>{html.escape(name)}</td>"
            f"<td>{_fmt_pct(mm.get('accuracy'), 1)}</td>"
            f"<td>{mm.get('brier'):.3f}</td>"
            f"<td>{mm.get('log_loss'):.3f}</td></tr>"
        )
    out.append("</tbody></table>")

    log_coefs = coefficients.get("logistic") or {}
    feats = log_coefs.get("features") or []
    coefs = log_coefs.get("coefficients") or []
    intercept = log_coefs.get("intercept")
    if feats and coefs:
        out.append("<h3 class='subhead'>Model coefficients · Elo-only logistic</h3>")
        out.append("<table><thead><tr>"
                   "<th>Feature</th><th>Coefficient</th><th>Interpretation</th>"
                   "</tr></thead><tbody>")
        for n, c in zip(feats, coefs):
            sign = "raises" if c > 0 else "lowers"
            if n == "diff_elo_pre":
                interp = f"+1 Elo point on player_a {sign} P(A wins) marginally"
            elif n == "diff_surface_elo_pre":
                interp = f"+1 surface-Elo point on player_a {sign} P(A wins) marginally"
            else:
                interp = f"{sign} P(A wins) per +1 unit"
            out.append(
                f"<tr><td><code>{html.escape(n)}</code></td>"
                f"<td>{c:+.4f}</td>"
                f"<td class='small gray'>{html.escape(interp)}</td></tr>"
            )
        if intercept is not None:
            out.append(
                f"<tr><td><code>(intercept)</code></td>"
                f"<td>{intercept:+.4f}</td>"
                f"<td class='small gray'>baseline log-odds for player_a</td></tr>"
            )
        out.append("</tbody></table>")
    return "".join(out)


# --------------------------------------------------------------------------- #
# Page renderer                                                                #
# --------------------------------------------------------------------------- #

def render_page(*, metrics_path: str | None, coefficients_path: str | None,
                watchlist_path: str | None, sim_state_path: str | None = None,
                available_bots: List[dict], current_bot_key: str,
                tab_key: str = "watchlist") -> str:
    """Top-level renderer. Returns a complete HTML document using the
    standard dashboard's CSS chrome.

    Only the watchlist tab is rendered here. Home and History tabs
    on the tab bar are explicit links to ``/`` and ``/?tab=history``
    respectively, so clicking them takes the user out of the tennis
    bot context and into the cross-bot dashboard.
    """
    metrics = load_metrics(metrics_path)
    coefficients = load_coefficients(coefficients_path)
    payload = load_watchlist(watchlist_path)
    sim_state = load_sim_state(sim_state_path)

    # Lazy-import the standard CSS so a tennis-only test wouldn't drag
    # the whole dashboard module in.
    from .dashboard import CSS  # type: ignore

    rows = payload.get("rows") or []

    out: List[str] = ["<!doctype html><html><head><meta charset='utf-8'>"]
    out.append("<title>Kalshi simulation dashboard</title>")
    out.append(f"<style>{CSS}</style>")
    out.append("</head><body>")
    out.append("<h1>Kalshi simulation dashboard</h1>")
    out.append(
        f"<div class='meta'>Loaded "
        f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%SZ')}"
        f" · live updates every 60s · DRY-RUN mode (no real orders)</div>"
    )
    out.append(_render_tab_bar(active="watchlist"))

    # ── Watchlist section ────────────────────────────────────────────────
    out.append("<div class='section'><h2>Watchlist — model vs market</h2>"
               "<div class='body'>")
    out.append(_render_bot_dropdown(available_bots, current_bot_key))
    out.append(_render_current_prediction(metrics, sim_state))

    # Active paper bets — same idiom as the standard "Active bet" subhead.
    out.append("<h3 class='subhead'>Active paper bets</h3>")
    out.append(_render_active_paper_bets(sim_state))

    # The watchlist table itself.
    age = _last_updated_age(payload.get("generated_at"))
    out.append(
        f"<h3 class='subhead'>Tennis matches · {len(rows)} "
        f"<span class='small gray'>(generated {html.escape(age)})</span></h3>"
    )
    out.append(_render_watchlist_table(payload))

    # Recent settles — the realized-P&L history, like NBA's contract
    # history but per-match.
    out.append("<h3 class='subhead'>Recent settled paper bets</h3>")
    out.append(_render_recent_settles(sim_state))

    out.append("</div></div>")  # /body /section

    # ── Model card section (bottom of page, like NBA's contract rules) ──
    out.append("<div class='section'><h2>Model card · Baseline Break</h2>"
               "<div class='body'>")
    out.append(_render_model_card_section(metrics, coefficients))
    out.append("</div></div>")

    out.append("</body></html>")
    return "".join(out)
