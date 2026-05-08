"""Tennis-forecast (Baseline Break) dashboard view.

Different shape than the gas-bot-style page:

  - Source is JSON (the watchlist file written by the tennis-forecast
    project's ``src/dashboard/export_watchlist.py``), not SQLite.
  - There are no Kalshi tickers / strikes / hedges. The "watchlist"
    is one row per upcoming-or-live tennis match, with the model's
    pre-match probability, the live-adjusted probability, the
    market-implied probability, and the resulting edge / EV / signal.
  - The home tab surfaces the *model* (accuracy / Brier / log-loss /
    logistic coefficients + top GBT features) instead of bot equity
    and Kalshi candlesticks.

Reads are cheap (small JSON files) and best-effort — missing files
render the empty state instead of raising.

Two tabs ("home" / "watchlist") mirror the rest of the dashboard so
users can hop between bots without learning a second navigation idiom.
"""
from __future__ import annotations

import html
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

log = logging.getLogger("dashboard.tennis")

TENNIS_TABS = [("home", "Home"), ("watchlist", "Watchlist")]

# Color palette matching the existing dashboard signal-pill aesthetic.
_LABEL_COLORS = {
    "STRONG_EDGE":          ("#3fb950", "tradeable"),
    "SMALL_EDGE":           ("#56d364", "tradeable"),
    "MARKET_OVERREACTION":  ("#e3b341", "tradeable"),
    "WATCH":                ("#58a6ff", "monitor"),
    "AVOID_VOLATILE":       ("#d29922", "skip"),
    "INJURY_RISK":          ("#f85149", "skip"),
    "NO_TRADE":             ("#8b949e", "skip"),
}


# --------------------------------------------------------------------------- #
# Data loaders                                                                #
# --------------------------------------------------------------------------- #

def load_watchlist(path: str | None) -> Dict[str, Any]:
    """Read ``watchlist.json`` written by the tennis-forecast pipeline.

    Returns ``{"generated_at": ..., "rows": [...]}``. Empty payload on
    missing/corrupt files — callers render the empty state.
    """
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
    """Read training metrics. Tolerates missing file."""
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
    file (returns an empty-state stub so the renderer shows a clean
    'no positions yet' panel)."""
    empty = {"open_positions": [], "closed_positions": [],
             "stats": {"open_count": 0, "total_closed": 0, "wins": 0,
                       "losses": 0, "total_realized_pnl": 0.0,
                       "total_unrealized_pnl": 0.0, "total_staked": 0.0,
                       "win_rate": None, "roi": None}}
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
    # Backfill any missing fields.
    for k, v in empty.items():
        data.setdefault(k, v)
    for k, v in empty["stats"].items():
        data["stats"].setdefault(k, v)
    return data


def summary_for_rollup(sim_state_path: str | None) -> Dict[str, Any]:
    """Tennis summary in the shape the cross-bot rollup expects.

    The trading dashboard's home-page summary cards aggregate
    ``open_count``, ``period_bets_made``, ``period_net_pnl_cents``,
    ``period_wins``, ``period_losses``, ``period_money_spent_cents``,
    ``period_money_gained_cents``, ``potential_gain_cents`` across
    bots. We map the tennis sim state into those names so the user
    sees a single set of numbers.

    Cents conversion: tennis stake is in dollars (1.0 = $1); multiply
    by 100 to match the rest of the dashboard.
    """
    s = load_sim_state(sim_state_path)
    stats = s.get("stats") or {}
    open_positions = s.get("open_positions") or []
    closed = s.get("closed_positions") or []

    money_spent_cents = int(round(sum(
        float(c.get("stake", 0)) * 100.0 for c in closed
    )))
    # Money "gained" = stake when we won, 0 when we lost. (Realized P&L
    # = money_gained - money_spent.) We don't have a per-bet payout
    # field on closed positions; reconstruct from won + realized_pnl.
    money_gained_cents = 0
    for c in closed:
        stake = float(c.get("stake", 0))
        pnl = float(c.get("realized_pnl", 0))
        money_gained_cents += int(round((stake + pnl) * 100.0))
    realized_pnl_cents = money_gained_cents - money_spent_cents
    # Potential gain on the open book: sum of (1 - entry) * stake, the
    # max additional payoff if all open positions resolve our way.
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


def model_summary_for_card(metrics_path: str | None) -> Dict[str, Any]:
    """Return a dict shaped like ``fetch_latest_model``'s output so the
    cross-bot card grid on the standard dashboards can render the
    tennis bot without a code branch.

    Mapping:
      classifier_accuracy ← metrics.blended.accuracy
      training_f1         ← computed from accuracy as a stand-in (we
                             don't compute F1 in the tennis pipeline;
                             the card cell shows "—" if missing)
      feature_count       ← length of the pre-match feature list

    Tennis ROI / win-rate are rendered on the tennis page itself.
    """
    metrics = load_metrics(metrics_path)
    if not metrics:
        return {}
    blended = metrics.get("blended") or metrics.get("ensemble") or {}
    return {
        "classifier_accuracy": blended.get("accuracy"),
        "training_brier": blended.get("brier"),
        "training_log_loss": blended.get("log_loss"),
        "feature_count": 12,  # see PREMATCH_FEATURES in the tennis repo
        "actual_wins": 0,
        "actual_losses": 0,
        # Empty values for the standard card cells we don't compute —
        # keeps the renderer happy without lying.
        "training_f1": None,
        "training_precision": None,
        "training_recall": None,
        "training_roc_auc": None,
    }


# --------------------------------------------------------------------------- #
# Rendering                                                                   #
# --------------------------------------------------------------------------- #

def _fmt_pct(v, decimals: int = 1) -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v) * 100:.{decimals}f}%"
    except (TypeError, ValueError):
        return "—"


def _fmt_signed_pp(v) -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v) * 100:+.1f}pp"
    except (TypeError, ValueError):
        return "—"


def _fmt_signed_ev(v) -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v):+.3f}"
    except (TypeError, ValueError):
        return "—"


def _label_pill(label: str) -> str:
    color, _bucket = _LABEL_COLORS.get(label, ("#8b949e", ""))
    return (
        f"<span class='pill' style='background:{color}22;color:{color};"
        f"border:1px solid {color}55'>{html.escape(label)}</span>"
    )


def _last_updated_age(generated_at: str | None) -> str:
    if not generated_at:
        return "never"
    try:
        ts = datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
        delta = (datetime.now(timezone.utc) - ts).total_seconds()
        if delta < 60:
            return f"{int(delta)}s ago"
        if delta < 3600:
            return f"{int(delta // 60)}m {int(delta % 60)}s ago"
        return f"{int(delta // 3600)}h {int((delta % 3600) // 60)}m ago"
    except (TypeError, ValueError):
        return "—"


def _summary_stats(rows: List[dict]) -> Dict[str, Any]:
    if not rows:
        return {
            "total": 0, "live": 0, "actionable": 0, "skip": 0,
            "avg_confidence": 0.0, "max_edge_pp": 0.0,
            "best_match": "—",
        }
    total = len(rows)
    actionable = sum(1 for r in rows if r.get("recommended_action") in
                     ("STRONG_EDGE", "SMALL_EDGE", "MARKET_OVERREACTION"))
    skip = sum(1 for r in rows if r.get("recommended_action") in
               ("INJURY_RISK", "AVOID_VOLATILE", "NO_TRADE"))
    live = sum(1 for r in rows
               if (r.get("current_score") or "0-0") not in ("0-0", "—"))
    confs = [float(r.get("confidence_score") or 0) for r in rows]
    edges = [abs(float(r.get("edge_a") or 0)) for r in rows]
    best = max(rows, key=lambda r: abs(float(r.get("edge_a") or 0)))
    return {
        "total": total, "live": live, "actionable": actionable, "skip": skip,
        "avg_confidence": sum(confs) / max(1, len(confs)),
        "max_edge_pp": max(edges) * 100 if edges else 0.0,
        "best_match": f"{best.get('player_a','?')} vs {best.get('player_b','?')}",
    }


def _signed_dollars(cents: float | int) -> str:
    sign = "+" if cents >= 0 else "−"
    return f"{sign}${abs(cents) / 100:.2f}"


def render_simulation_section(sim_state: dict) -> str:
    """Open + closed paper positions panel — the actual paper-trading
    ledger for the tennis bot. Same idiom as the active-bets table on
    the gas/jobless pages.
    """
    stats = sim_state.get("stats") or {}
    open_positions = sim_state.get("open_positions") or []
    closed = list(sim_state.get("closed_positions") or [])
    closed.sort(key=lambda c: c.get("closed_at", ""), reverse=True)

    out: List[str] = []
    out.append("<div class='card' style='margin-top:18px'>")
    out.append("<h3 class='section-title'>Simulation · paper trades</h3>")
    out.append("<p class='small gray'>$1 unit stake. Position opens "
               "when a match's signal label is STRONG_EDGE, SMALL_EDGE, "
               "or MARKET_OVERREACTION; mark-to-market every tick; "
               "settles when the match completes. Slippage is the "
               "configured per-trade cost (half-spread + book-walk).</p>")

    # Headline numbers.
    win_rate = stats.get("win_rate")
    roi = stats.get("roi")
    realized = float(stats.get("total_realized_pnl") or 0.0)
    unrealized = float(stats.get("total_unrealized_pnl") or 0.0)
    out.append("<div class='row compact'>")
    out.append(_stat("Open positions", str(len(open_positions))))
    out.append(_stat("Closed positions", str(len(closed))))
    pnl_cls = "green" if realized > 0 else ("red" if realized < 0 else "")
    out.append(_stat(
        "Realized P&L", f"{realized:+.2f}", cls=pnl_cls,
        small="includes slippage",
    ))
    u_cls = "green" if unrealized > 0 else ("red" if unrealized < 0 else "")
    out.append(_stat("Unrealized P&L", f"{unrealized:+.2f}", cls=u_cls))
    out.append(_stat(
        "Win rate",
        "—" if win_rate is None else f"{win_rate * 100:.0f}%",
    ))
    out.append(_stat(
        "ROI",
        "—" if roi is None else f"{roi * 100:+.1f}%",
    ))
    out.append("</div>")

    # Open positions table.
    out.append("<h4 class='subsection-title' style='margin-top:18px'>"
               "Open positions</h4>")
    if not open_positions:
        out.append("<div class='empty'>No open positions right now.</div>")
    else:
        out.append("<div style='overflow-x:auto'>"
                   "<table class='watchlist-table'><thead><tr>"
                   "<th>Match</th><th>Side</th><th>Entry</th>"
                   "<th>Mark</th><th>Live model</th>"
                   "<th>Unrealized</th><th>Label</th><th>Opened</th>"
                   "</tr></thead><tbody>")
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
        out.append("</tbody></table></div>")

    # Closed positions table — most recent first, capped to 25 rows so
    # the page stays reasonable. The full ledger is in sim_state.json.
    out.append("<h4 class='subsection-title' style='margin-top:18px'>"
               "Recent closed positions</h4>")
    if not closed:
        out.append("<div class='empty'>No settled positions yet — "
                   "wait for a match to complete.</div>")
    else:
        out.append("<div style='overflow-x:auto'>"
                   "<table class='watchlist-table'><thead><tr>"
                   "<th>Match</th><th>Side</th><th>Entry</th>"
                   "<th>Result</th><th>Realized P&L</th>"
                   "<th>Closed</th>"
                   "</tr></thead><tbody>")
        for c in closed[:25]:
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
        out.append("</tbody></table></div>")

    out.append("</div>")  # /card
    return "".join(out)


def render_home(metrics: dict, coefficients: dict, watchlist_payload: dict,
                sim_state: dict | None = None) -> str:
    """Tennis "home" tab — model card + headline numbers.

    The model card shows the metrics that matter for a probability
    forecast (accuracy, Brier, log-loss) plus the logistic regression
    coefficients used by the Elo-only baseline. We deliberately don't
    show the GBT's raw weights because they're not interpretable; the
    top-N feature importances stand in.
    """
    rows = watchlist_payload.get("rows") or []
    stats = _summary_stats(rows)
    out: List[str] = ["<div class='tennis-home'>"]

    # Headline cards — same idiom as the gas/jobless dashboards.
    out.append("<div class='row compact'>")
    out.append(_stat("Matches tracked", str(stats["total"])))
    out.append(_stat("Live right now", str(stats["live"])))
    out.append(_stat("Actionable signals", str(stats["actionable"]),
                     cls="green" if stats["actionable"] else ""))
    out.append(_stat("Largest edge", f"{stats['max_edge_pp']:.1f}pp"))
    out.append(_stat("Avg confidence", f"{stats['avg_confidence']*100:.0f}%"))
    out.append("</div>")

    # Model card.
    out.append("<div class='card' style='margin-top:18px'>")
    out.append("<h3 class='section-title'>Model card · Baseline Break</h3>")
    out.append("<p class='small gray'>Pre-match probability blends a logistic "
               "regression on Elo (overall + surface) with a calibrated "
               "boosted ensemble. Live adjustment is a transparent rules layer "
               "(score-state, serve %, momentum, tiebreak / decider / medical "
               "flags). Signals only fire when model and market disagree by "
               "more than the configured edge floor.</p>")

    blended = metrics.get("blended") or {}
    elo_only = metrics.get("elo_only") or {}
    ens = metrics.get("ensemble") or {}

    out.append("<div class='row compact' style='margin-top:8px'>")
    out.append(_stat("Accuracy", _fmt_pct(blended.get("accuracy"), 2),
                     small="held-out 12 months"))
    out.append(_stat("Brier score", f"{blended.get('brier', 0):.3f}",
                     small="lower is better"))
    out.append(_stat("Log loss", f"{blended.get('log_loss', 0):.3f}"))
    out.append(_stat("Hold-out rows", f"{int(metrics.get('rows_test', 0)):,}"))
    out.append(_stat("Train rows", f"{int(metrics.get('rows_train', 0)):,}"))
    out.append("</div>")

    # Component metrics: which sub-model contributes which lift?
    out.append("<h4 class='subsection-title' style='margin-top:18px'>"
               "Component breakdown</h4>")
    out.append("<table class='kv-table'><thead><tr>"
               "<th>Component</th><th>Accuracy</th><th>Brier</th><th>Log loss</th>"
               "</tr></thead><tbody>")
    for label, mm in [("Elo-only logistic", elo_only),
                      ("GBT ensemble", ens),
                      ("Blended (live)", blended)]:
        out.append(
            f"<tr><td>{html.escape(label)}</td>"
            f"<td>{_fmt_pct(mm.get('accuracy'), 2)}</td>"
            f"<td>{mm.get('brier', '—'):.3f}</td>"
            f"<td>{mm.get('log_loss', '—'):.3f}</td></tr>"
            if isinstance(mm.get("brier"), (int, float)) else
            f"<tr><td>{html.escape(label)}</td><td>—</td><td>—</td><td>—</td></tr>"
        )
    out.append("</tbody></table>")

    # Logistic coefficients.
    log_coefs = (coefficients.get("logistic") or {})
    feats = log_coefs.get("features") or []
    coefs = log_coefs.get("coefficients") or []
    intercept = log_coefs.get("intercept")
    if feats and coefs:
        out.append("<h4 class='subsection-title' style='margin-top:18px'>"
                   "Model coefficients · Elo-only logistic</h4>")
        out.append("<p class='small gray'>The logistic baseline is fully "
                   "interpretable. A positive coefficient on a player_a-minus-player_b "
                   "feature means: when player_a's value is larger, the model "
                   "raises P(A wins).</p>")
        out.append("<table class='kv-table'><thead><tr>"
                   "<th>Feature</th><th>Coefficient</th><th>Interpretation</th>"
                   "</tr></thead><tbody>")
        for name, c in zip(feats, coefs):
            interp = _coef_interpretation(name, c)
            out.append(
                f"<tr><td><code>{html.escape(name)}</code></td>"
                f"<td>{c:+.4f}</td>"
                f"<td class='small gray'>{interp}</td></tr>"
            )
        if intercept is not None:
            out.append(
                f"<tr><td><code>(intercept)</code></td>"
                f"<td>{intercept:+.4f}</td>"
                f"<td class='small gray'>baseline log-odds for player_a</td></tr>"
            )
        out.append("</tbody></table>")

    # GBT top features (gain importance) — only present when xgboost is installed.
    top_feats = coefficients.get("ensemble_top_features") or []
    if top_feats:
        out.append("<h4 class='subsection-title' style='margin-top:18px'>"
                   "GBT top features · gain importance</h4>")
        out.append("<table class='kv-table'><thead><tr>"
                   "<th>Feature</th><th>Importance</th></tr></thead><tbody>")
        for f in top_feats:
            out.append(
                f"<tr><td><code>{html.escape(f.get('name',''))}</code></td>"
                f"<td>{f.get('importance', 0):.3f}</td></tr>"
            )
        out.append("</tbody></table>")

    # Blend + Elo knobs — surfaces the actual config the prod model uses.
    blend = coefficients.get("blend") or {}
    elo = coefficients.get("elo") or {}
    if blend or elo:
        out.append("<h4 class='subsection-title' style='margin-top:18px'>"
                   "Blend + Elo knobs</h4>")
        out.append("<dl class='kv-dl'>")
        if blend:
            out.append(
                f"<dt>Ensemble weight</dt><dd>{blend.get('ensemble_weight', '—')}</dd>"
                f"<dt>Logistic weight</dt><dd>{blend.get('logistic_weight', '—')}</dd>"
            )
        if elo:
            out.append(
                f"<dt>Elo K (new players)</dt><dd>{elo.get('k_base', '—')}</dd>"
                f"<dt>Elo K (seasoned)</dt><dd>{elo.get('k_floor', '—')}</dd>"
                f"<dt>Surface blend</dt><dd>{int(elo.get('surface_blend', 0)*100)}%</dd>"
            )
        out.append("</dl>")

    out.append("</div>")  # /card

    # --- Simulation panel (paper trades) -------------------------------
    if sim_state is not None:
        out.append(render_simulation_section(sim_state))

    out.append("</div>")  # /tennis-home
    return "".join(out)


def _coef_interpretation(name: str, coef: float) -> str:
    sign = "raises" if coef > 0 else "lowers"
    if name == "diff_elo_pre":
        return f"+1 Elo point on player_a {sign} P(A wins) marginally"
    if name == "diff_surface_elo_pre":
        return f"+1 surface-Elo point on player_a {sign} P(A wins) marginally"
    return f"{sign} P(A wins) per +1 unit"


def render_watchlist(watchlist_payload: dict) -> str:
    rows = watchlist_payload.get("rows") or []
    rows_sorted = sorted(
        rows,
        key=lambda r: (
            -1 if r.get("recommended_action") in
            ("STRONG_EDGE", "MARKET_OVERREACTION") else
            -0.5 if r.get("recommended_action") == "SMALL_EDGE" else 0,
            -abs(float(r.get("edge_a") or 0)),
        ),
    )

    out: List[str] = ["<div class='tennis-watchlist'>"]
    out.append("<div class='card'>")
    age = _last_updated_age(watchlist_payload.get("generated_at"))
    out.append(
        f"<h3 class='section-title'>Watchlist · {len(rows_sorted)} matches</h3>"
        f"<p class='small gray'>Generated {html.escape(age)}. Sorted by signal "
        f"priority then absolute edge. <code>edge</code> is from player_a's "
        f"perspective — positive = model thinks A is undervalued by the "
        f"market.</p>"
    )
    if not rows_sorted:
        out.append("<div class='empty'>No matches yet. Run "
                   "<code>scripts/run_daily_prematch.py</code> on the tennis "
                   "bot to generate a watchlist.</div>")
        out.append("</div></div>")
        return "".join(out)

    out.append("<div style='overflow-x:auto'>")
    out.append("<table class='watchlist-table'>")
    out.append("<thead><tr>"
               "<th>Match</th><th>Tournament</th><th>Surface</th>"
               "<th>Score</th><th>Market</th><th>Pre-match</th>"
               "<th>Live</th><th>Edge</th><th>EV</th>"
               "<th>Conf</th><th>Vol</th><th>Risk</th>"
               "<th>Signal</th>"
               "</tr></thead><tbody>")
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
    out.append("</tbody></table></div>")
    out.append("</div></div>")  # /card /tennis-watchlist
    return "".join(out)


def _stat(label: str, value: str, small: str = "", cls: str = "") -> str:
    cls_attr = f" class='value {cls}'" if cls else " class='value'"
    small_html = f"<div class='small gray'>{html.escape(small)}</div>" if small else ""
    return (f"<div class='card'>"
            f"<div class='label'>{html.escape(label)}</div>"
            f"<div{cls_attr}>{html.escape(value)}</div>"
            f"{small_html}"
            f"</div>")


# --------------------------------------------------------------------------- #
# CSS — appended to the page once. Kept minimal; reuses existing dashboard
# styles (.card, .row.compact, .pill, .green/.red/.gray) so a Tennis page
# inherits the rest of the chrome from the standard renderer.
# --------------------------------------------------------------------------- #

_TENNIS_CSS = """
.tennis-home .row.compact { gap: 12px; }
.tennis-home table.kv-table,
.tennis-watchlist table.watchlist-table {
  width: 100%; border-collapse: collapse; font-size: 13px; margin-top: 6px;
}
.tennis-home table.kv-table th,
.tennis-watchlist table.watchlist-table th {
  text-align: left; color: #8b949e; font-weight: 500; font-size: 11px;
  text-transform: uppercase; letter-spacing: 0.05em;
  border-bottom: 1px solid #30363d; padding: 8px;
}
.tennis-home table.kv-table td,
.tennis-watchlist table.watchlist-table td {
  padding: 8px; border-bottom: 1px solid #21262d; vertical-align: middle;
}
.tennis-watchlist table.watchlist-table tr:hover td { background: #1c222b; }
.tennis-home dl.kv-dl {
  display: grid; grid-template-columns: max-content auto;
  gap: 4px 18px; margin: 6px 0 0 0; font-size: 13px;
}
.tennis-home dl.kv-dl dt { color: #8b949e; }
.tennis-home dl.kv-dl dd { margin: 0; color: #c9d1d9; }
.tennis-home h4.subsection-title {
  margin: 0; font-size: 12px; text-transform: uppercase; color: #8b949e;
  letter-spacing: 0.06em;
}
.tennis-home code, .tennis-watchlist code {
  background: #1d232c; padding: 1px 5px; border-radius: 4px;
  font-size: 12px; color: #c9d1d9;
}
.tennis-watchlist .pill { display: inline-block; padding: 2px 8px;
  border-radius: 12px; font-size: 11px; font-weight: 600; }
"""


def render_page(*, metrics_path: str | None, coefficients_path: str | None,
                watchlist_path: str | None, sim_state_path: str | None = None,
                available_bots: List[dict], current_bot_key: str,
                tab_key: str = "home") -> str:
    """Top-level renderer for the tennis-forecast page.

    Reuses the standard dashboard's chrome (header, bot filter, tab bar)
    by emitting a self-contained HTML document with the same look.
    The dashboard.py dispatcher hands the URL straight to us and uses
    whatever bytes we return.
    """
    metrics = load_metrics(metrics_path)
    coefficients = load_coefficients(coefficients_path)
    payload = load_watchlist(watchlist_path)
    sim_state = load_sim_state(sim_state_path) if sim_state_path else None

    if tab_key == "watchlist":
        body = render_watchlist(payload)
    else:
        body = render_home(metrics, coefficients, payload, sim_state=sim_state)

    return _wrap_shell(body, available_bots, current_bot_key, tab_key)


def _wrap_shell(body: str, available_bots: List[dict], current_bot_key: str,
                tab_key: str) -> str:
    """Header + tab bar + bot filter, all inline. Mirrors the visual
    style of the standard dashboard so the tennis page doesn't feel
    like a different site."""
    bot_filter_html = _render_bot_filter(available_bots, current_bot_key)
    tabs_html = _render_tab_bar(current_bot_key, tab_key)

    return f"""<!doctype html>
<html><head><meta charset="utf-8">
<title>Tennis Forecast · Trading Dashboard</title>
<style>{_BASE_CSS}{_TENNIS_CSS}</style>
</head>
<body>
<header>
  <h1>🎾 Tennis Forecast</h1>
  <div class="header-meta">Baseline Break · pre-match + live adjustment</div>
</header>
<nav class="bot-filter">{bot_filter_html}</nav>
<nav class="tab-bar">{tabs_html}</nav>
<main>{body}</main>
</body></html>"""


def _render_bot_filter(available_bots: List[dict], current_bot_key: str) -> str:
    out = []
    for b in available_bots:
        key = b.get("key", "")
        active = " active" if key == current_bot_key else ""
        out.append(f'<a class="bot-pill{active}" href="?bot={html.escape(key)}">'
                   f'{html.escape(b.get("name", key))}</a>')
    return "".join(out)


def _render_tab_bar(bot_key: str, tab_key: str) -> str:
    out = []
    for k, label in TENNIS_TABS:
        active = " active" if k == tab_key else ""
        out.append(f'<a class="tab-pill{active}" '
                   f'href="?bot={html.escape(bot_key)}&tab={k}">'
                   f'{html.escape(label)}</a>')
    return "".join(out)


# Minimal base CSS — pulled in only because the dispatcher passes raw HTML
# back without sharing the standard renderer's CSS. Stays close to the
# visual idioms of the rest of the dashboard.
_BASE_CSS = """
* { box-sizing: border-box; }
body { margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI",
       Roboto, Helvetica, Arial, sans-serif; background: #0d1117; color: #c9d1d9; }
header { background: #161b22; border-bottom: 1px solid #30363d;
         padding: 14px 24px; display: flex; align-items: center;
         justify-content: space-between; }
header h1 { font-size: 18px; margin: 0; font-weight: 600; color: #f0f6fc; }
.header-meta { color: #8b949e; font-size: 12px; }
nav.bot-filter, nav.tab-bar { padding: 10px 24px; background: #161b22;
                              border-bottom: 1px solid #30363d; }
nav.bot-filter { padding-bottom: 6px; }
nav.tab-bar { padding-top: 6px; }
.bot-pill, .tab-pill { display: inline-block; padding: 4px 10px;
       border-radius: 12px; color: #8b949e; text-decoration: none;
       font-size: 12px; margin-right: 8px; border: 1px solid transparent; }
.bot-pill:hover, .tab-pill:hover { color: #f0f6fc; border-color: #30363d; }
.bot-pill.active, .tab-pill.active { color: #f0f6fc;
       background: #1d232c; border-color: #30363d; }
main { max-width: 1200px; margin: 0 auto; padding: 18px 24px; }
.card { background: #161b22; border: 1px solid #30363d; border-radius: 8px;
        padding: 16px 18px; margin-bottom: 14px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.4); }
.card .label { font-size: 10px; text-transform: uppercase; color: #8b949e;
               letter-spacing: 0.06em; margin-bottom: 4px; }
.card .value { font-size: 20px; font-weight: 600; color: #f0f6fc; }
.section-title { margin: 0 0 6px 0; font-size: 14px; color: #f0f6fc; }
.row { display: flex; gap: 14px; flex-wrap: wrap; }
.row.compact .card { flex: 1 1 0; min-width: 130px; padding: 12px 14px; }
.row.compact .card .label { font-size: 10px; }
.row.compact .card .value { font-size: 18px; }
.empty { color: #8b949e; padding: 18px; text-align: center; }
.green { color: #3fb950; } .red { color: #f85149; } .gray { color: #8b949e; }
.small { font-size: 11px; }
.pill { display: inline-block; padding: 2px 8px; border-radius: 12px;
        font-size: 11px; font-weight: 600; }
"""
