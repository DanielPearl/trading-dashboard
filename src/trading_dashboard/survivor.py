"""Survivor-elimination dashboard view.

Reads watchlist.json + metrics.json + model_coefficients.json written
by the survivor-elimination bot (one row per active contestant per
open Kalshi market). Renders a Survivor-shaped watchlist + Models
page using the same page chrome as the tennis bot.

Bot-config shape (from dashboard.yaml):

    - key: survivor
      name: Survivor Elimination
      dashboard_type: survivor
      series_ticker: KXSURVIVOR
      watchlist_json_path: /root/survivor-elimination/data/outputs/watchlist.json
      metrics_path:        /root/survivor-elimination/data/processed/artifacts/metrics.json
      coefficients_path:   /root/survivor-elimination/data/processed/artifacts/model_coefficients.json
      sim_state_path:      /root/survivor-elimination/data/outputs/sim_state.json

The sim_state file is optional (the bot doesn't currently paper-trade
elimination markets) — the renderer degrades gracefully when it's
missing.
"""
from __future__ import annotations

import html
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

log = logging.getLogger("dashboard.survivor")


_VERDICT_COLORS = {
    "BUY YES": "#3fb950",
    "BUY NO":  "#f85149",
    "WATCH":   "#e3b341",
    "SKIP":    "#8b949e",
}


# --------------------------------------------------------------------------- #
# Loaders                                                                     #
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


def load_metrics(path: str | None) -> Dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    try:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def load_coefficients(path: str | None) -> Dict[str, Any]:
    return load_metrics(path)  # same json-loader shape


def model_summary_for_card(metrics_path: str | None,
                            sim_state_path: str | None = None
                            ) -> Dict[str, Any]:
    """Project metrics.json into the shape the cross-bot card grid
    expects. Same eight cells every other card shows (Accuracy, F1,
    Precision, ROC AUC, Recall, Features, Actual win %, Gain / loss)."""
    metrics = load_metrics(metrics_path)
    if not metrics:
        return {}
    blended = metrics.get("blended") or {}
    return {
        "classifier_accuracy": blended.get("accuracy"),
        "training_brier": blended.get("brier"),
        "training_log_loss": blended.get("log_loss"),
        "training_f1": blended.get("f1"),
        "training_precision": blended.get("precision"),
        "training_recall": blended.get("recall"),
        "training_roc_auc": blended.get("roc_auc"),
        "feature_count": int(metrics.get("feature_count", 0) or 0),
        "actual_wins": 0,
        "actual_losses": 0,
    }


def summary_for_rollup(sim_state_path: str | None) -> Dict[str, Any]:
    """Stub — no paper-trade ledger today, so every count is zero.
    Keeps the cross-bot rollup symmetric without special-casing."""
    return {
        "open_count": 0, "period_bets_made": 0, "period_net_pnl_cents": 0,
        "period_wins": 0, "period_losses": 0, "period_money_spent_cents": 0,
        "period_money_gained_cents": 0, "potential_gain_cents": 0,
        "total_bets": 0, "realized_pnl_cents": 0,
        "wins_lifetime": 0, "losses_lifetime": 0,
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
    try:
        pp = float(v) * 100
        if round(pp, 1) == 0: return "0"
        return f"{pp:+.1f}pp"
    except (TypeError, ValueError):
        return "—"


def _fmt_signed_ev(v) -> str:
    if v is None: return "—"
    try:
        x = float(v)
        if round(x, 3) == 0: return "0"
        return f"{x:+.3f}"
    except (TypeError, ValueError):
        return "—"


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


def _verdict_badge(verdict: str, blockers: List[str]) -> str:
    color = _VERDICT_COLORS.get(verdict, "#8b949e")
    tip = ""
    if blockers:
        tip = f" title='Blockers: {html.escape(', '.join(blockers))}'"
    cls = "badge-yes" if verdict == "BUY YES" else \
          "badge-no" if verdict == "BUY NO" else \
          "badge-hedge" if verdict == "WATCH" else "badge-skip"
    return (f"<span class='badge {cls}'{tip}>{html.escape(verdict)}</span>")


# --------------------------------------------------------------------------- #
# Sections                                                                    #
# --------------------------------------------------------------------------- #

def _render_tab_bar(bot_key: str, active: str = "watchlist") -> str:
    tabs = [
        ("home", "Home", "/"),
        ("watchlist", "Watchlist", f"?bot={bot_key}&tab=watchlist"),
        ("models", "Models", f"?bot={bot_key}&tab=models"),
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


def _render_bot_dropdown(available_bots: List[dict], current_key: str) -> str:
    if not available_bots:
        return ""
    out = ["<div class='bot-filter-bar'>",
           "<label class='filter-label' for='survivor-bot-select'>Bot</label>",
           "<select id='survivor-bot-select' class='bot-select' "
           "onchange='if(this.value)window.location=this.value'>"]
    for b in available_bots:
        key = b.get("key", "")
        name = b.get("name", key)
        sel = " selected" if key == current_key else ""
        out.append(
            f"<option value='?bot={html.escape(key)}'{sel}>"
            f"{html.escape(name)}</option>"
        )
    out.append("</select></div>")
    return "".join(out)


def _render_current_prediction(payload: Dict[str, Any],
                                metrics: Dict[str, Any]) -> str:
    """Six-card row at the top of the page: season / episode / active
    contestants / model accuracy / brier / generated-at."""
    blended = metrics.get("blended") or {}
    cards = [
        ("Season",            str(payload.get("season") or "—"), ""),
        ("Episode",           str(payload.get("current_episode") or "—"), ""),
        ("Active contestants", str(payload.get("active_contestants") or
                                    len(payload.get("rows", []))), ""),
        ("Holdout accuracy",  _fmt_pct(blended.get("accuracy"), 1), ""),
        ("Holdout Brier",     (f"{blended.get('brier'):.3f}"
                                 if blended.get("brier") is not None else "—"),
                              "lower better"),
        ("Updated",           _last_updated_age(payload.get("generated_at")), ""),
    ]
    out = ["<div class='row compact'>"]
    for label, value, sub in cards:
        sub_html = (f"<div class='small gray'>{html.escape(sub)}</div>"
                    if sub else "")
        out.append(
            f"<div class='card'><div class='label'>{html.escape(label)}</div>"
            f"<div class='value'>{html.escape(str(value))}</div>"
            f"{sub_html}</div>"
        )
    out.append("</div>")
    return "".join(out)


def _render_watchlist_table(payload: Dict[str, Any]) -> str:
    """Per-contestant table: Ticker | Contestant | Kalshi % | Model % |
    Edge | EV YES | EV NO | Confidence | Verdict.
    """
    rows = payload.get("rows") or []
    if not rows:
        return ("<div class='empty'>No active Survivor markets right now — "
                "the live monitor will pick up the next episode's markets "
                "as soon as Kalshi opens them.</div>")
    out = ["<div class='watchlist-scroll'>",
           "<table id='survivor-watchlist-table'>",
           "<thead><tr>",
           "<th>Ticker</th>",
           "<th>Contestant</th>",
           "<th title='Kalshi market type. elimination = per-episode boot market (YES = eliminated this episode). season_win = win the whole season (YES = takes the title).'>Type</th>",
           "<th class='num' title='Kalshi YES price for the contestant on the displayed market type.'>Kalshi %</th>",
           "<th class='num' title='Model probability for the same side: for elimination markets this is P(eliminated this episode); for season-winner markets it is a chained P(wins season) derived from the per-episode model.'>Model %</th>",
           "<th class='num' title='Per-episode P(eliminated) from the model — the headline elimination forecast regardless of which Kalshi market type is active.'>Boot P</th>",
           "<th class='num' title='Gap between model and Kalshi (model − market) in percentage points.'>Gap</th>",
           "<th class='num' title='Expected value per $1 contract for YES on the displayed market.'>EV YES</th>",
           "<th class='num' title='Expected value per $1 contract for NO on the displayed market.'>EV NO</th>",
           "<th class='num' title='Model confidence on this row (0..1) — combines edge magnitude with distance from a coin-flip price.'>Confidence</th>",
           "<th>Verdict</th>",
           "</tr></thead><tbody>"]
    for r in rows:
        ticker = r.get("ticker") or r.get("match_id") or ""
        if ticker.upper().startswith("KX"):
            kalshi_url = f"https://kalshi.com/markets/{ticker.lower()}"
            ticker_cell = (f"<a href='{html.escape(kalshi_url)}' "
                            f"target='_blank' rel='noopener noreferrer' "
                            f"class='ticker-link'>{html.escape(ticker)}</a>")
        else:
            ticker_cell = html.escape(str(ticker))
        contestant = r.get("contestant") or "—"
        title = r.get("title") or ""
        contestant_cell = (f"<strong>{html.escape(str(contestant))}</strong>"
                            f"<br><span class='small gray' title='"
                            f"{html.escape(str(title))}'>"
                            f"{html.escape(str(title)[:80])}</span>")
        market_type = r.get("market_type") or "elimination"
        mt_short = "elim" if market_type == "elimination" else \
                   "season" if market_type == "season_win" else "?"
        mt_cell = (f"<span class='small gray' title='"
                    f"{html.escape(market_type)}'>{html.escape(mt_short)}</span>")

        mkt = r.get("market_prob") if r.get("market_prob") is not None \
              else r.get("market_prob_eliminated")
        mdl = r.get("model_prob") if r.get("model_prob") is not None \
              else r.get("model_prob_eliminated")
        edge = r.get("edge")
        ev_yes = r.get("ev_yes")
        ev_no = r.get("ev_no")
        conf = r.get("confidence_score")

        edge_cls = ("green" if edge is not None and edge >= 0.06 else
                    "yellow" if edge is not None and edge > 0 else
                    "red" if edge is not None and edge <= -0.06 else "gray")
        ev_yes_cls = ("green" if ev_yes is not None and ev_yes >= 0.03 else
                       "red" if ev_yes is not None and ev_yes <= 0 else
                       "yellow" if ev_yes is not None else "gray")
        ev_no_cls = ("green" if ev_no is not None and ev_no >= 0.03 else
                      "red" if ev_no is not None and ev_no <= 0 else
                      "yellow" if ev_no is not None else "gray")

        verdict = r.get("verdict") or "SKIP"
        blockers = r.get("buy_blockers") or []
        verdict_pill = _verdict_badge(verdict, blockers)

        row_cls = "survivor-row"
        if verdict == "BUY YES":
            row_cls += " row-bought bought-yes"
        elif verdict == "BUY NO":
            row_cls += " row-bought bought-no"
        elif verdict == "WATCH":
            row_cls += " row-suspect"

        # "Boot P" — the headline per-episode P(eliminated) from the
        # model, regardless of which Kalshi market type is active.
        boot_p = r.get("model_prob_eliminated")
        out.append(
            f"<tr class='{row_cls}'>"
            f"<td class='mono small'>{ticker_cell}</td>"
            f"<td>{contestant_cell}</td>"
            f"<td>{mt_cell}</td>"
            f"<td class='num'>{_fmt_pct(mkt, 0)}</td>"
            f"<td class='num'>{_fmt_pct(mdl, 0)}</td>"
            f"<td class='num'>{_fmt_pct(boot_p, 0)}</td>"
            f"<td class='num {edge_cls}'>{_fmt_signed_pp(edge)}</td>"
            f"<td class='num {ev_yes_cls}'>{_fmt_signed_ev(ev_yes)}</td>"
            f"<td class='num {ev_no_cls}'>{_fmt_signed_ev(ev_no)}</td>"
            f"<td class='num'>{conf if conf is not None else '—'}</td>"
            f"<td>{verdict_pill}</td>"
            f"</tr>"
        )
    out.append("</tbody></table></div>")
    return "".join(out)


def _render_reddit_panel(payload: Dict[str, Any]) -> str:
    """Compact panel showing the per-contestant Reddit signals
    embedded in the watchlist rows. Useful for QA — confirms the
    Reddit ingestion is firing and the boot-prediction regex is
    catching mentions."""
    rows = payload.get("rows") or []
    if not rows:
        return ""
    interesting = sorted(rows,
                          key=lambda r: -(r.get("reddit_boot_pick_count") or 0))[:8]
    if not any((r.get("reddit_boot_pick_count") or 0) for r in interesting):
        return ("<div class='empty small gray'>No Reddit boot-prediction "
                "signal yet — either creds aren't configured or no "
                "post-episode discussion has named a contestant as the "
                "next boot.</div>")
    out = ["<table>",
           "<thead><tr>"
           "<th>Contestant</th>"
           "<th class='num'>Mentions</th>"
           "<th class='num'>Boot picks</th>"
           "<th class='num'>Sentiment</th>"
           "<th class='num'>Target share</th>"
           "</tr></thead><tbody>"]
    for r in interesting:
        sent = r.get("reddit_sentiment")
        sent_cls = ("green" if (sent or 0) > 0.05 else
                    "red" if (sent or 0) < -0.05 else "")
        out.append(
            f"<tr><td><strong>{html.escape(str(r.get('contestant') or ''))}"
            f"</strong></td>"
            f"<td class='num'>{int(r.get('reddit_mention_count') or 0)}</td>"
            f"<td class='num'>{int(r.get('reddit_boot_pick_count') or 0)}</td>"
            f"<td class='num {sent_cls}'>{float(sent or 0):+.2f}</td>"
            f"<td class='num'>{float(r.get('reddit_target_share') or 0):.0%}</td>"
            f"</tr>"
        )
    out.append("</tbody></table>")
    return "".join(out)


def _render_validators_panel(payload: Dict[str, Any]) -> str:
    """Surface validator stats for the current watchlist so the user
    can audit what's blocking buys at a glance."""
    rows = payload.get("rows") or []
    if not rows:
        return ""
    total = len(rows)
    buys = sum(1 for r in rows if r.get("verdict") == "BUY YES"
                                  or r.get("verdict") == "BUY NO")
    watches = sum(1 for r in rows if r.get("verdict") == "WATCH")
    skips = sum(1 for r in rows if r.get("verdict") == "SKIP")
    blocker_counts: Dict[str, int] = {}
    for r in rows:
        for b in (r.get("buy_blockers") or []):
            blocker_counts[b] = blocker_counts.get(b, 0) + 1
    out = [
        "<div class='row compact'>",
        f"<div class='card'><div class='label'>Total rows</div>"
        f"<div class='value'>{total}</div></div>",
        f"<div class='card'><div class='label'>BUY rows</div>"
        f"<div class='value green'>{buys}</div></div>",
        f"<div class='card'><div class='label'>WATCH</div>"
        f"<div class='value yellow'>{watches}</div></div>",
        f"<div class='card'><div class='label'>SKIP</div>"
        f"<div class='value gray'>{skips}</div></div>",
        "</div>",
    ]
    if blocker_counts:
        out.append("<table><thead><tr><th>Validator blocker</th>"
                    "<th class='num'>Rows</th></tr></thead><tbody>")
        for reason, n in sorted(blocker_counts.items(),
                                  key=lambda kv: -kv[1]):
            out.append(
                f"<tr><td>{html.escape(reason)}</td>"
                f"<td class='num'>{n}</td></tr>"
            )
        out.append("</tbody></table>")
    return "".join(out)


def _render_models_section(metrics: Dict[str, Any],
                            coefficients: Dict[str, Any]) -> str:
    blended = metrics.get("blended") or {}
    logistic = metrics.get("logistic") or {}
    gbt = metrics.get("calibrated_gbt") or {}
    out: List[str] = []
    out.append(
        "<p class='small gray'>The elimination model is trained on "
        "every modern-era Survivor boot (seasons 41–49) one row per "
        "active contestant per episode. The dependent variable is "
        "<code>eliminated_this_episode</code>. Features include "
        "season + episode structure, tribe state, on-show signal "
        "(immunity, idols, prior votes / target mentions, visibility "
        "spike, negative-edit score, strategic isolation, prior "
        "challenge performance) and Reddit-derived columns "
        "(mention count, boot-pick count, sentiment, target share). "
        "Two models are trained: an L2-regularised logistic regression "
        "(interpretable) and a calibrated HistGradientBoosting "
        "ensemble. The lower-Brier model is used in production.</p>"
    )
    out.append("<h3 class='subhead'>Component breakdown</h3>")
    out.append("<table><thead><tr>"
               "<th>Component</th><th>Accuracy</th><th>Brier</th>"
               "<th>Log loss</th><th>ROC AUC</th></tr></thead><tbody>")
    for name, mm in [("Logistic regression", logistic),
                      ("Calibrated GBT",     gbt),
                      ("Blended (live)",     blended)]:
        if not mm or mm.get("brier") is None:
            out.append(f"<tr><td>{html.escape(name)}</td>"
                        "<td>—</td><td>—</td><td>—</td><td>—</td></tr>")
            continue
        out.append(
            f"<tr><td>{html.escape(name)}</td>"
            f"<td>{_fmt_pct(mm.get('accuracy'), 1)}</td>"
            f"<td>{mm.get('brier'):.3f}</td>"
            f"<td>{mm.get('log_loss'):.3f}</td>"
            f"<td>{_fmt_pct(mm.get('roc_auc'), 1)}</td></tr>"
        )
    out.append("</tbody></table>")
    out.append(
        f"<p class='small gray'>Train seasons: "
        f"{html.escape(str(metrics.get('train_seasons') or '—'))} · "
        f"Test seasons: "
        f"{html.escape(str(metrics.get('test_seasons') or '—'))} · "
        f"Best model: <code>{html.escape(str(metrics.get('best_model') or '—'))}</code></p>"
    )

    log_coefs = (coefficients.get("logistic") or {})
    feats = log_coefs.get("features") or []
    coefs = log_coefs.get("coefficients") or []
    intercept = log_coefs.get("intercept")
    if feats and coefs:
        out.append("<h3 class='subhead'>Model coefficients · logistic regression</h3>")
        out.append("<table><thead><tr>"
                    "<th>Feature</th><th>Coefficient</th><th>Interpretation</th>"
                    "</tr></thead><tbody>")
        ranked = sorted(zip(feats, coefs),
                         key=lambda fc: -abs(fc[1]))
        for n, c in ranked:
            sign = "raises" if c > 0 else "lowers"
            interp = _coef_interpretation(n, sign)
            out.append(
                f"<tr><td><code>{html.escape(n)}</code></td>"
                f"<td>{c:+.4f}</td>"
                f"<td class='small gray'>{html.escape(interp)}</td></tr>"
            )
        if intercept is not None:
            out.append(
                f"<tr><td><code>(intercept)</code></td>"
                f"<td>{intercept:+.4f}</td>"
                f"<td class='small gray'>baseline log-odds of elimination per episode</td></tr>"
            )
        out.append("</tbody></table>")
    return "".join(out)


def _coef_interpretation(name: str, sign: str) -> str:
    table = {
        "season": f"more recent seasons {sign} elimination probability per episode",
        "episode": f"later episodes {sign} elimination probability",
        "remaining": f"more contestants remaining {sign} per-contestant elimination probability",
        "tribe_size": f"larger tribe size {sign} per-contestant elimination probability",
        "starting_tribe_size": f"larger starting tribe {sign} per-contestant elimination probability",
        "merged": f"post-merge episodes {sign} per-contestant elimination probability",
        "swap_phase": f"swap-phase episodes {sign} elimination probability",
        "episode_share_remaining": f"deep-in-the-season rows {sign} probability",
        "pre_merge_phase": f"pre-merge episodes {sign} probability",
        "is_finale": f"final-episode rows {sign} probability",
        "immunity_won": f"holding individual immunity {sign} probability",
        "tribe_immunity": f"winning tribe immunity {sign} elimination probability",
        "has_idol": f"holding a hidden idol {sign} elimination probability",
        "in_main_alliance": f"being in the dominant alliance {sign} probability",
        "prior_votes_against": f"each prior vote against {sign} probability",
        "times_targeted": f"each prior target mention {sign} probability",
        "confessional_count": f"more confessionals this episode {sign} probability",
        "visibility_score": f"higher rolling visibility {sign} probability",
        "visibility_spike": f"a positive visibility spike {sign} probability",
        "negative_edit_score": f"a stronger negative edit {sign} probability",
        "strategic_isolation": f"being more isolated {sign} probability",
        "prior_perf_score": f"stronger prior challenge performance {sign} probability",
        "reddit_mention_count": f"more Reddit mentions {sign} probability",
        "reddit_boot_pick_count": f"more Reddit boot picks {sign} probability",
        "reddit_sentiment": f"more positive Reddit sentiment {sign} probability",
        "reddit_visibility_score": f"higher Reddit visibility share {sign} probability",
        "reddit_target_share": f"larger share of Reddit boot picks {sign} probability",
    }
    return table.get(name, f"{sign} elimination probability per +1 unit")


# --------------------------------------------------------------------------- #
# Top-level page render                                                       #
# --------------------------------------------------------------------------- #

_PAGE_HEAD = """<!DOCTYPE html>
<html lang='en'><head><meta charset='utf-8'>
<title>Survivor Elimination · Kalshi dashboard</title>
<link rel='icon' type='image/svg+xml' href='/static/favicon.svg'>
<style>
body { background:#0d1117; color:#c9d1d9;
       font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
       margin:0; padding:24px; }
h1 { margin:0 0 12px; font-size:20px; }
h2.subhead { margin:24px 0 8px; font-size:15px; color:#f0f6fc; }
h3.subhead { margin:16px 0 8px; font-size:13px; color:#f0f6fc; }
.section { background:#161b22; border:1px solid #21262d; border-radius:8px;
           padding:14px 18px; margin-bottom:18px; }
.row.compact { display:grid; grid-template-columns:repeat(auto-fit,minmax(140px,1fr));
               gap:10px; }
.card { background:#0d1117; border:1px solid #21262d; border-radius:6px;
        padding:10px 12px; }
.card .label { font-size:11px; color:#8b949e; text-transform:uppercase;
               letter-spacing:0.5px; }
.card .value { font-size:18px; color:#f0f6fc; font-weight:600;
               margin-top:4px; }
.green { color:#3fb950; } .red { color:#f85149; }
.yellow { color:#e3b341; } .gray { color:#8b949e; }
.small { font-size:11px; }
.mono { font-family:"SF Mono",Menlo,Consolas,monospace; font-size:12px; }
.empty { padding:24px; color:#8b949e; text-align:center; }
.tab-bar { display:flex; gap:8px; margin-bottom:16px; }
.tab-pill { padding:6px 12px; background:#161b22; border:1px solid #21262d;
            border-radius:999px; color:#c9d1d9; text-decoration:none;
            font-size:13px; }
.tab-pill-active { background:#1f6feb; color:#fff; border-color:#1f6feb; }
.bot-filter-bar { display:flex; gap:8px; align-items:center; margin-bottom:16px; }
.bot-select { background:#0d1117; color:#c9d1d9; border:1px solid #21262d;
              border-radius:6px; padding:4px 8px; }
.filter-label { font-size:12px; color:#8b949e; }
table { width:100%; border-collapse:collapse; }
th,td { padding:6px 8px; border-bottom:1px solid #21262d; text-align:left;
        font-size:13px; }
th { color:#8b949e; font-weight:500; }
.num { text-align:right; }
.cell-sep { color:#30363d; }
.watchlist-scroll { max-height:600px; overflow:auto; border:1px solid #21262d;
                     border-radius:6px; }
.badge { display:inline-block; padding:2px 8px; border-radius:999px;
         font-size:11px; font-weight:600; }
.badge-yes { background:#3fb95022; color:#3fb950; border:1px solid #3fb95055; }
.badge-no  { background:#f8514922; color:#f85149; border:1px solid #f8514955; }
.badge-hedge { background:#e3b34122; color:#e3b341;
               border:1px solid #e3b34155; }
.badge-skip { background:#8b949e22; color:#8b949e;
              border:1px solid #8b949e55; }
.ticker-link { color:#58a6ff; text-decoration:none; }
.ticker-link:hover { text-decoration:underline; }
code { background:#0d1117; padding:1px 4px; border-radius:3px;
       border:1px solid #21262d; font-size:12px; }
tr.row-bought.bought-yes td { background:#3fb9500c; }
tr.row-bought.bought-no  td { background:#f851490c; }
tr.row-suspect td { background:#e3b3410a; }
</style></head><body>
"""


def render_page(metrics_path: str | None,
                coefficients_path: str | None,
                watchlist_path: str | None,
                sim_state_path: str | None,
                available_bots: List[dict],
                current_bot_key: str,
                tab_key: str = "watchlist") -> str:
    """Render the survivor page for the requested tab.

    Tabs:
      watchlist  — current-episode contestants + EV table
      models     — training metrics + coefficients

    Home and History tabs route through the standard cross-bot renderer.
    """
    payload = load_watchlist(watchlist_path)
    metrics = load_metrics(metrics_path)
    coefficients = load_coefficients(coefficients_path)

    out = [_PAGE_HEAD,
           "<h1>Survivor Elimination</h1>",
           _render_tab_bar(current_bot_key, active=tab_key),
           _render_bot_dropdown(available_bots, current_bot_key)]

    if tab_key == "models":
        out.append("<div class='section'>")
        out.append("<h2 class='subhead'>Model overview</h2>")
        out.append(_render_models_section(metrics, coefficients))
        out.append("</div>")
        out.append("</body></html>")
        return "".join(out)

    # Watchlist tab (default).
    out.append("<div class='section'>")
    out.append("<h2 class='subhead'>Current prediction</h2>")
    out.append(_render_current_prediction(payload, metrics))
    out.append("</div>")

    out.append("<div class='section'>")
    out.append("<h2 class='subhead'>Contestants · Kalshi markets vs model</h2>")
    if payload.get("synthesized_state"):
        out.append(
            "<p class='small gray'>"
            "⚠ No hand-edited <code>current_state.json</code> found — every "
            "active contestant is being scored with default features, so "
            "model probabilities are uniform. Edit "
            "<code>data/raw/current_state.json</code> to populate per-"
            "contestant signal (visibility, prior votes, alliance state, "
            "etc.) and the model will differentiate.</p>"
        )
    out.append(_render_watchlist_table(payload))
    out.append("</div>")

    out.append("<div class='section'>")
    out.append("<h2 class='subhead'>Validators</h2>")
    out.append(_render_validators_panel(payload))
    out.append("</div>")

    out.append("<div class='section'>")
    out.append("<h2 class='subhead'>Reddit signal</h2>")
    out.append(_render_reddit_panel(payload))
    out.append("</div>")

    out.append("</body></html>")
    return "".join(out)
