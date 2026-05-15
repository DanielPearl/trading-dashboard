"""Survivor-elimination dashboard view.

Reads watchlist.json + metrics.json + model_coefficients.json written
by the survivor-elimination bot (one row per active "Will X be
eliminated …" Kalshi market). Renders the watchlist + Models tabs
using the same CSS / section / body chrome the tennis bot uses, so
the page is visually indistinguishable from the other JSON-source
bot pages.

Only per-episode elimination markets are surfaced — season-winner
markets are filtered out by the bot's exporter (see
``survivor.dashboard.export_watchlist.build_watchlist``). When no
elimination markets are active, ``is_available`` returns False so
the dashboard hides the bot card + redirects ``?bot=survivor`` to a
friendly "not yet" stub.
"""
from __future__ import annotations

import html
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

log = logging.getLogger("dashboard.survivor")


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


def is_available(watchlist_path: str | None) -> bool:
    """The bot is "available" only when the watchlist file lists at
    least one active elimination market. Used by the dashboard's bot
    registry to hide the homepage card and the bot dropdown entry
    when nothing's tradeable today.
    """
    payload = load_watchlist(watchlist_path)
    rows = payload.get("rows") or []
    if not rows:
        return False
    for r in rows:
        if (r.get("market_type") or "") == "elimination" \
                and (r.get("status") or "").lower() not in {"closed", "settled",
                                                              "finalized", "cancelled"}:
            return True
    return False


def model_summary_for_card(metrics_path: str | None,
                            sim_state_path: str | None = None
                            ) -> Dict[str, Any]:
    """Project metrics.json into the shape the cross-bot card grid
    expects. P/R/F1 come from the trainer's tuned threshold."""
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
        "threshold": blended.get("threshold"),
        "feature_count": int(metrics.get("feature_count", 0) or 0),
        "actual_wins": 0,
        "actual_losses": 0,
    }


def summary_for_rollup(sim_state_path: str | None) -> Dict[str, Any]:
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
    if v is None: return "0"
    try:
        pp = float(v) * 100
        if round(pp, 1) == 0: return "0"
        return f"{pp:+.1f}pp"
    except (TypeError, ValueError):
        return "0"


def _fmt_signed_ev(v) -> str:
    if v is None: return "0"
    try:
        x = float(v)
        if round(x, 3) == 0: return "0"
        return f"{x:+.3f}"
    except (TypeError, ValueError):
        return "0"


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
    """Map verdict -> the same .badge-* / .status-pill classes the
    other watchlist pages use, so the colour vocabulary matches."""
    tip = ""
    if blockers:
        tip = f" title='Blockers: {html.escape(', '.join(blockers))}'"
    cls = "badge-yes" if verdict == "BUY YES" else \
          "badge-no" if verdict == "BUY NO" else \
          "badge-hedge" if verdict == "WATCH" else "badge-skip"
    return f"<span class='badge {cls}'{tip}>{html.escape(verdict)}</span>"


# --------------------------------------------------------------------------- #
# Sections (rendered inside section/body wrappers identical to tennis)        #
# --------------------------------------------------------------------------- #

def _render_tab_bar(current_bot_key: str, active: str = "watchlist") -> str:
    """Three-tab bar matching the tennis renderer's chrome.

    Home → / (cross-bot home), Watchlist → ?bot=…&tab=watchlist,
    Models → ?bot=…&tab=models, History → /?tab=history.
    """
    tabs = [
        ("home", "Home", "/"),
        ("watchlist", "Watchlist", f"?bot={current_bot_key}&tab=watchlist"),
        ("models", "Models", f"?bot={current_bot_key}&tab=models"),
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
    """Same bot-filter dropdown the standard renderer uses, scoped to
    bots currently available."""
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
    """Six-card row at the top of the watchlist tab.

    Same .row / .card structure every other watchlist page uses, so
    the spacing + typography line up.
    """
    blended = metrics.get("blended") or {}
    cards = [
        ("Season",            str(payload.get("season") or "—"), ""),
        ("Episode",           str(payload.get("current_episode") or "—"), ""),
        ("Active contestants", str(payload.get("active_contestants") or
                                    len(payload.get("rows", []))), ""),
        ("Holdout accuracy",  _fmt_pct(blended.get("accuracy"), 1), ""),
        ("Holdout F1",        _fmt_pct(blended.get("f1"), 1), "tuned threshold"),
        ("Updated",           _last_updated_age(payload.get("generated_at")), ""),
    ]
    out = ["<div class='row'>"]
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


def _ticker_cell(ticker: str | None) -> str:
    """Render the Kalshi ticker as a real, clickable link to the
    actual market page on kalshi.com (same idiom the tennis +
    standard watchlist pages use)."""
    if not ticker:
        return "—"
    ticker = str(ticker)
    if ticker.upper().startswith("KX") or ticker.upper().startswith("SURVIVOR"):
        url = f"https://kalshi.com/markets/{ticker.lower()}"
        return (f"<a href='{html.escape(url)}' target='_blank' "
                f"rel='noopener noreferrer' class='ticker-link'>"
                f"{html.escape(ticker)}</a>")
    return html.escape(ticker)


def _render_watchlist_table(payload: Dict[str, Any]) -> str:
    """Contestants table.

    Column shape mirrors the tennis watchlist: Ticker | Title | Side |
    Contracts | Kalshi % | Model % | Edge | EV | Verdict — adapted to
    the per-episode elimination question.
    """
    rows = payload.get("rows") or []
    # Only "Will X be eliminated" markets that are still open.
    rows = [
        r for r in rows
        if (r.get("market_type") or "") == "elimination"
        and (r.get("status") or "").lower() not in {"closed", "settled",
                                                       "finalized", "cancelled"}
    ]
    if not rows:
        return ("<div class='empty'>No active elimination markets right "
                "now.</div>")

    out = ["<div class='watchlist-scroll'>",
           "<table id='survivor-watchlist-table'>",
           "<thead><tr>"
           "<th>Ticker</th>"
           "<th>Title</th>"
           "<th>Contestant</th>"
           "<th class='num' title='Open interest — number of YES contracts open on this market.'>Contracts</th>"
           "<th class='num' title='Kalshi YES implied probability of elimination this episode.'>Kalshi %</th>"
           "<th class='num' title='Model probability of elimination this episode.'>Model %</th>"
           "<th class='num' title='Gap between model and Kalshi (model − market) in percentage points.'>Gap</th>"
           "<th class='num' title='Expected value per $1 contract for YES (= will be eliminated) net of slippage.'>EV YES</th>"
           "<th class='num' title='Expected value per $1 contract for NO (= will survive this episode).'>EV NO</th>"
           "<th class='num' title='Model confidence on this row (0..1).'>Conf</th>"
           "<th>Verdict</th>"
           "</tr></thead><tbody>"]

    for r in rows:
        ticker = r.get("ticker") or r.get("match_id") or ""
        title_text = r.get("title") or ""
        contestant = r.get("contestant") or "—"
        oi = r.get("open_interest")
        oi_str = f"{int(oi):,}" if oi is not None else "—"
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
        verdict_pill = _verdict_badge(verdict, r.get("buy_blockers") or [])

        # Row classes match the standard watchlist's row idiom so the
        # green/red tinting + hover behave the same as tennis/NBA.
        row_classes = []
        if verdict == "BUY YES":
            row_classes += ["row-bought", "bought-yes"]
        elif verdict == "BUY NO":
            row_classes += ["row-bought", "bought-no"]
        elif verdict == "WATCH":
            row_classes += ["row-suspect"]
        row_cls = " ".join(row_classes)

        title_cell = (
            f"<td title='{html.escape(str(title_text))}' "
            f"style='max-width:320px;'>"
            f"<span class='small gray' style='display:block;overflow:hidden;"
            f"text-overflow:ellipsis;white-space:nowrap;'>"
            f"{html.escape(str(title_text))}</span></td>"
        )
        out.append(
            f"<tr class='{row_cls}'>"
            f"<td class='mono small'>{_ticker_cell(ticker)}</td>"
            f"{title_cell}"
            f"<td><strong>{html.escape(str(contestant))}</strong></td>"
            f"<td class='num'>{oi_str}</td>"
            f"<td class='num'>{_fmt_pct(mkt, 0)}</td>"
            f"<td class='num'>{_fmt_pct(mdl, 0)}</td>"
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
    rows = payload.get("rows") or []
    rows = [r for r in rows if (r.get("market_type") or "") == "elimination"]
    if not rows:
        return ""
    interesting = sorted(rows,
                          key=lambda r: -(r.get("reddit_boot_pick_count") or 0))[:8]
    if not any((r.get("reddit_boot_pick_count") or 0) for r in interesting):
        return ("<div class='empty'>No Reddit boot-prediction signal yet — "
                "either creds aren't configured or no post-episode "
                "discussion has named a contestant as the next boot.</div>")
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
    rows = payload.get("rows") or []
    rows = [r for r in rows if (r.get("market_type") or "") == "elimination"]
    if not rows:
        return ""
    total = len(rows)
    buys = sum(1 for r in rows if r.get("verdict") in ("BUY YES", "BUY NO"))
    watches = sum(1 for r in rows if r.get("verdict") == "WATCH")
    skips = sum(1 for r in rows if r.get("verdict") == "SKIP")
    blocker_counts: Dict[str, int] = {}
    for r in rows:
        for b in (r.get("buy_blockers") or []):
            blocker_counts[b] = blocker_counts.get(b, 0) + 1
    out = ["<div class='row'>",
           f"<div class='card'><div class='label'>Total rows</div>"
           f"<div class='value'>{total}</div></div>",
           f"<div class='card'><div class='label'>BUY rows</div>"
           f"<div class='value green'>{buys}</div></div>",
           f"<div class='card'><div class='label'>WATCH</div>"
           f"<div class='value yellow'>{watches}</div></div>",
           f"<div class='card'><div class='label'>SKIP</div>"
           f"<div class='value gray'>{skips}</div></div>",
           "</div>"]
    if blocker_counts:
        out.append("<table><thead><tr><th>Validator blocker</th>"
                    "<th class='num'>Rows</th></tr></thead><tbody>")
        for reason, n in sorted(blocker_counts.items(), key=lambda kv: -kv[1]):
            out.append(
                f"<tr><td>{html.escape(reason)}</td>"
                f"<td class='num'>{n}</td></tr>"
            )
        out.append("</tbody></table>")
    return "".join(out)


def _render_models_section(metrics: Dict[str, Any],
                            coefficients: Dict[str, Any]) -> str:
    blended = metrics.get("blended") or {}
    blended_train = metrics.get("blended_train") or {}
    logistic = metrics.get("logistic") or {}
    logistic_train = metrics.get("logistic_train") or {}
    gbt = metrics.get("calibrated_gbt") or {}
    gbt_train = metrics.get("calibrated_gbt_train") or {}
    out: List[str] = []
    out.append(
        "<p class='small gray'>The elimination model is trained on "
        "every modern-era Survivor boot (seasons 41–49), one row per "
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
    threshold = blended.get("threshold")
    pos_rate_train = metrics.get("train_positive_rate")
    pos_rate_test = metrics.get("test_positive_rate")
    if threshold is not None:
        out.append(
            f"<p class='small gray'>Class balance is heavily skewed — "
            f"about <strong>{_fmt_pct(pos_rate_train, 1)}</strong> of "
            f"training rows and "
            f"<strong>{_fmt_pct(pos_rate_test, 1)}</strong> of holdout "
            f"rows are boots, so the trainer sweeps the prediction "
            f"threshold on the training-set probabilities and locks in "
            f"the F1-maximising value (≥ 30% recall floor). The blended "
            f"model is using threshold <code>{threshold:.2f}</code>.</p>"
        )

    out.append("<h3 class='subhead'>Probabilistic quality "
                "<span class='small gray'>(threshold-independent)</span></h3>")
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

    out.append("<h3 class='subhead'>Predicted vs actual "
                "<span class='small gray'>(P / R / F1 at tuned threshold)"
                "</span></h3>")
    out.append("<table><thead><tr>"
               "<th>Component</th><th>Split</th>"
               "<th>Precision</th><th>Recall</th><th>F1</th>"
               "<th>Accuracy</th></tr></thead><tbody>")
    rows = [
        ("Logistic regression", "Train", logistic_train),
        ("Logistic regression", "Test",  logistic),
        ("Calibrated GBT",      "Train", gbt_train),
        ("Calibrated GBT",      "Test",  gbt),
        ("Blended (live)",      "Train", blended_train),
        ("Blended (live)",      "Test",  blended),
    ]
    for name, split, mm in rows:
        if not mm:
            out.append(f"<tr><td>{html.escape(name)}</td>"
                        f"<td>{split}</td><td>—</td><td>—</td>"
                        f"<td>—</td><td>—</td></tr>")
            continue
        out.append(
            f"<tr><td>{html.escape(name)}</td>"
            f"<td><span class='small gray'>{split}</span></td>"
            f"<td>{_fmt_pct(mm.get('precision'), 1)}</td>"
            f"<td>{_fmt_pct(mm.get('recall'), 1)}</td>"
            f"<td>{_fmt_pct(mm.get('f1'), 1)}</td>"
            f"<td>{_fmt_pct(mm.get('accuracy'), 1)}</td></tr>"
        )
    out.append("</tbody></table>")

    out.append(
        f"<p class='small gray'>Train seasons: "
        f"{html.escape(str(metrics.get('train_seasons') or '—'))} · "
        f"Test seasons: "
        f"{html.escape(str(metrics.get('test_seasons') or '—'))} · "
        f"Best model: <code>{html.escape(str(metrics.get('best_model') or '—'))}</code> · "
        f"Rows train/test: "
        f"{metrics.get('rows_train', '—')} / "
        f"{metrics.get('rows_test', '—')}</p>"
    )

    log_coefs = (coefficients.get("logistic") or {})
    feats = log_coefs.get("features") or []
    coefs = log_coefs.get("coefficients") or []
    intercept = log_coefs.get("intercept")
    if feats and coefs:
        out.append("<h3 class='subhead'>Model coefficients · "
                    "logistic regression</h3>")
        out.append("<table><thead><tr>"
                    "<th>Feature</th><th>Coefficient</th>"
                    "</tr></thead><tbody>")
        ranked = sorted(zip(feats, coefs),
                         key=lambda fc: -abs(fc[1]))
        for n, c in ranked:
            out.append(
                f"<tr><td><code>{html.escape(n)}</code></td>"
                f"<td>{c:+.4f}</td></tr>"
            )
        if intercept is not None:
            out.append(
                f"<tr><td><code>(intercept)</code></td>"
                f"<td>{intercept:+.4f}</td></tr>"
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
    """Render the survivor page using the standard dashboard's CSS.

    Tabs:
      watchlist  — current-episode contestants + EV table
      models     — training metrics + coefficients

    Home and History tabs route through the standard cross-bot
    renderer (the tab bar links to ``/`` and ``/?tab=history``).
    """
    metrics = load_metrics(metrics_path)
    coefficients = load_coefficients(coefficients_path)
    payload = load_watchlist(watchlist_path)

    # Lazy-import the standard CSS so an isolated test wouldn't drag
    # the whole dashboard module in.
    from .dashboard import CSS  # type: ignore

    rows_all = payload.get("rows") or []
    elim_rows = [r for r in rows_all
                 if (r.get("market_type") or "") == "elimination"
                 and (r.get("status") or "").lower() not in
                     {"closed", "settled", "finalized", "cancelled"}]
    has_active = bool(elim_rows)

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
    active_tab = tab_key if tab_key in ("watchlist", "models") else "watchlist"
    out.append(_render_tab_bar(current_bot_key, active=active_tab))

    if active_tab == "models":
        out.append("<div class='section'><h2>Model</h2><div class='body'>")
        out.append(_render_bot_dropdown(available_bots, current_bot_key))
        out.append(_render_models_section(metrics, coefficients))
        out.append("</div></div>")
    else:
        # Watchlist tab.
        out.append("<div class='section'><h2>Watchlist — model vs market</h2>"
                    "<div class='body'>")
        out.append(_render_bot_dropdown(available_bots, current_bot_key))

        if not has_active:
            out.append(
                "<div class='empty'>"
                "No active <em>Will&nbsp;X&nbsp;be&nbsp;eliminated</em> "
                "markets on Kalshi right now. The watchlist will populate "
                "as soon as the next episode's elimination contracts open."
                "</div></div></div></body></html>"
            )
            return "".join(out)

        out.append(_render_current_prediction(payload, metrics))

        age = _last_updated_age(payload.get("generated_at"))
        out.append(
            f"<h3 class='subhead'>Active elimination markets · "
            f"{len(elim_rows)} <span class='small gray'>"
            f"generated {html.escape(age)}</span></h3>"
        )
        out.append(_render_watchlist_table(payload))

        out.append("<h3 class='subhead'>Validators</h3>")
        out.append(_render_validators_panel(payload))

        out.append("<h3 class='subhead'>Reddit signal</h3>")
        out.append(_render_reddit_panel(payload))

        out.append("</div></div>")

    out.append("</body></html>")
    return "".join(out)
