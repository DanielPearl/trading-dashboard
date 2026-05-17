"""Billboard Charts dashboard view.

Reads watchlist.json + metrics.json + model_coefficients.json written
by the billboard-charts bot (one row per open Kalshi Billboard market
— typically KXTOPALBUM-* contracts for the upcoming Billboard 200 #1
event). Renders the watchlist + Models tabs using the same CSS /
section / body chrome the survivor + tennis bots use, so the page is
visually indistinguishable from the other JSON-source bot pages.

Same JSON-source pattern as the survivor adapter — see
``trading_dashboard/survivor.py`` for the prototype this is modelled
on. The shape difference is per-album rows (Album / Artist columns)
instead of per-contestant rows.
"""
from __future__ import annotations

import html
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

log = logging.getLogger("dashboard.billboard")


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
    return load_metrics(path)


def is_available(metrics_path: str | None) -> bool:
    """Available whenever the trained model artifact exists. The
    homepage card and bot dropdown stay visible even when no
    Billboard markets are currently open — the watchlist page
    surfaces the empty state inside the standard chrome.
    """
    return bool(metrics_path and Path(metrics_path).exists())


def model_summary_for_card(metrics_path: str | None,
                            sim_state_path: str | None = None
                            ) -> Dict[str, Any]:
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


def closed_positions_for_rollup(sim_state_path: str | None,
                                  limit: int = 100) -> List[Dict[str, Any]]:
    """Billboard sim_state shares the tennis schema (open_positions /
    closed_positions). Delegating to the tennis adapter avoids
    duplicating the projection logic. Returns [] when sim_state.json
    is absent (the watchlist mode is read-only and doesn't need it).
    """
    if not sim_state_path or not Path(sim_state_path).exists():
        return []
    from . import tennis as _tennis  # local import — no circular dep
    return _tennis.closed_positions_for_rollup(sim_state_path, limit=limit)


def card_summary_for_home(metrics_path: str | None,
                            watchlist_path: str | None) -> Dict[str, Any]:
    """Compact dict the home-page Billboard card consumes (top predicted
    album / artist, model probability, market probability, edge,
    coefficients summary, freshness).

    Read by render_card_html below and by the cross-bot home-page
    grid via dashboard.py's get-bot-model-cards logic.
    """
    metrics = load_metrics(metrics_path)
    payload = load_watchlist(watchlist_path)
    rows = payload.get("rows") or []
    top = rows[0] if rows else (payload.get("current_top_prediction") or {})
    return {
        "model_name": "Billboard Charts",
        "target": "Billboard 200 #1",
        "top_album": (top or {}).get("album"),
        "top_artist": (top or {}).get("artist"),
        "model_prob": (top or {}).get("model_prob"),
        "market_prob": (top or {}).get("market_prob"),
        "edge": (top or {}).get("edge"),
        "ev_yes": (top or {}).get("ev_yes"),
        "verdict": (top or {}).get("verdict"),
        "ticker": (top or {}).get("ticker"),
        "generated_at": payload.get("generated_at"),
        "metrics": metrics.get("blended") or {},
        "row_count": len(rows),
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
    tip = ""
    if blockers:
        tip = f" title='Blockers: {html.escape(', '.join(blockers))}'"
    cls = "badge-yes" if verdict == "BUY YES" else \
          "badge-no" if verdict == "BUY NO" else \
          "badge-hedge" if verdict == "WATCH" else "badge-skip"
    return f"<span class='badge {cls}'{tip}>{html.escape(verdict)}</span>"


def _ticker_cell(ticker: str | None) -> str:
    if not ticker:
        return "—"
    ticker = str(ticker)
    if ticker.upper().startswith("KX"):
        url = f"https://kalshi.com/markets/{ticker.lower()}"
        return (f"<a href='{html.escape(url)}' target='_blank' "
                f"rel='noopener noreferrer' class='ticker-link'>"
                f"{html.escape(ticker)}</a>")
    return html.escape(ticker)


# --------------------------------------------------------------------------- #
# Tab bar + bot dropdown — identical to survivor                              #
# --------------------------------------------------------------------------- #

def _render_tab_bar(current_bot_key: str, active: str = "watchlist") -> str:
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
    if not available_bots:
        return ""
    out = ["<div class='bot-filter-bar'>",
           "<label class='filter-label' for='billboard-bot-select'>Bot</label>",
           "<select id='billboard-bot-select' class='bot-select' "
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


# --------------------------------------------------------------------------- #
# Watchlist sections                                                          #
# --------------------------------------------------------------------------- #

def _render_current_prediction(payload: Dict[str, Any],
                                metrics: Dict[str, Any]) -> str:
    blended = metrics.get("blended") or {}
    rows = payload.get("rows") or []
    top = rows[0] if rows else {}
    cards = [
        ("Top predicted album", str(top.get("album") or "—"), ""),
        ("Artist", str(top.get("artist") or "—"), ""),
        ("Model P(#1)", _fmt_pct(top.get("model_prob"), 1), ""),
        ("Kalshi implied", _fmt_pct(top.get("market_prob"), 1), ""),
        ("Edge", _fmt_signed_pp(top.get("edge")), "model − market"),
        ("Updated", _last_updated_age(payload.get("generated_at")), ""),
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


def _render_watchlist_table(payload: Dict[str, Any]) -> str:
    rows = payload.get("rows") or []
    if not rows:
        return ("<div class='empty'>No active Billboard markets right "
                "now. The watchlist will populate as soon as the next "
                "chart-week's Kalshi contracts open.</div>")

    out = ["<div class='watchlist-scroll'>",
           "<table id='billboard-watchlist-table'>",
           "<thead><tr>"
           "<th>Ticker</th>"
           "<th>Title</th>"
           "<th>Album</th>"
           "<th>Artist</th>"
           "<th class='num' title='Yes bid in cents.'>Yes bid</th>"
           "<th class='num' title='Yes ask in cents.'>Yes ask</th>"
           "<th class='num' title='No bid in cents.'>No bid</th>"
           "<th class='num' title='No ask in cents.'>No ask</th>"
           "<th class='num' title='Last trade price in cents.'>Last</th>"
           "<th class='num' title='Kalshi YES-side implied probability.'>Kalshi %</th>"
           "<th class='num' title='Model probability album hits #1.'>Model %</th>"
           "<th class='num' title='Edge = model − market (percentage points).'>Edge</th>"
           "<th class='num' title='Expected value per $1 contract for YES (net of slippage).'>EV YES</th>"
           "<th class='num' title='Expected value per $1 contract for NO.'>EV NO</th>"
           "<th class='num' title='Cumulative open interest on this market.'>OI</th>"
           "<th class='num' title='Cumulative volume on this market.'>Vol</th>"
           "<th class='num' title='Model confidence on this row (0..1).'>Conf</th>"
           "<th>Verdict</th>"
           "</tr></thead><tbody>"]

    for r in rows:
        ticker = r.get("ticker") or ""
        title_text = r.get("title") or ""
        album = r.get("album") or "—"
        artist = r.get("artist") or "—"
        mkt = r.get("market_prob")
        mdl = r.get("model_prob")
        edge = r.get("edge")
        ev_yes = r.get("ev_yes")
        ev_no = r.get("ev_no")
        conf = r.get("confidence_score")
        oi = r.get("open_interest")
        vol = r.get("volume")

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
            f"style='max-width:280px;'>"
            f"<span class='small gray' style='display:block;overflow:hidden;"
            f"text-overflow:ellipsis;white-space:nowrap;'>"
            f"{html.escape(str(title_text))}</span></td>"
        )

        def _cents(v):
            return "—" if v is None else f"{int(v)}"

        def _intnum(v):
            try:
                return f"{int(v):,}" if v is not None else "—"
            except (TypeError, ValueError):
                return "—"

        out.append(
            f"<tr class='{row_cls}'>"
            f"<td class='mono small'>{_ticker_cell(ticker)}</td>"
            f"{title_cell}"
            f"<td><strong>{html.escape(str(album))}</strong></td>"
            f"<td>{html.escape(str(artist))}</td>"
            f"<td class='num'>{_cents(r.get('yes_bid_cents'))}</td>"
            f"<td class='num'>{_cents(r.get('yes_ask_cents'))}</td>"
            f"<td class='num'>{_cents(r.get('no_bid_cents'))}</td>"
            f"<td class='num'>{_cents(r.get('no_ask_cents'))}</td>"
            f"<td class='num'>{_cents(r.get('last_price_cents'))}</td>"
            f"<td class='num'>{_fmt_pct(mkt, 0)}</td>"
            f"<td class='num'>{_fmt_pct(mdl, 0)}</td>"
            f"<td class='num {edge_cls}'>{_fmt_signed_pp(edge)}</td>"
            f"<td class='num {ev_yes_cls}'>{_fmt_signed_ev(ev_yes)}</td>"
            f"<td class='num {ev_no_cls}'>{_fmt_signed_ev(ev_no)}</td>"
            f"<td class='num'>{_intnum(oi)}</td>"
            f"<td class='num'>{_intnum(vol)}</td>"
            f"<td class='num'>{conf if conf is not None else '—'}</td>"
            f"<td>{verdict_pill}</td>"
            f"</tr>"
        )
    out.append("</tbody></table></div>")
    return "".join(out)


def _render_external_signals_panel(payload: Dict[str, Any]) -> str:
    """Per-album external-signal snapshot (Spotify popularity,
    Reddit chatter, Google Trends, YouTube views). Surfaces the
    inputs the live scorer used so the user can sanity-check why
    a particular row scored the way it did."""
    rows = payload.get("rows") or []
    if not rows:
        return ""
    out = ["<table>",
           "<thead><tr>"
           "<th>Album</th>"
           "<th>Artist</th>"
           "<th class='num' title='Spotify artist popularity (0-100).'>Sp pop</th>"
           "<th class='num' title='Spotify album popularity (0-100).'>Sp alb pop</th>"
           "<th class='num' title='Reddit mentions across music subreddits, last 14 days.'>Reddit n</th>"
           "<th class='num' title='Average Reddit-post sentiment polarity (-1..+1).'>Sentiment</th>"
           "<th class='num' title='Google Trends 7-day search score (0-100).'>Trends 7d</th>"
           "<th class='num' title='Trends velocity: 7-day avg / 30-day avg.'>Velocity</th>"
           "<th class='num' title='YouTube top-matching-video view count.'>YT views</th>"
           "</tr></thead><tbody>"]
    for r in rows[:30]:
        out.append(
            f"<tr><td><strong>{html.escape(str(r.get('album') or '—'))}</strong></td>"
            f"<td>{html.escape(str(r.get('artist') or '—'))}</td>"
            f"<td class='num'>{r.get('spotify_artist_popularity') or 0}</td>"
            f"<td class='num'>{r.get('spotify_album_popularity') or 0}</td>"
            f"<td class='num'>{r.get('reddit_mention_count') or 0}</td>"
            f"<td class='num'>{float(r.get('reddit_sentiment') or 0):+.2f}</td>"
            f"<td class='num'>{r.get('trends_score_7d') or 0:.1f}</td>"
            f"<td class='num'>{r.get('trends_velocity_7d_vs_30d') or 1.0:.2f}</td>"
            f"<td class='num'>{int(r.get('youtube_top_video_views') or 0):,}</td>"
            f"</tr>"
        )
    out.append("</tbody></table>")
    return "".join(out)


def _render_validators_panel(payload: Dict[str, Any]) -> str:
    rows = payload.get("rows") or []
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


# --------------------------------------------------------------------------- #
# Price/probability sparkline panel                                            #
# --------------------------------------------------------------------------- #

def _render_price_chart(payload: Dict[str, Any]) -> str:
    """Inline SVG sparkline of model_prob vs market_prob for the
    top predicted album. Cheap, no JS dependency — pure server-
    side render keeps the page within the dashboard's existing
    no-runtime-JS chrome.

    The bot writes a snapshot of (model_prob, market_prob, ts) per
    tick into data/market_snapshots/<ticker>.jsonl; for the v1
    payload we don't have that yet, so the sparkline falls back to
    a static side-by-side gauge.
    """
    rows = payload.get("rows") or []
    if not rows:
        return ""
    top = rows[0]
    mdl = top.get("model_prob") or 0
    mkt = top.get("market_prob") or 0
    out = ["<div class='card' style='max-width:480px;'>",
           f"<div class='label'>Probability snapshot · "
           f"{html.escape(str(top.get('album') or '—'))}</div>",
           "<svg viewBox='0 0 200 80' width='100%' height='80'>",
           f"<rect x='10' y='20' width='{int(mdl * 180)}' height='12' "
           f"fill='#4caf50'/>",
           f"<text x='10' y='17' font-size='10' fill='#333'>Model "
           f"{_fmt_pct(mdl, 1)}</text>",
           f"<rect x='10' y='50' width='{int(mkt * 180)}' height='12' "
           f"fill='#888'/>",
           f"<text x='10' y='47' font-size='10' fill='#333'>Market "
           f"{_fmt_pct(mkt, 1)}</text>",
           "</svg></div>"]
    return "".join(out)


# --------------------------------------------------------------------------- #
# Models tab                                                                  #
# --------------------------------------------------------------------------- #

def _render_models_section(metrics: Dict[str, Any],
                            coefficients: Dict[str, Any]) -> str:
    blended = metrics.get("blended") or {}
    families = metrics.get("families") or {}
    best_model = metrics.get("best_model") or "—"

    out: List[str] = []
    out.append(
        "<p class='small gray'>The Billboard 200 #1 model is a binary "
        "classifier trained on the weekly Billboard 200 chart "
        "archive. The dependent variable is "
        "<code>is_billboard_200_number_1</code> — 1 if the album hit "
        "rank #1 that chart week, 0 otherwise. Features cover the "
        "album's prior chart performance (peak position so far, "
        "weeks on chart, debut rank, 3-week best rank), the "
        "artist's career history (lifetime prior #1s, weeks since "
        "last #1, prior top-10 weeks), release context (track "
        "count, deluxe/surprise flags, release month/dow) and "
        "external signals (Spotify popularity + followers, Reddit "
        "chatter + sentiment, Google Trends 7d score + velocity, "
        "YouTube view counts). The trainer fits logistic regression, "
        "Random Forest, HistGradientBoosting, and — when their "
        "packages are installed — XGBoost and LightGBM, then picks "
        "the family with the best held-out F1 as the production "
        "model. Class balance is heavily skewed (~0.5% of "
        "album-weeks are #1), so the trainer sweeps the prediction "
        "threshold on the training set with a 30% recall floor and "
        "locks in the F1-maximising cutoff for both train and test "
        "evaluation.</p>"
    )

    threshold = blended.get("threshold")
    pos_rate_train = metrics.get("train_positive_rate")
    pos_rate_test = metrics.get("test_positive_rate")
    if threshold is not None:
        out.append(
            f"<p class='small gray'>Train positive rate: "
            f"<strong>{_fmt_pct(pos_rate_train, 2)}</strong> · "
            f"Test positive rate: "
            f"<strong>{_fmt_pct(pos_rate_test, 2)}</strong> · "
            f"Blended threshold: <code>{threshold:.2f}</code>.</p>"
        )

    # Per-family held-out metrics.
    if families:
        out.append("<h3 class='subhead'>Held-out test performance "
                    "<span class='small gray'>(chart weeks "
                    f"{html.escape(str(metrics.get('test_chart_dates') or ''))})"
                    "</span></h3>")
        out.append("<table><thead><tr>"
                    "<th>Family</th><th>F1</th><th>Precision</th>"
                    "<th>Recall</th><th>Accuracy</th>"
                    "<th>Brier</th><th>ROC AUC</th>"
                    "</tr></thead><tbody>")
        ranked = sorted(families.items(),
                         key=lambda kv: -(kv[1].get("test") or {}).get("f1", 0))
        for name, fam in ranked:
            t = fam.get("test") or {}
            if not t:
                continue
            mark = " ←" if name == best_model else ""
            out.append(
                f"<tr><td><code>{html.escape(str(name))}</code>"
                f"{html.escape(mark)}</td>"
                f"<td>{_fmt_pct(t.get('f1'), 1)}</td>"
                f"<td>{_fmt_pct(t.get('precision'), 1)}</td>"
                f"<td>{_fmt_pct(t.get('recall'), 1)}</td>"
                f"<td>{_fmt_pct(t.get('accuracy'), 1)}</td>"
                f"<td>{t.get('brier', 0):.4f}</td>"
                f"<td>{_fmt_pct(t.get('roc_auc'), 1)}</td>"
                f"</tr>"
            )
        out.append("</tbody></table>")

    # Train vs test for the winning family — exposes drift /
    # over-fitting at a glance.
    if best_model and best_model in families:
        fam = families[best_model]
        out.append("<h3 class='subhead'>Predicted vs actual · winning "
                    f"model <code>{html.escape(best_model)}</code> "
                    "<span class='small gray'>(P / R / F1 at tuned "
                    "threshold)</span></h3>")
        out.append("<table><thead><tr>"
                    "<th>Split</th><th>Precision</th><th>Recall</th>"
                    "<th>F1</th><th>Accuracy</th><th>Brier</th>"
                    "</tr></thead><tbody>")
        for split, mm in [("Train", fam.get("train") or {}),
                            ("Test",  fam.get("test")  or {})]:
            if not mm:
                continue
            out.append(
                f"<tr><td>{split}</td>"
                f"<td>{_fmt_pct(mm.get('precision'), 1)}</td>"
                f"<td>{_fmt_pct(mm.get('recall'), 1)}</td>"
                f"<td>{_fmt_pct(mm.get('f1'), 1)}</td>"
                f"<td>{_fmt_pct(mm.get('accuracy'), 1)}</td>"
                f"<td>{mm.get('brier', 0):.4f}</td>"
                f"</tr>"
            )
        out.append("</tbody></table>")

    out.append(
        f"<p class='small gray'>Train chart weeks: "
        f"{html.escape(str(metrics.get('train_chart_dates') or '—'))} · "
        f"Test chart weeks: "
        f"{html.escape(str(metrics.get('test_chart_dates') or '—'))} · "
        f"Best model: <code>{html.escape(str(metrics.get('best_model') or '—'))}</code> · "
        f"Rows train/test: "
        f"{metrics.get('rows_train', '—')} / "
        f"{metrics.get('rows_test', '—')}</p>"
    )

    # Logistic-regression coefficients (always populated by the trainer
    # so this table renders even when a tree-based family wins).
    log_coefs = (coefficients.get("logistic") or {})
    feats = log_coefs.get("features") or []
    coefs = log_coefs.get("coefficients") or []
    intercept = log_coefs.get("intercept")
    if feats and coefs:
        out.append("<h3 class='subhead'>Model coefficients · "
                    "logistic regression "
                    "<span class='small gray'>(ranked by absolute "
                    "magnitude)</span></h3>")
        out.append("<table><thead><tr>"
                    "<th>Feature</th><th class='num'>Coefficient</th>"
                    "</tr></thead><tbody>")
        ranked = sorted(zip(feats, coefs),
                         key=lambda fc: -abs(fc[1]))
        for n, c in ranked:
            out.append(
                f"<tr><td><code>{html.escape(n)}</code></td>"
                f"<td class='num'>{c:+.4f}</td></tr>"
            )
        if intercept is not None:
            out.append(
                f"<tr><td><code>(intercept)</code></td>"
                f"<td class='num'>{intercept:+.4f}</td></tr>"
            )
        out.append("</tbody></table>")

    # Tree-based feature importance for whichever non-logistic family
    # is present (the winner is most useful).
    for k in (best_model, "random_forest", "xgboost", "lightgbm", "hist_gbt"):
        if k in coefficients and "importances" in coefficients[k]:
            ci = coefficients[k]
            out.append(f"<h3 class='subhead'>Feature importance · "
                        f"<code>{html.escape(k)}</code></h3>")
            out.append("<table><thead><tr>"
                        "<th>Feature</th><th class='num'>Importance</th>"
                        "</tr></thead><tbody>")
            ranked = sorted(zip(ci["features"], ci["importances"]),
                             key=lambda fc: -fc[1])
            for n, v in ranked:
                out.append(
                    f"<tr><td><code>{html.escape(n)}</code></td>"
                    f"<td class='num'>{v:.4f}</td></tr>"
                )
            out.append("</tbody></table>")
            break

    # Known limitations call-out — surfacing them in the dashboard
    # avoids people mistaking the model for production-grade.
    out.append(
        "<h3 class='subhead'>Known limitations</h3>"
        "<ul class='small gray'>"
        "<li>External-signal features (Spotify popularity, Reddit "
        "chatter, Google Trends, YouTube views) are scored LIVE for "
        "current Kalshi candidates, but are zero-filled for historical "
        "training rows. The model can't learn their weights without "
        "point-in-time historical snapshots — by design (avoids "
        "leakage from today's signals predicting past charts).</li>"
        "<li>The Billboard 200 history loader falls back to a synthetic "
        "panel on a fresh checkout. Cache "
        "<code>data/raw/billboard_200_history.csv</code> to upgrade.</li>"
        "<li>Backtest is currently a stub against a constant market "
        "probability — the dashboard's edge column is the live signal "
        "until Kalshi price-history caching lands.</li>"
        "</ul>"
    )
    return "".join(out)


# --------------------------------------------------------------------------- #
# Page renderer                                                                #
# --------------------------------------------------------------------------- #

def render_page(*, metrics_path: str | None, coefficients_path: str | None,
                watchlist_path: str | None, sim_state_path: str | None = None,
                available_bots: List[dict], current_bot_key: str,
                tab_key: str = "watchlist") -> str:
    """Render the Billboard page using the standard dashboard's CSS.

    Tabs:
      watchlist  — current Kalshi markets + edge / EV table + signals panel
      models     — training metrics + coefficients + limitations
    """
    metrics = load_metrics(metrics_path)
    coefficients = load_coefficients(coefficients_path)
    payload = load_watchlist(watchlist_path)

    from .dashboard import CSS  # type: ignore

    rows = payload.get("rows") or []
    has_active = bool(rows)

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
    out.append(_render_bot_dropdown(available_bots, current_bot_key))
    out.append(_render_tab_bar(current_bot_key, active=active_tab))

    if active_tab == "models":
        out.append("<div class='section'><h2>Model</h2><div class='body'>")
        out.append(_render_models_section(metrics, coefficients))
        out.append("</div></div>")
    else:
        out.append("<div class='section'><h2>Watchlist — model vs market</h2>"
                    "<div class='body'>")

        if not has_active:
            out.append(
                "<div class='empty'>"
                "No active Billboard markets on Kalshi right now. The "
                "watchlist will populate as soon as the next chart-week's "
                "<code>KXTOPALBUM</code> contracts open."
                "</div></div></div></body></html>"
            )
            return "".join(out)

        out.append(_render_current_prediction(payload, metrics))
        out.append(_render_price_chart(payload))

        age = _last_updated_age(payload.get("generated_at"))
        out.append(
            f"<h3 class='subhead'>Active Billboard markets · "
            f"{len(rows)} <span class='small gray'>"
            f"generated {html.escape(age)}</span></h3>"
        )
        out.append(_render_watchlist_table(payload))

        out.append("<h3 class='subhead'>Validators</h3>")
        out.append(_render_validators_panel(payload))

        out.append("<h3 class='subhead'>External signals snapshot</h3>")
        out.append(_render_external_signals_panel(payload))

        out.append("</div></div>")

    out.append("</body></html>")
    return "".join(out)
