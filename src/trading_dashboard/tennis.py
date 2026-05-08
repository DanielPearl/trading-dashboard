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


def active_bets_for_rollup(sim_state_path: str | None,
                             watchlist_path: str | None = None
                             ) -> List[Dict[str, Any]]:
    """Return tennis open paper positions in the dict shape the
    standard ``_render_active_bets_table`` expects.

    Mapping from sim_state.json position record → standard schema:
      ticker          ← match_id (= the real Kalshi event_ticker)
      _match          ← "{player_a} vs {player_b}"
      _side_player    ← side_player (the player we're betting on)
      side            ← "YES" (we always buy the favoured side)
      contracts       ← stake (= 1.0 default; expressed as $1 = 1 contract)
      entry_price_cents ← entry_market_prob * 100
      mark_mid        ← current_market_prob * 100
      opened_at       ← opened_at
      minutes_to_close ← derived from the matching watchlist row's
                         ``expected_expiration_time`` so the standard
                         "Closes in" cell renders the time to match
                         resolution rather than dashing out.
      _bot_name       ← caller fills in

    Tennis stake is in dollars rather than Kalshi contracts; we use a
    1-contract / dollar mapping so the existing dollar columns
    (Entry cost / Potential gain) render in the same units as Kalshi
    bets without special-casing the renderer.
    """
    s = load_sim_state(sim_state_path)
    # Build a per-match_id → expected_expiration_time map from the
    # canonical live-state file so the standard "Closes in" cell can
    # render a real countdown. When the watchlist path isn't in a
    # standard layout, the lookup falls through and Closes-in dashes.
    exp_by_id: Dict[str, str] = {}
    if watchlist_path:
        try:
            wl_path = Path(watchlist_path).parent.parent / "raw" / "live_state.json"
            if wl_path.exists():
                with wl_path.open("r", encoding="utf-8") as f:
                    for rec in json.load(f) or []:
                        mid = rec.get("match_id")
                        exp = rec.get("expected_expiration_time")
                        if mid and exp:
                            exp_by_id[str(mid)] = str(exp)
        except (OSError, json.JSONDecodeError):
            pass
    now = datetime.now(timezone.utc)
    out: List[Dict[str, Any]] = []
    for p in s.get("open_positions") or []:
        entry = p.get("entry_market_prob")
        if entry is None:
            # No real Kalshi quote at open — shouldn't happen given the
            # simulator's filter, but if it slips through we drop the
            # row from the table rather than show a fabricated 50%.
            continue
        entry = float(entry)
        mark = float(p.get("current_market_prob") or entry)
        mid = p.get("match_id", "")
        mtc: float | None = None
        exp = exp_by_id.get(mid)
        if exp:
            try:
                ts = datetime.fromisoformat(exp.replace("Z", "+00:00"))
                mtc = max(0.0, (ts - now).total_seconds() / 60.0)
            except (TypeError, ValueError):
                mtc = None
        out.append({
            "ticker": mid,
            "_match": f"{p.get('player_a','')} vs {p.get('player_b','')}",
            "_side_player": p.get("side_player", ""),
            # Kalshi-published contract title for the side we're
            # betting on. The shared active-bets renderer reads this
            # via ``_title`` (or the standard ``title`` field, which
            # we also fill in for symmetry with the Kalshi-bot path).
            "_title": p.get("title") or "",
            "title": p.get("title") or "",
            "_tournament": p.get("tournament", ""),
            "_surface": p.get("surface", ""),
            "side": "YES",  # tennis always buys the favoured side
            # 1 contract per paper bet — same convention as Kalshi
            # (1 contract = $1 face value at settlement). The
            # standard renderer multiplies entry_price_cents × contracts
            # / 100 for Entry cost; with ``contracts=1`` and entry =
            # real cents from Kalshi's yes_ask, the dollar columns
            # match what the user would pay on the actual exchange.
            "contracts": 1,
            "entry_price_cents": int(round(entry * 100)),
            "mark_mid": mark * 100,
            "opened_at": p.get("opened_at", ""),
            "minutes_to_close": mtc,
            "label_at_open": p.get("label_at_open", ""),
            "reason_at_open": p.get("reason_at_open", ""),
            # Required by the renderer's "why was this bet chosen" hook.
            "model_yes_prob_at_entry": float(p.get("entry_model_prob") or entry),
            "kalshi_yes_prob_at_entry": entry,
        })
    return out


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
    """Tennis equivalent of the standard 'Active bet' table.

    Columns (sport idiom): Ticker | Match | Side | Entry | Mark |
    Live model | Unrealized | Label | Opened. The Match cell shows
    "Player A vs Player B" with tournament + surface underneath; the
    Kalshi YES question text would be redundant on a sport bet so
    the Title column is dropped here too.
    """
    open_positions = sim_state.get("open_positions") or []
    if not open_positions:
        return "<div class='empty'>No active paper bets right now.</div>"
    out = ["<table>",
           "<thead><tr>"
           "<th>Ticker</th>"
           "<th>Match</th><th>Side</th><th>Entry</th>"
           "<th>Mark</th><th>Live model</th>"
           "<th>Unrealized</th><th>Label</th><th>Opened</th>"
           "</tr></thead><tbody>"]
    for p in sorted(open_positions, key=lambda r: r.get("opened_at", ""),
                    reverse=True):
        unr = float(p.get("unrealized_pnl") or 0.0)
        unr_cls = "green" if unr > 0 else ("red" if unr < 0 else "gray")
        mid = str(p.get("match_id") or "")
        if mid.upper().startswith("KX"):
            kalshi_url = f"https://kalshi.com/markets/{mid.lower()}"
            ticker_cell = (
                f"<a href='{html.escape(kalshi_url)}' target='_blank' "
                f"rel='noopener noreferrer' class='ticker-link'>"
                f"{html.escape(mid)}</a>"
            )
        else:
            ticker_cell = html.escape(mid)
        out.append(
            # Reuse the watchlist row's class + data-mid so the
            # forecast-graph click handler picks active-bet rows up
            # as readily as ticker-table rows. Clicking either swaps
            # the chart to that match's projection.
            f"<tr class='tennis-row' data-mid='{html.escape(mid)}'>"
            f"<td class='mono small'>{ticker_cell}</td>"
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
    """Tennis matches table.

    Rows are clickable — the page's JS hook listens for clicks on the
    table body and updates the projected-forecast graph above with
    the row's probabilities. Each row carries a ``data-mid`` attribute
    so the JS can look up the match in the embedded forecast payload.

    Only tradeable rows are rendered: a ticker is shown only when the
    underlying Kalshi market is active *and* has at least one
    published quote (yes_ask or no_ask). Most upcoming-day tennis
    books sit unquoted until close to tipoff; surfacing them here
    would dilute the watchlist with rows the user can't actually
    place a bet on.
    """
    rows_all = payload.get("rows") or []
    rows = [r for r in rows_all if r.get("market_prob_a") is not None]
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
        unquoted = len(rows_all)
        if unquoted:
            return (f"<div class='empty'>No tradeable tennis markets right "
                    f"now — {unquoted} match{'es' if unquoted != 1 else ''} "
                    f"awaiting a Kalshi quote.</div>")
        return "<div class='empty'>No active tennis markets.</div>"

    # Sport-table column shape (mirrors the NBA watchlist):
    #
    #   Ticker | Side | Contracts | Kalshi YES % | Kalshi NO %
    #          | My YES % | My NO % | EV YES | EV NO | Verdict
    #
    # Side = who's going to win — the favoured player the bot is
    #        betting on, with the opponent stacked underneath. The
    #        Kalshi-published "Will X win?" title would be redundant
    #        with this on a sport bet watchlist, so it's dropped.
    out = ["<table id='tennis-watchlist-table'>",
           "<thead><tr>"
           "<th>Ticker</th>"
           "<th title='Who the bot is betting will win.'>Side</th>"
           "<th class='num' title='Open interest — number of YES contracts currently held open on this side.'>Contracts</th>"
           "<th class='num' title='Kalshi market price for YES | NO sides — implied probability each side wins.'>Kalshi % <span class='small gray'>(yes | no)</span></th>"
           "<th class='num' title='Bot model probability for YES | NO.'>My % <span class='small gray'>(yes | no)</span></th>"
           "<th class='num' title='Edge = my probability − Kalshi price, per side. Positive means the bot disagrees with Kalshi in that direction.'>Edge <span class='small gray'>(yes | no)</span></th>"
           "<th class='num' title='Expected value per $1 contract for YES | NO, net of slippage.'>EV <span class='small gray'>(yes | no)</span></th>"
           "<th>Verdict</th>"
           "</tr></thead><tbody>"]
    for r in rows_sorted:
        edge_a = r.get("edge_a") or 0.0
        player_a = str(r.get("player_a", ""))
        player_b = str(r.get("player_b", ""))
        # Side (who the bot thinks will win) is whichever player the
        # model's edge favours.
        favoured_player = player_a if edge_a >= 0 else player_b
        opponent = player_b if favoured_player == player_a else player_a
        match_text = f"{player_a} vs {player_b}"
        # The Side cell shows the favoured player on top with the
        # opponent stacked underneath in small gray text — same idiom
        # the NBA watchlist uses for its Side cell.
        side_html = (
            f"<strong>{html.escape(favoured_player)}</strong>"
            f"<br><span class='small gray'>vs {html.escape(opponent)}</span>"
        )

        mid = str(r.get("match_id") or "")
        if mid.upper().startswith("KX"):
            kalshi_url = f"https://kalshi.com/markets/{mid.lower()}"
            ticker_cell = (
                f"<a href='{html.escape(kalshi_url)}' target='_blank' "
                f"rel='noopener noreferrer' class='ticker-link'>"
                f"{html.escape(mid)}</a>"
            )
        else:
            ticker_cell = html.escape(mid)

        oi = r.get("open_interest")
        oi_str = f"{int(oi):,}" if oi is not None else "—"
        kyes_str = _fmt_pct(r.get("market_prob_a"), 0)
        # Kalshi NO % = 100 − Kalshi YES %. We compute from the raw
        # probability rather than 100 − cents to keep one rounding step.
        mkt_a = r.get("market_prob_a")
        kno_str = (f"{(1.0 - float(mkt_a)) * 100:.0f}%"
                    if mkt_a is not None else "—")
        my_yes_str = _fmt_pct(r.get("live_prob_a"), 0)
        live_a = r.get("live_prob_a")
        my_no_str = (f"{(1.0 - float(live_a)) * 100:.0f}%"
                      if live_a is not None else "—")
        ev_yes = r.get("ev_a")
        ev_no = r.get("ev_b")
        ev_yes_str = (f"${ev_yes:+.3f}" if ev_yes is not None else "—")
        ev_no_str = (f"${ev_no:+.3f}" if ev_no is not None else "—")
        ev_yes_cls = ("green" if ev_yes is not None and ev_yes >= 0.03
                      else "red" if ev_yes is not None and ev_yes <= 0
                      else "yellow" if ev_yes is not None else "gray")
        ev_no_cls = ("green" if ev_no is not None and ev_no >= 0.03
                     else "red" if ev_no is not None and ev_no <= 0
                     else "yellow" if ev_no is not None else "gray")

        # Edge — model probability for the side minus Kalshi's market
        # price for the same side. Positive = bot disagrees with Kalshi
        # in that side's favour; negative = market sees more upside than
        # the model. Half-spread / slippage are not deducted here (that
        # cost shows up in the EV column).
        edge_yes_v = (float(live_a) - float(mkt_a)
                      if (live_a is not None and mkt_a is not None) else None)
        edge_no_v = ((1.0 - float(live_a)) - (1.0 - float(mkt_a))
                     if (live_a is not None and mkt_a is not None) else None)
        def _edge_fmt(e):
            if e is None:
                return "—", "gray"
            cls_ = ("green" if e >= 0.05 else
                    "yellow" if e > 0 else
                    "red" if e <= -0.02 else "gray")
            return f"{e * 100:+.0f}%", cls_
        edge_yes_str, edge_yes_cls = _edge_fmt(edge_yes_v)
        edge_no_str, edge_no_cls = _edge_fmt(edge_no_v)

        verdict_pill = _label_pill(str(r.get("recommended_action", "NO_TRADE")))

        # Combined cells: yes / no per side, each span coloured to
        # preserve the per-side cue after the merge. Slash uses the
        # shared .cell-sep class so the divider stays muted.
        kalshi_cell = (
            f"<td class='num'>"
            f"<span>{kyes_str}</span>"
            f"<span class='cell-sep'> | </span>"
            f"<span>{kno_str}</span></td>"
        )
        my_cell = (
            f"<td class='num'>"
            f"<span>{my_yes_str}</span>"
            f"<span class='cell-sep'> | </span>"
            f"<span>{my_no_str}</span></td>"
        )
        edge_cell = (
            f"<td class='num'>"
            f"<span class='{edge_yes_cls}'>{edge_yes_str}</span>"
            f"<span class='cell-sep'> | </span>"
            f"<span class='{edge_no_cls}'>{edge_no_str}</span></td>"
        )
        ev_cell = (
            f"<td class='num'>"
            f"<span class='{ev_yes_cls}'>{ev_yes_str}</span>"
            f"<span class='cell-sep'> | </span>"
            f"<span class='{ev_no_cls}'>{ev_no_str}</span></td>"
        )
        out.append(
            f"<tr class='tennis-row' data-mid='{html.escape(mid)}' "
            f"style='cursor:pointer'>"
            f"<td class='mono small'>{ticker_cell}</td>"
            f"<td>{side_html}</td>"
            f"<td class='num'>{oi_str}</td>"
            f"{kalshi_cell}"
            f"{my_cell}"
            f"{edge_cell}"
            f"{ev_cell}"
            f"<td>{verdict_pill}</td>"
            "</tr>"
        )
    out.append("</tbody></table>")
    return "".join(out)


def _render_forecast_graph(rows: List[dict]) -> str:
    """Interactive forecast graph rendered as an SVG with vanilla JS.

    Default state: shows the top-edge match. Clicking a row in the
    ticker table below updates the graph with that match's data.
    The graph plots three series for the selected match — pre-match
    probability, live model probability, and market-implied
    probability — anchored at the current moment, plus a 95%
    confidence band around the live probability so the user can see
    the model's uncertainty alongside the point estimate.
    """
    if not rows:
        return "<div class='empty'>No active bets right now.</div>"
    # Build a JSON map of match_id → forecast payload that the JS
    # reads to swap the graph contents on row click.
    payload_map: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        mid = str(r.get("match_id") or "")
        if not mid:
            continue
        live = float(r.get("live_prob_a") or 0.5)
        # 95% CI half-width — we don't currently expose calibration std
        # so we approximate uncertainty as a function of volatility:
        # a calm match (vol ~ 0.05) has ±5pp; a tiebreak (vol ~ 0.55)
        # blows out to ±25pp. Mirrors the way the live rules engine
        # already uses ``volatility_score``.
        vol = float(r.get("volatility_score") or 0.05)
        ci_half = max(0.03, min(0.30, 0.05 + vol * 0.45))
        # Synthesize a per-match "market rules" string so the panel's
        # rules pane reads like the NBA contract-rules block. Tennis
        # markets here aren't real Kalshi contracts, so we describe
        # them in plain English: "settles at 100¢ if {favoured player}
        # beats {opponent} in {tournament} on {surface}; otherwise 0¢."
        favoured = (r.get("player_a", "") if (r.get("edge_a") or 0) >= 0
                     else r.get("player_b", ""))
        opponent = (r.get("player_b", "") if favoured == r.get("player_a", "")
                     else r.get("player_a", ""))
        rules_str = (
            f"YES settles at $1.00 if {favoured} beats {opponent} in this "
            f"{r.get('tournament', '')} match on {r.get('surface', 'Hard')} "
            f"({r.get('round_label', '')}); $0 otherwise. Settled paper "
            f"trade — no Kalshi exchange fees applied."
        )
        payload_map[mid] = {
            "player_a": r.get("player_a", ""),
            "player_b": r.get("player_b", ""),
            "favoured": favoured,
            "opponent": opponent,
            "tournament": r.get("tournament", ""),
            "surface": r.get("surface", ""),
            "pre": float(r.get("pre_match_prob_a") or 0.5),
            "live": live,
            "market": (float(r["market_prob_a"])
                        if r.get("market_prob_a") is not None else None),
            "ci_low": max(0.0, live - ci_half),
            "ci_high": min(1.0, live + ci_half),
            "edge": (float(r["edge_a"])
                      if r.get("edge_a") is not None else None),
            "label": r.get("recommended_action", "NO_TRADE"),
            "score": r.get("current_score", ""),
            "rules": rules_str,
        }
    # Pick the top-edge match as the default.
    default_mid = ""
    if rows:
        sorted_rows = sorted(
            rows, key=lambda r: -abs(float(r.get("edge_a") or 0))
        )
        default_mid = str(sorted_rows[0].get("match_id") or "")

    js_payload = json.dumps(payload_map, default=str)
    return (
        "<div id='tennis-forecast-graph' "
        f"data-default-mid='{html.escape(default_mid)}' "
        "style='background:#0d1117;border:1px solid #21262d;"
        "border-radius:8px;padding:14px 18px;margin:6px 0 10px 0;'>"
        "<div id='tfg-title' style='font-size:13px;color:#f0f6fc;"
        "margin-bottom:6px;font-weight:600;'></div>"
        "<div id='tfg-sub' class='small gray' style='margin-bottom:10px;'></div>"
        "<svg id='tfg-svg' width='100%' height='220' "
        "viewBox='0 0 700 220' preserveAspectRatio='none' "
        "style='display:block;'></svg>"
        "<div id='tfg-legend' class='small' style='margin-top:8px;"
        "display:flex;gap:18px;flex-wrap:wrap;color:#8b949e;'></div>"
        f"<script type='application/json' id='tfg-data'>{js_payload}</script>"
        "</div>"
        + _FORECAST_GRAPH_JS
    )


# JS: vanilla, ~80 lines. Reads the payload from the inline JSON tag,
# wires up row clicks on the ticker table, redraws an SVG. No D3 / no
# Chart.js — keeps the dashboard's stdlib-only footprint. The chart
# layout is a horizontal bar showing the three probability points on
# a 0-100% axis, with a confidence band shaded behind the live point.
_FORECAST_GRAPH_JS = """
<script>
(function() {
  const dataEl = document.getElementById('tfg-data');
  if (!dataEl) return;
  const payload = JSON.parse(dataEl.textContent || '{}');
  const svg = document.getElementById('tfg-svg');
  const titleEl = document.getElementById('tfg-title');
  const subEl = document.getElementById('tfg-sub');
  const legendEl = document.getElementById('tfg-legend');
  const W = 700, H = 220, PAD_L = 50, PAD_R = 30, PAD_T = 30, PAD_B = 40;
  const innerW = W - PAD_L - PAD_R;
  const innerH = H - PAD_T - PAD_B;
  const xOf = (p) => PAD_L + p * innerW;

  function el(tag, attrs, children) {
    const e = document.createElementNS('http://www.w3.org/2000/svg', tag);
    for (const [k, v] of Object.entries(attrs || {})) e.setAttribute(k, v);
    (children || []).forEach(c => e.appendChild(c));
    return e;
  }
  function txt(t) { return document.createTextNode(String(t)); }
  function tspan(content) { return el('text', {}, [txt(content)]); }

  function draw(mid) {
    const d = payload[mid];
    svg.innerHTML = '';
    legendEl.innerHTML = '';
    if (!d) {
      titleEl.textContent = 'No forecast available';
      subEl.textContent = '';
      return;
    }
    titleEl.textContent = d.player_a + ' vs ' + d.player_b;
    const labelTxt = (d.label || '').replace('_', ' ');
    const edgeStr = (d.edge !== null && d.edge !== undefined)
      ? (d.edge >= 0 ? '+' : '') + (d.edge * 100).toFixed(1) + 'pp'
      : '—';
    subEl.textContent = d.tournament + ' · ' + d.surface
      + ' · score ' + (d.score || '0-0')
      + ' · edge ' + edgeStr
      + ' · ' + labelTxt;

    // Axis: probability bar 0..1 (player_a's perspective).
    const axisY = PAD_T + innerH - 18;
    // Background track.
    svg.appendChild(el('rect', {
      x: PAD_L, y: axisY - 8,
      width: innerW, height: 16,
      fill: '#1d232c', stroke: '#30363d', 'stroke-width': '1', rx: 4,
    }));
    // 50% reference line.
    svg.appendChild(el('line', {
      x1: xOf(0.5), x2: xOf(0.5),
      y1: PAD_T + 8, y2: PAD_T + innerH + 4,
      stroke: '#30363d', 'stroke-dasharray': '3,4',
    }));
    // Confidence band around live.
    if (d.ci_low !== undefined && d.ci_high !== undefined) {
      svg.appendChild(el('rect', {
        x: xOf(d.ci_low), y: axisY - 14,
        width: Math.max(2, xOf(d.ci_high) - xOf(d.ci_low)),
        height: 28, fill: '#58a6ff22', stroke: '#58a6ff55',
        'stroke-width': '1', rx: 3,
      }));
    }

    // Plot points: pre / live / market.
    const points = [
      { v: d.pre, color: '#8b949e', label: 'Pre-match' },
      { v: d.live, color: '#58a6ff', label: 'Live model' },
      { v: d.market, color: '#e3b341', label: 'Market' },
    ];
    points.forEach((p) => {
      if (p.v === null || p.v === undefined) return;
      const x = xOf(p.v);
      svg.appendChild(el('circle', {
        cx: x, cy: axisY,
        r: 7, fill: p.color, stroke: '#0d1117', 'stroke-width': '2',
      }));
      const lbl = el('text', {
        x: x, y: axisY - 16, fill: p.color,
        'text-anchor': 'middle', 'font-size': '11', 'font-weight': '600',
      });
      lbl.appendChild(txt((p.v * 100).toFixed(0) + '%'));
      svg.appendChild(lbl);
    });

    // X-axis ticks: 0%, 25%, 50%, 75%, 100% — gives the user a
    // visual sense of where the dot sits without staring at numbers.
    [0, 0.25, 0.5, 0.75, 1].forEach((t) => {
      const x = xOf(t);
      svg.appendChild(el('line', {
        x1: x, x2: x, y1: axisY + 10, y2: axisY + 14,
        stroke: '#30363d',
      }));
      const lbl = el('text', {
        x: x, y: axisY + 28, fill: '#8b949e',
        'text-anchor': 'middle', 'font-size': '10',
      });
      lbl.appendChild(txt((t * 100).toFixed(0) + '%'));
      svg.appendChild(lbl);
    });
    // Y-axis label — implicit 'P(' + player_a + ' wins)'.
    const yAxisLbl = el('text', {
      x: PAD_L, y: PAD_T + 14, fill: '#8b949e', 'font-size': '11',
    });
    yAxisLbl.appendChild(txt('P(' + d.player_a + ' wins)'));
    svg.appendChild(yAxisLbl);

    // Legend.
    points.forEach((p) => {
      if (p.v === null || p.v === undefined) return;
      const item = document.createElement('span');
      item.style.display = 'inline-flex';
      item.style.alignItems = 'center';
      item.style.gap = '6px';
      const dot = document.createElement('span');
      dot.style.cssText = 'display:inline-block;width:10px;height:10px;'
        + 'border-radius:50%;background:' + p.color + ';';
      item.appendChild(dot);
      item.appendChild(txt(p.label + ' ' + (p.v * 100).toFixed(0) + '%'));
      legendEl.appendChild(item);
    });
    if (d.ci_low !== undefined && d.ci_high !== undefined) {
      const ci = document.createElement('span');
      ci.style.color = '#8b949e';
      ci.appendChild(txt('Live 95% CI ' + (d.ci_low * 100).toFixed(0)
        + '% – ' + (d.ci_high * 100).toFixed(0) + '%'));
      legendEl.appendChild(ci);
    }
  }

  // Highlight the currently-selected row.
  function setSelected(mid) {
    document.querySelectorAll('tr.tennis-row').forEach((tr) => {
      tr.classList.toggle('tennis-row-selected', tr.dataset.mid === mid);
    });
  }

  const container = document.getElementById('tennis-forecast-graph');
  const defaultMid = container ? container.dataset.defaultMid : '';
  if (defaultMid) { draw(defaultMid); setSelected(defaultMid); }

  // Wire row clicks → redraw.
  document.addEventListener('click', function (ev) {
    const tr = ev.target.closest('tr.tennis-row');
    if (!tr) return;
    const mid = tr.dataset.mid;
    if (!mid) return;
    draw(mid);
    setSelected(mid);
    // Smooth-scroll the graph into view if it's offscreen.
    const c = document.getElementById('tennis-forecast-graph');
    if (c) c.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  });
})();
</script>
<style>
tr.tennis-row { cursor: pointer; }
tr.tennis-row-selected td { background: #1f2630 !important; }
tr.tennis-row:hover td { background: #1c222b; }
</style>
"""


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
    # Order per user spec: Active paper bets at top, forecast graph in
    # the middle (interactive — click a ticker row to plot it), ticker
    # table at the bottom. The model card lives on the home page (the
    # cross-bot bot grid card), not here.
    out.append("<div class='section'><h2>Watchlist — model vs market</h2>"
               "<div class='body'>")
    out.append(_render_bot_dropdown(available_bots, current_bot_key))
    out.append(_render_current_prediction(metrics, sim_state))

    out.append("<h3 class='subhead'>Active paper bets</h3>")
    out.append(_render_active_paper_bets(sim_state))

    # Filter to tradeable rows for the forecast graph. A row is
    # tradeable when Kalshi has published a quote (market_prob_a not
    # None). The watchlist table does its own filter (and uses the
    # untradeable count for an informative empty state).
    tradeable = [r for r in rows if r.get("market_prob_a") is not None]

    # Active-bets line chart — sport-style row-click changes which
    # match is plotted (pre-match, live model, market probabilities
    # with a 95% CI band). No header label — the chart is the single
    # active-bets visual on the page, no panel chrome.
    out.append(_render_forecast_graph(tradeable))

    age = _last_updated_age(payload.get("generated_at"))
    out.append(
        f"<h3 class='subhead'>Tennis matches · {len(tradeable)} "
        f"<span class='small gray'>(generated {html.escape(age)})</span></h3>"
    )
    out.append(_render_watchlist_table(payload))

    out.append("</div></div>")  # /body /section

    out.append("</body></html>")
    return "".join(out)
