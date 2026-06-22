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
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

log = logging.getLogger("dashboard.tennis")


def _kalshi_fee_cents(price_cents: int | None, contracts: int | None) -> int:
    """Inline copy of dashboard.kalshi_fee_cents to avoid a circular
    import. Kalshi's published fee = ceil(0.07 × contracts × p × (1−p))
    where p is the price in dollars; returns the equivalent cents.
    Zero on inputs that wouldn't be charged (settled / missing).
    """
    if price_cents is None or contracts is None:
        return 0
    try:
        p = int(price_cents)
        n = int(contracts)
    except (TypeError, ValueError):
        return 0
    if n <= 0 or p <= 0 or p >= 100:
        return 0
    return int(math.ceil(0.07 * n * p * (100 - p) / 100.0))

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
        # Training- and held-out test-set sizes. Surface on the Home
        # tab's bot card as "Train rows" / "Test rows".
        "rows_train": metrics.get("rows_train"),
        "rows_test": metrics.get("rows_test"),
        "actual_wins": int(stats.get("wins", 0) or 0),
        "actual_losses": int(stats.get("losses", 0) or 0),
    }


def build_standard_watchlist_rows(
    payload: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Translate tennis watchlist.json rows into the row shape that the
    standard sport-bot ``_render_watchlist`` consumes. One row per match,
    with player A mapped to the YES side and player B to NO.

    Kalshi's ``yes_ask_cents_a`` and ``yes_ask_cents_b`` are the per-side
    market prices — the "NO" cents column is just the other player's
    YES contract price (the pair sums to ~100¢ minus the spread).

    Rows with zero open interest are dropped so the standard renderer
    surfaces only tradeable matches, matching the old tennis-specific
    table's filter.
    """
    raw_rows = payload.get("rows") or []
    out: List[Dict[str, Any]] = []
    for r in raw_rows:
        match_id = str(r.get("match_id") or "")
        if not match_id:
            continue
        oi = r.get("open_interest")
        try:
            if oi is None or float(oi) <= 0:
                continue
        except (TypeError, ValueError):
            continue
        # Prefer the live (in-play adjusted) probability since that's what
        # the bot actually trades on; fall back to the pre-match prior.
        p_a = r.get("live_prob_a")
        if p_a is None:
            p_a = r.get("pre_match_prob_a")
        # Decide which player goes on the "YES" / top side of the row.
        # The standard sport-bot renderer treats YES as the favoured /
        # action side — top of every stacked cell (My %, Kalshi %,
        # Edge, EV) and the bold player name in the Side cell. For
        # tennis, both players have their own YES market (separate
        # Kalshi tickers), and player A is just alphabetical first —
        # not necessarily the favoured side. If we leave the mapping
        # unconditional, rows where player B has the edge render with
        # player A as the bold "Side" name but the bot is actually
        # betting on B (Title cell + Verdict reflect this), inverting
        # the visual. Flip every side-paired field so YES always tracks
        # the favoured side.
        b_favoured = (p_a is not None and float(p_a) < 0.5)
        if b_favoured:
            top_ask = r.get("yes_ask_cents_b")
            bot_ask = r.get("yes_ask_cents_a")
            top_prob = (1.0 - float(p_a)) if p_a is not None else None
            top_raw = ((1.0 - float(r.get("pre_match_prob_a")))
                       if r.get("pre_match_prob_a") is not None else None)
            top_label = r.get("player_b")
            bot_label = r.get("player_a")
            top_title = r.get("title_b") or r.get("title") or ""
        else:
            top_ask = r.get("yes_ask_cents_a")
            bot_ask = r.get("yes_ask_cents_b")
            top_prob = p_a
            top_raw = r.get("pre_match_prob_a")
            top_label = r.get("player_a")
            bot_label = r.get("player_b")
            top_title = r.get("title_a") or r.get("title") or ""
        # The dashboard's Title cell should match what Kalshi shows on
        # the event page the user lands on when they click the ticker
        # (e.g. "Choinski vs Herbert", not "Will Herbert win the
        # Choinski vs Herbert: Qual R2 match?"). Prefer the bot's
        # stored event_title when present; fall back to the per-side
        # market title for older rows that haven't been re-exported.
        display_title = r.get("event_title") or top_title
        buy_eligible = bool(r.get("buy_eligible"))
        buy_side = (r.get("buy_side") or "").upper()
        # ``BUY_YES`` = act on the favoured (top-of-row) side. Whether
        # that's PLAYER_A or PLAYER_B in the underlying row is now
        # encoded by the flip above.
        if buy_eligible and ((buy_side == "A" and not b_favoured)
                              or (buy_side == "B" and b_favoured)):
            verdict = "BUY_YES"
        elif buy_eligible:
            verdict = "BUY_NO"
        else:
            verdict = "SKIP"
        blockers = r.get("buy_blockers") or []
        if blockers:
            rej_reason = ", ".join(str(b) for b in blockers)
        else:
            rej_reason = str(r.get("reason_for_signal") or "")
        out.append({
            "ticker": match_id,
            "direction": "yes",
            "strike_low": None,
            "strike_high": None,
            "yes_ask_cents": top_ask,
            "no_ask_cents": bot_ask,
            "spread_cents": r.get("spread_cents"),
            "volume": r.get("volume"),
            "open_interest": oi,
            "model_prob_yes": top_prob,
            "raw_model_prob_yes": top_raw,
            "bot_verdict": verdict,
            "rejection_reason": rej_reason,
            "title": display_title,
            "minutes_to_close": None,
            "_yes_label": top_label,
            "_no_label": bot_label,
        })
    return out


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
            # Prefer the Kalshi event-page heading so the active-bets
            # Title column matches what the user sees on click-through.
            # Falls back to the per-side market question for pre-fix
            # rows that don't have event_title stored.
            "_title": p.get("event_title") or p.get("title") or "",
            "title": p.get("event_title") or p.get("title") or "",
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
            # Reconstruct the net-of-fee EV the bot saw at open from
            # (entry_model_prob, entry_market_prob). Same formula the
            # closed-position rollup uses, so the buy-criteria popup
            # on the cross-bot active-bets row shows a real EV figure
            # for sport bots instead of a missing value.
            "expected_ev_at_entry": (
                float(p.get("entry_model_prob") or entry) - entry
                - (_kalshi_fee_cents(int(round(entry * 100)), 1) / 100.0)
            ),
        })
    return out


def closed_positions_for_rollup(sim_state_path: str | None,
                                  limit: int = 100) -> List[Dict[str, Any]]:
    """Project tennis ``closed_positions`` into the shape the standard
    ``_render_bet_history_block`` expects.

    Mapping:
      ticker             ← match_id (= real Kalshi event_ticker)
      side               ← "YES" (tennis always buys YES on the
                            favoured side; the dashboard's outcome
                            badge keys off realized_pnl sign anyway)
      entry_price_cents  ← entry_market_prob × 100
      exit_price_cents   ← exit_market_prob × 100 (hedge exit) or
                            settle_market_prob × 100 (natural settle)
      contracts          ← 1 (tennis uses 1-contract = $1 face value)
      realized_pnl_cents ← realized_pnl × 100
      opened_at / exited_at ← as recorded
      _title             ← the Kalshi-published YES question
      error_type         ← exit_reason from the hedge engine
                            (hedge_pl / hedge_sl) — surfaces on the
                            Outcome column tooltip when present.
    """
    s = load_sim_state(sim_state_path)
    closed = list(s.get("closed_positions") or [])
    # Most recently closed first; honour the caller's limit so the
    # cross-bot history loop doesn't pull thousands of rows from a
    # long-running paper-trade ledger.
    closed.sort(key=lambda c: c.get("closed_at", ""), reverse=True)
    out: List[Dict[str, Any]] = []
    for c in closed[:limit]:
        entry = c.get("entry_market_prob")
        exit_p = (c.get("exit_market_prob")
                    or c.get("settle_market_prob"))
        try:
            entry_cents = (int(round(float(entry) * 100))
                            if entry is not None else None)
        except (TypeError, ValueError):
            entry_cents = None
        try:
            exit_cents = (int(round(float(exit_p) * 100))
                           if exit_p is not None else None)
        except (TypeError, ValueError):
            exit_cents = None
        try:
            realized_cents = int(round(float(c.get("realized_pnl", 0)) * 100))
        except (TypeError, ValueError):
            realized_cents = 0
        # Recover Entry EV from what the bot recorded at open. Tennis-
        # style sim state doesn't persist the dashboard's per-row EV,
        # but it does record (entry_model_prob, entry_market_prob), so
        # we can reconstruct the same net-of-fee figure the watchlist
        # column shows for open bets:
        #     EV = entry_model_prob − entry_market_prob − fee_at_entry
        # (no half-spread term — sport bots don't store the bid-ask
        # spread on the closed-position record.)
        entry_model_p = c.get("entry_model_prob")
        try:
            entry_model_p = (float(entry_model_p)
                             if entry_model_p is not None else None)
            entry_market_p = float(entry) if entry is not None else None
        except (TypeError, ValueError):
            entry_model_p = entry_market_p = None
        if (entry_model_p is not None and entry_market_p is not None
                and entry_cents is not None):
            fee_d = _kalshi_fee_cents(entry_cents, 1) / 100.0
            expected_ev = entry_model_p - entry_market_p - fee_d
        else:
            expected_ev = None
        out.append({
            "ticker": c.get("match_id"),
            # Prefer the Kalshi event-page heading so the History tab
            # title matches what the user sees on click-through. Falls
            # back to the per-side market question for pre-fix rows.
            "_title": c.get("event_title") or c.get("title", ""),
            "side": "YES",
            "entry_price_cents": entry_cents,
            "exit_price_cents": exit_cents,
            "contracts": 1,
            "realized_pnl_cents": realized_cents,
            "opened_at": c.get("opened_at", ""),
            "exited_at": c.get("closed_at", ""),
            "error_type": c.get("exit_reason"),
            "model_yes_prob_at_entry": c.get("entry_model_prob"),
            "kalshi_yes_prob_at_entry": entry,
            "expected_ev_at_entry": expected_ev,
            "break_even_probability": entry,
            # > 1 when multiple flap-trades on this match were
            # collapsed into this row by the dedupe pass.
            "merged_trade_count": int(c.get("merged_trade_count", 1) or 1),
            "merged_position_ids": c.get("merged_position_ids"),
        })
    return out


def summary_for_rollup(sim_state_path: str | None) -> Dict[str, Any]:
    """Tennis summary in the shape the cross-bot rollup expects.
    Cents conversion: tennis stake is dollars (1.0 = $1) → ×100 for cents.

    Tennis convention: each paper bet is 1 contract face-value
    (``active_bets_for_rollup`` returns contracts=1, entry in cents
    from the market prob × 100). The active-bets totals mirror that
    so the Home-tab summary cards agree with the rendered table.
    """
    from .dashboard import kalshi_fee_cents
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
    active_contracts = 0
    active_money_spent_cents = 0
    potential_gain_cents = 0
    for p in open_positions:
        entry = p.get("entry_market_prob")
        if entry is None:
            continue
        entry_c = int(round(float(entry) * 100))
        ctr = 1  # tennis paper bet face-value matches the table renderer
        fee_c = kalshi_fee_cents(entry_c, ctr)
        active_contracts += ctr
        active_money_spent_cents += entry_c * ctr + fee_c
        potential_gain_cents += (100 - entry_c) * ctr - fee_c
    return {
        "open_count": len(open_positions),
        "active_contracts": active_contracts,
        "period_bets_made": int(stats.get("total_closed", 0)) + len(open_positions),
        "period_net_pnl_cents": realized_pnl_cents,
        "period_wins": int(stats.get("wins", 0)),
        "period_losses": int(stats.get("losses", 0)),
        "period_money_spent_cents": money_spent_cents,
        "period_money_gained_cents": money_gained_cents,
        "potential_gain_cents": potential_gain_cents,
        "active_money_spent_cents": active_money_spent_cents,
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
    """Edge in percentage points. Zero / missing → plain "0" so dense
    tables don't render a sea of "—" for low-edge strikes."""
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
        ("models", "Models", "?bot=tennis&tab=models"),
        # History routes to the cross-bot history page, where the
        # Kalshi-sourced tennis section renders at the top.
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


def _render_current_prediction(metrics: dict, sim_state: dict,
                                  sim_state_path: str | None = None) -> str:
    """Tennis equivalent of the standard 'Current prediction' card row.

    Six small cards, same layout as the gas / NBA / CPI watchlists.
    Surfaces the model's held-out metrics (Accuracy / F1 / ROC AUC /
    Brier) PLUS real-money trading stats.

    The Realized P&L / Open / win rate cards used to read from
    ``sim_state.stats``, which:
      * inflated wins (the pre-fix hedge_pl/hedge_sl bookkeeping
        credited closes that never actually sold on Kalshi)
      * included canceled-order phantom "settles"
      * disagreed with /portfolio/settlements by ~3× on win rate
    Now we drive the same cards off the Kalshi-derived rows the
    History tab uses (settlements + fills), so every panel on the
    page tells the same true story.
    """
    blended = metrics.get("blended") or {}

    # Real numbers from Kalshi settlements + fills, when reachable.
    # Falls back to sim_state stats when the Kalshi pull fails (e.g.
    # no creds in this venv) so the page never goes blank.
    kalshi_rows: list[dict] = []
    try:
        kalshi_rows = build_tennis_history_for_page(sim_state)
    except Exception:  # noqa: BLE001
        log.exception("build_tennis_history_for_page failed; "
                       "falling back to sim_state stats")

    if kalshi_rows:
        roll = compute_tennis_history_rollup(kalshi_rows)
        realized_cents = int(roll.get("period_net_pnl_cents") or 0)
        wins = int(roll.get("period_wins") or 0)
        losses = int(roll.get("period_losses") or 0)
        n_closed = wins + losses
        win_rate = (wins / n_closed) if n_closed > 0 else None
        # Open count comes from sim_state (live position file) — that's
        # the live state, not historical.
        open_count = int((sim_state.get("stats") or {}).get("open_count") or 0)
        stake_dollars = (
            (roll.get("period_money_spent_cents") or 0) / 100.0
        )
        roi = (realized_cents / 100.0) / stake_dollars if stake_dollars > 0 else None
    else:
        # Fallback — sim_state numbers are stale/inflated but a value
        # is better than blank when Kalshi is unreachable.
        stats = sim_state.get("stats") or {}
        realized_cents = int(round(
            float(stats.get("total_realized_pnl") or 0) * 100
        ))
        win_rate = stats.get("win_rate")
        open_count = int(stats.get("open_count") or 0)
        n_closed = int(stats.get("total_closed") or 0)
        wins = int(stats.get("wins") or 0)
        roi = stats.get("roi")

    pnl_cls = ("green" if realized_cents > 0
                else "red" if realized_cents < 0 else "gray")
    win_rate_str = ("—" if win_rate is None
                     else f"{float(win_rate) * 100:.0f}%")
    win_cls = ("green" if win_rate is not None and win_rate >= 0.5
                else "red" if win_rate is not None else "gray")
    cards = [
        ("Accuracy",         _fmt_pct(blended.get("accuracy"), 1), ""),
        ("F1",               _fmt_pct(blended.get("f1"), 1), ""),
        ("ROC AUC",          _fmt_pct(blended.get("roc_auc"), 1), ""),
        ("Brier",            f"{blended.get('brier', 0):.3f}"
                              if blended.get("brier") is not None else "—",
                              "lower better"),
        ("Open paper bets",  str(open_count), ""),
        ("Realized P&L",     _fmt_signed_dollars(realized_cents), pnl_cls),
    ]
    # Tack on the win-rate card so the cards row directly shows how
    # the model is doing on real money. (Closed bet count goes in the
    # label so we don't blow the layout up.)
    cards.append(
        (f"Real win rate", win_rate_str,
         win_cls if win_cls in ("green", "red") else "")
    )
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


def _tennis_verdict_badge(row: dict, held_side: str | None) -> str:
    """Verdict badge in the standard dashboard vocabulary.

    Mirrors the sim.db-bot watchlist renderer in ``dashboard.py``:

      HOLDING YES / HOLDING NO  — sim_state has an open paper position
                                   on this match (side = PLAYER_A/B)
      BUY YES / BUY NO          — every BUY gate has cleared; the
                                   simulator will fire on this row
      SKIP                      — no positive EV (or no quote and the
                                   model is on the wrong side)
      WATCH                     — positive EV but a gate failed (thin
                                   book, wide spread, volatility), OR
                                   the row has no Kalshi quote yet.

    On a tennis match the two Kalshi YES sides correspond to the two
    players: YES on player_a = bet that A wins; NO (= YES on the other
    ticker) = bet that B wins. We map buy_side accordingly so the
    column reads the same way as NBA — BUY YES means "buy the YES
    contract on the ticker we display", BUY NO means "buy the NO
    side of that same ticker (= YES on the opponent)".
    """
    if held_side == "PLAYER_A":
        return ("<span class='badge badge-yes' "
                "title='You are holding YES on player_a'>HOLDING YES</span>")
    if held_side == "PLAYER_B":
        return ("<span class='badge badge-no' "
                "title='You are holding NO on player_a (YES on player_b)'>"
                "HOLDING NO</span>")
    eligible = bool(row.get("buy_eligible"))
    side = row.get("buy_side")  # "A" / "B" / None
    ev = row.get("buy_side_ev")
    blockers = row.get("buy_blockers") or []

    if eligible and side in ("A", "B"):
        cls = "badge-yes" if side == "A" else "badge-no"
        label = "BUY YES" if side == "A" else "BUY NO"
        return f"<span class='badge {cls}'>{label}</span>"

    # No quote yet → watching for entry, but model-only forecast.
    if not blockers or blockers == ["no quoted market"]:
        if row.get("market_prob_a") is None:
            return ("<span class='badge badge-hedge' "
                    "title='No Kalshi quote yet — model-only forecast'>"
                    "WATCH</span>")
        # Quoted but no blockers AND not eligible — shouldn't happen,
        # but fall through to SKIP.
        return "<span class='badge badge-skip'>SKIP</span>"

    # Quoted + at least one gate failed. If EV is non-positive on the
    # model's favoured side, this is a SKIP (no edge to chase). If EV
    # is positive but a gate (thin book / wide spread / volatility)
    # blocks the trade, this is a WATCH — keep an eye on it.
    tip = "Blocked by gate(s): " + ", ".join(blockers)
    if ev is None or ev <= 0:
        return (f"<span class='badge badge-skip' "
                f"title='{html.escape(tip)}'>SKIP</span>")
    return (f"<span class='badge badge-hedge' "
            f"title='{html.escape(tip)}'>WATCH</span>")


def _render_watchlist_table(payload: dict,
                              sim_state: dict | None = None) -> str:
    """Tennis matches table.

    Rows are clickable — the page's JS hook listens for clicks on the
    table body and updates the projected-forecast graph above with the
    row's probabilities. Each row carries a ``data-mid`` attribute so
    the JS can look up the match in the embedded forecast payload.

    Only rows with at least one open contract (``open_interest > 0``)
    are rendered — matches with no tradeable depth would clutter the
    table without giving the user anything to act on. Unquoted upcoming
    matches and zero-OI rows are filtered out.
    """
    raw_rows = payload.get("rows") or []
    if not raw_rows:
        return "<div class='empty'>No active tennis markets.</div>"
    rows_all = [
        r for r in raw_rows
        if r.get("open_interest") is not None
        and float(r.get("open_interest") or 0) > 0
    ]
    if not rows_all:
        return (f"<div class='empty'>No tennis matches with open "
                f"contracts right now — {len(raw_rows)} match"
                f"{'es' if len(raw_rows) != 1 else ''} pulled from Kalshi, "
                f"none with depth on the book.</div>")
    # Map of match_id → "PLAYER_A"/"PLAYER_B" for rows we already hold
    # paper positions on. Feeds the HOLDING YES / HOLDING NO verdict.
    held_sides: dict[str, str] = {}
    for p in ((sim_state or {}).get("open_positions") or []):
        mid = str(p.get("match_id") or "")
        if mid:
            held_sides[mid] = str(p.get("side") or "")
    # Sort: BUY-eligible first (by buy_score desc), then quoted rows by
    # |edge|, then unquoted upcoming matches.
    def _sort_key(r):
        eligible = bool(r.get("buy_eligible"))
        quoted = r.get("market_prob_a") is not None
        score = float(r.get("buy_score") or 0)
        edge_mag = abs(float(r.get("edge_a") or 0))
        return (
            0 if eligible else (1 if quoted else 2),
            -score, -edge_mag,
        )
    rows_sorted = sorted(rows_all, key=_sort_key)

    # Column shape matches the NBA watchlist exactly: Ticker | Title |
    # Side | Contracts | Kalshi % | My % | Edge | EV | Verdict. Title
    # carries the Kalshi-published YES question (same field the NBA
    # watchlist surfaces), Side carries the favoured player + opponent.
    #
    # Wrapped in a ``.watchlist-scroll`` container — same scroll
    # behaviour as the standard watchlist so the 300+ ITF / Challenger
    # rows don't push the rest of the page down.
    out = ["<div class='watchlist-scroll'>",
           "<table id='tennis-watchlist-table'>",
           "<thead><tr>"
           "<th>Ticker</th>"
           "<th title='Kalshi-published contract title — the YES question shown on the market page.'>Title</th>"
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

        # Standard verdict vocabulary — BUY YES / BUY NO / HOLDING YES /
        # HOLDING NO / SKIP / WATCH — using the same badge classes the
        # sim.db-bot watchlist uses.
        held_side = held_sides.get(str(r.get("match_id") or ""))
        verdict_pill = _tennis_verdict_badge(r, held_side)

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
        # Title cell — Kalshi-published YES question for the side the
        # bot favours, falling back to the row's pre-computed title.
        title_text = (r.get("title_a") if edge_a >= 0 else r.get("title_b")) \
                       or r.get("title") or ""
        title_cell = (
            f"<td title='{html.escape(str(title_text))}' "
            f"style='max-width:340px;'>"
            f"<span class='small gray' style='display:block;overflow:hidden;"
            f"text-overflow:ellipsis;white-space:nowrap;'>"
            f"{html.escape(str(title_text))}</span></td>"
        )
        # Row classes match the standard sim.db-bot watchlist:
        #   row-bought bought-yes/no  → we already own this match
        #   row-suspect              → quoted but a gate failed (SKIP/WATCH)
        #   no special class          → BUY-eligible OR unquoted upcoming
        row_classes = ["tennis-row"]
        row_title = ""
        if held_side == "PLAYER_A":
            row_classes += ["row-bought", "bought-yes"]
        elif held_side == "PLAYER_B":
            row_classes += ["row-bought", "bought-no"]
        elif not r.get("buy_eligible") and r.get("market_prob_a") is not None:
            row_classes.append("row-suspect")
            blockers = r.get("buy_blockers") or []
            if blockers:
                row_title = (" title='Blocked by gate(s): "
                              + html.escape(", ".join(blockers)) + "'")
        row_cls = " ".join(row_classes)
        out.append(
            f"<tr class='{row_cls}' data-mid='{html.escape(mid)}' "
            f"style='cursor:pointer'{row_title}>"
            f"<td class='mono small'>{ticker_cell}</td>"
            f"{title_cell}"
            f"<td>{side_html}</td>"
            f"<td class='num'>{oi_str}</td>"
            f"{kalshi_cell}"
            f"{my_cell}"
            f"{edge_cell}"
            f"{ev_cell}"
            f"<td>{verdict_pill}</td>"
            "</tr>"
        )
    out.append("</tbody></table></div>")
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
        # Empty state is already covered by the "Active paper bets"
        # section above and the "No tradeable tennis markets" copy on
        # the matches table below. Rendering another "No active bets
        # right now." here just duplicated the message — return an
        # empty string so the chart slot collapses cleanly.
        return ""
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


# --------------------------------------------------------------------------- #
# History page — sourced directly from Kalshi's settlements API                #
# --------------------------------------------------------------------------- #

# Cached Kalshi client + settlements/fills lists. Settlements and
# fills are immutable once written, so a 60-second cache is just
# politeness to the portfolio endpoint, not a freshness compromise.
_SETTLEMENTS_CACHE: dict[str, Any] = {"at": 0.0, "rows": []}
_FILLS_CACHE: dict[str, Any] = {"at": 0.0, "rows": []}
_SETTLEMENTS_TTL_S = 60.0
_KALSHI_CLIENT: Any = None


def _get_kalshi_client():
    """Lazy Kalshi client init — same env-var auth the live executor uses."""
    global _KALSHI_CLIENT
    if _KALSHI_CLIENT is not None:
        return _KALSHI_CLIENT
    import os
    try:
        from kalshi_sdk import KalshiClient
    except ImportError:
        log.warning("kalshi_sdk unavailable; tennis history will be empty")
        return None
    api = os.environ.get("KALSHI_API_KEY_ID", "").strip()
    pkey = os.environ.get("KALSHI_PRIVATE_KEY_PATH", "").strip()
    if not api or not pkey:
        log.warning("KALSHI_API_KEY_ID / KALSHI_PRIVATE_KEY_PATH unset; "
                     "tennis history will be empty")
        return None
    try:
        _KALSHI_CLIENT = KalshiClient(api_key_id=api, private_key_path=pkey)
    except Exception:  # noqa: BLE001
        log.exception("KalshiClient init failed; tennis history will be empty")
        return None
    return _KALSHI_CLIENT


def _fetch_tennis_settlements(force: bool = False) -> list[dict]:
    """All Kalshi settlements where the ticker starts with KXATPMATCH or
    KXWTAMATCH. Cached for ``_SETTLEMENTS_TTL_S`` to keep the History tab
    cheap; settlements never change after they land, so even a longer TTL
    would be safe — 60s is just paranoia."""
    import time
    now = time.time()
    if not force and now - _SETTLEMENTS_CACHE["at"] < _SETTLEMENTS_TTL_S:
        return _SETTLEMENTS_CACHE["rows"]
    client = _get_kalshi_client()
    if client is None:
        return []
    try:
        atp = client.iter_settlements(ticker_prefix="KXATPMATCH")
        wta = client.iter_settlements(ticker_prefix="KXWTAMATCH")
    except Exception:  # noqa: BLE001
        log.exception("settlements fetch failed")
        return _SETTLEMENTS_CACHE["rows"]  # serve stale on error
    rows = atp + wta
    _SETTLEMENTS_CACHE.update({"at": now, "rows": rows})
    return rows


def _fetch_tennis_fills(force: bool = False) -> list[dict]:
    """All Kalshi fills on KX[ATP|WTA]MATCH tickers. Same caching
    pattern as settlements."""
    import time
    now = time.time()
    if not force and now - _FILLS_CACHE["at"] < _SETTLEMENTS_TTL_S:
        return _FILLS_CACHE["rows"]
    client = _get_kalshi_client()
    if client is None:
        return []
    try:
        atp = client.iter_fills(ticker_prefix="KXATPMATCH")
        wta = client.iter_fills(ticker_prefix="KXWTAMATCH")
    except Exception:  # noqa: BLE001
        log.exception("fills fetch failed")
        return _FILLS_CACHE["rows"]
    rows = atp + wta
    _FILLS_CACHE.update({"at": now, "rows": rows})
    return rows


def _summarize_fills(fills: list[dict]) -> dict[str, dict]:
    """Group fills by ticker, splitting OPEN (action=buy) from CLOSE
    (action=sell) legs. The earlier shape summed both into a single
    weighted average, which double-counted contracts and blended the
    entry and exit prices into a meaningless midpoint — e.g.
    bought-61¢/closed-60¢ rendered as "50¢ entry · 2 contracts".

    All prices are normalized to yes-equivalent. For a "sell no @ 0.40"
    close fill, yes_p = 0.60 is the effective price at which the YES
    position was offset; using yes_p uniformly across buy/sell fills
    puts entry and exit on the same axis.
    """
    by_ticker: dict[str, dict] = {}
    for f in fills:
        t = f.get("ticker") or f.get("market_ticker") or ""
        if not t:
            continue
        n = float(f.get("count_fp") or 0)
        if n <= 0:
            continue
        action = (f.get("action") or "").lower()
        side = f.get("side") or "yes"
        yes_p = float(f.get("yes_price_dollars") or 0)
        d = by_ticker.setdefault(t, {
            "ticker": t,
            "_open_n_sum": 0.0,
            "_open_yesp_x_n_sum": 0.0,
            "_close_n_sum": 0.0,
            "_close_yesp_x_n_sum": 0.0,
            "is_taker": False,
            "side": side,
            "order_id": f.get("order_id"),
            "first_fill_time": f.get("created_time") or "",
            "fee_sum_dollars": 0.0,
        })
        if action == "sell":
            d["_close_n_sum"] += n
            d["_close_yesp_x_n_sum"] += yes_p * n
        else:
            # action=buy (or missing/unknown — treat as open for back-
            # compat so the row doesn't disappear). The opened side
            # comes from the first buy fill seen.
            if d["_open_n_sum"] == 0:
                d["side"] = side
                d["order_id"] = f.get("order_id") or d["order_id"]
            d["_open_n_sum"] += n
            d["_open_yesp_x_n_sum"] += yes_p * n
        d["fee_sum_dollars"] += float(f.get("fee_cost") or 0)
        if f.get("is_taker"):
            d["is_taker"] = True
        ct = f.get("created_time") or ""
        if ct and (not d["first_fill_time"] or ct < d["first_fill_time"]):
            d["first_fill_time"] = ct
    out: dict[str, dict] = {}
    for t, d in by_ticker.items():
        open_n = d["_open_n_sum"]
        close_n = d["_close_n_sum"]
        if open_n <= 0 and close_n <= 0:
            continue
        avg_open = (d["_open_yesp_x_n_sum"] / open_n) if open_n > 0 else None
        avg_close = (d["_close_yesp_x_n_sum"] / close_n) if close_n > 0 else None
        out[t] = {
            "ticker": t,
            "avg_price_dollars": avg_open,
            "contracts_filled": open_n,
            "close_avg_price_dollars": avg_close,
            "close_contracts": close_n,
            "closed_pre_settlement": close_n > 0,
            "is_taker": d["is_taker"],
            "side": d["side"],
            "order_id": d["order_id"],
            "first_fill_time": d["first_fill_time"],
            "fee_sum_dollars": d["fee_sum_dollars"],
        }
    return out


# Player → country (IOC 3-letter) lookup. Populated lazily from
# tennis-forecast's matches_clean.csv on first History-tab render and
# cached for the life of the process — the file only changes on
# retrain (~once a day) and re-reading it is cheap.
_PLAYER_IOC_CACHE: dict[str, dict[str, str]] = {"by_path": {}}


# IOC (Olympic) codes → ISO-3166-alpha-2. The two diverge on a few
# dozen countries — the ones below cover every IOC code that appears
# in the Sackmann ATP/WTA panel since 2015. Unmapped codes render
# with no flag (defensive: better empty than wrong).
_IOC_TO_ISO2: dict[str, str] = {
    # Common ATP/WTA countries — alphabetical for grep-ability.
    "ALG": "DZ", "ANG": "AO", "ARG": "AR", "ARM": "AM", "AUS": "AU",
    "AUT": "AT", "AZE": "AZ", "BAH": "BS", "BAN": "BD", "BAR": "BB",
    "BEL": "BE", "BIH": "BA", "BLR": "BY", "BOL": "BO", "BOT": "BW",
    "BRA": "BR", "BRN": "BH", "BUL": "BG", "BUR": "BF", "CAM": "KH",
    "CAN": "CA", "CHI": "CL", "CHN": "CN", "CIV": "CI", "CMR": "CM",
    "COD": "CD", "COL": "CO", "CRC": "CR", "CRO": "HR", "CUB": "CU",
    "CYP": "CY", "CZE": "CZ", "DEN": "DK", "DMA": "DM", "DOM": "DO",
    "ECU": "EC", "EGY": "EG", "ESA": "SV", "ESP": "ES", "EST": "EE",
    "ETH": "ET", "FIJ": "FJ", "FIN": "FI", "FRA": "FR", "GBR": "GB",
    "GEO": "GE", "GER": "DE", "GHA": "GH", "GRE": "GR", "GRN": "GD",
    "GUA": "GT", "HAI": "HT", "HKG": "HK", "HON": "HN", "HUN": "HU",
    "INA": "ID", "IND": "IN", "IRI": "IR", "IRL": "IE", "ISL": "IS",
    "ISR": "IL", "ITA": "IT", "JAM": "JM", "JOR": "JO", "JPN": "JP",
    "KAZ": "KZ", "KEN": "KE", "KGZ": "KG", "KOR": "KR", "KSA": "SA",
    "KUW": "KW", "LAT": "LV", "LBA": "LY", "LBN": "LB", "LCA": "LC",
    "LIB": "LB", "LIE": "LI", "LTU": "LT", "LUX": "LU", "MAD": "MG",
    "MAR": "MA", "MAS": "MY", "MDA": "MD", "MEX": "MX", "MGL": "MN",
    "MKD": "MK", "MLT": "MT", "MNE": "ME", "MON": "MC", "NCA": "NI",
    "NED": "NL", "NEP": "NP", "NGR": "NG", "NOR": "NO", "NZL": "NZ",
    "OMA": "OM", "PAK": "PK", "PAN": "PA", "PAR": "PY", "PER": "PE",
    "PHI": "PH", "POL": "PL", "POR": "PT", "PRK": "KP", "PUR": "PR",
    "QAT": "QA", "ROU": "RO", "RSA": "ZA", "RUS": "RU", "SEN": "SN",
    "SEY": "SC", "SIN": "SG", "SLO": "SI", "SMR": "SM", "SRB": "RS",
    "SRI": "LK", "SUI": "CH", "SVK": "SK", "SWE": "SE", "SYR": "SY",
    "TAH": "PF", "TAN": "TZ", "THA": "TH", "TJK": "TJ", "TKM": "TM",
    "TOG": "TG", "TPE": "TW", "TRI": "TT", "TUN": "TN", "TUR": "TR",
    "UAE": "AE", "UGA": "UG", "UKR": "UA", "URU": "UY", "USA": "US",
    "UZB": "UZ", "VEN": "VE", "VIE": "VN", "ZAM": "ZM", "ZIM": "ZW",
}


def _flag_emoji(iso2: str) -> str:
    """Two-letter ISO-3166-alpha-2 → flag emoji (regional indicator
    pair). Returns empty string for invalid input."""
    if not iso2 or len(iso2) != 2 or not iso2.isalpha():
        return ""
    cc = iso2.upper()
    return chr(0x1F1E6 + (ord(cc[0]) - ord("A"))) + chr(0x1F1E6 + (ord(cc[1]) - ord("A")))


def _player_country_map(sim_state_path: str | None) -> dict[str, str]:
    """Build a {full_name: IOC} lookup from tennis-forecast's
    matches_clean.csv. The file lives at
    ``<sim_state's grandparent>/processed/matches_clean.csv`` per the
    tennis-forecast repo layout; we derive the path rather than
    threading another config key through.

    Cached per-path at module load. Returns ``{}`` if the file is
    missing — the History renderer treats absent IOC as "no flag".
    """
    if not sim_state_path:
        return {}
    cache = _PLAYER_IOC_CACHE["by_path"]
    if sim_state_path in cache:
        return cache[sim_state_path]
    p = Path(sim_state_path)
    # outputs-live/sim_state.json → data → processed/matches_clean.csv
    candidates = [
        p.parent.parent / "processed" / "matches_clean.csv",
        p.parent.parent.parent / "processed" / "matches_clean.csv",
    ]
    matches_csv = next((c for c in candidates if c.exists()), None)
    if matches_csv is None:
        log.warning("matches_clean.csv not found near %s; flags disabled",
                     sim_state_path)
        cache[sim_state_path] = {}
        return {}
    import csv
    out: dict[str, str] = {}
    try:
        with open(matches_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Skip 'None' string and empties.
                wn, wi = row.get("winner_name"), row.get("winner_ioc")
                ln, li = row.get("loser_name"), row.get("loser_ioc")
                if wn and wi and wi != "None" and wn not in out:
                    out[wn] = wi
                if ln and li and li != "None" and ln not in out:
                    out[ln] = li
    except (OSError, csv.Error):
        log.exception("failed to read %s; flags disabled", matches_csv)
    log.info("loaded %d player→IOC entries from %s", len(out), matches_csv)
    cache[sim_state_path] = out
    return out


def _tour_badge(ticker: str) -> str:
    """ATP/WTA badge HTML — Kalshi-style coloured pill on the leftmost
    column of the History table. Derived from the Kalshi series prefix
    (KXATPMATCH-* → ATP, KXWTAMATCH-* → WTA)."""
    t = (ticker or "").upper()
    if t.startswith("KXATPMATCH"):
        return ("<span style='display:inline-block;padding:2px 8px;"
                "font-size:10px;font-weight:700;letter-spacing:0.5px;"
                "color:#fff;background:#1d3a8a;border-radius:4px;'>"
                "ATP</span>")
    if t.startswith("KXWTAMATCH"):
        return ("<span style='display:inline-block;padding:2px 8px;"
                "font-size:10px;font-weight:700;letter-spacing:0.5px;"
                "color:#fff;background:#15803d;border-radius:4px;'>"
                "WTA</span>")
    return ""


def _player_with_flag(name: str, ioc_lookup: dict[str, str]) -> str:
    """HTML-safe player cell: flag emoji + name. Returns the name
    alone (no leading flag) when the player has no IOC entry."""
    if not name:
        return ""
    safe_name = html.escape(name)
    ioc = ioc_lookup.get(name)
    if not ioc:
        return safe_name
    iso2 = _IOC_TO_ISO2.get(ioc.upper())
    if not iso2:
        return safe_name
    flag = _flag_emoji(iso2)
    if not flag:
        return safe_name
    return f"{flag} {safe_name}"


def _format_fill_time(iso: str) -> str:
    """Format a fill timestamp like Kalshi's history row:
    ``May 26, 2026 · 9:58:10AM EDT``. Best-effort — falls back to the
    raw string if parsing fails."""
    if not iso:
        return "—"
    try:
        # Kalshi gives UTC ISO; convert to ET for display (matches
        # Kalshi's UI). Use zoneinfo, fallback to UTC label.
        from datetime import datetime
        try:
            from zoneinfo import ZoneInfo
            tz = ZoneInfo("America/New_York")
            label_tz = ""
        except Exception:
            tz = timezone.utc
            label_tz = "UTC"
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00")).astimezone(tz)
        if not label_tz:
            label_tz = dt.tzname() or "EDT"
        return dt.strftime(f"%b %-d, %Y · %-I:%M:%S%p {label_tz}")
    except Exception:
        return iso[:19].replace("T", " ")


def _join_settlement_with_sim_state(s: dict, sim_state: dict,
                                       fills_by_ticker: dict[str, dict],
                                       ) -> dict | None:
    """Enrich one settlement row with sim_state (model prob, full
    player name) and Kalshi fills (avg price, contracts filled, order
    type, fill time). Returns ``None`` for phantom settlements where
    we held 0 contracts at settle AND have no recorded fill — the
    BUBSTR-style "canceled order" record that Kalshi still surfaces.
    """
    ticker = s.get("ticker") or ""
    event_ticker = s.get("event_ticker") or ""

    # Side we held + count, sourced from BOTH Kalshi's settlement
    # row (count at settle time) AND fills (actually executed). A
    # cancelled order has fills=0 contracts; an order that filled
    # but was later disposed of (we don't sell, so this shouldn't
    # happen, but defensive) would have fills>0 and count=0.
    yes_n_settle = float(s.get("yes_count_fp") or 0)
    no_n_settle = float(s.get("no_count_fp") or 0)
    fill_sum = fills_by_ticker.get(ticker)
    fill_contracts = float(fill_sum["contracts_filled"]) if fill_sum else 0.0
    fill_side = (fill_sum or {}).get("side", "yes")
    # Final position = what we held at settle (Kalshi's view). Use
    # fills as a tiebreak when both settle counts are zero — that's
    # the canceled-order phantom we want to filter out.
    if yes_n_settle > 0 or no_n_settle > 0:
        side_held = "yes" if yes_n_settle > 0 else "no"
        contracts = yes_n_settle if side_held == "yes" else no_n_settle
    elif fill_contracts > 0:
        # We filled then disposed — shouldn't happen in our bot.
        # Render the row with fill-derived side; settlement value
        # is whatever Kalshi paid (likely 0 since we didn't hold).
        side_held = fill_side
        contracts = fill_contracts
    else:
        # Pure phantom — Kalshi shows a settlement row but we never
        # held or filled. Skip entirely so the History tab doesn't
        # mis-attribute a "win" or "loss" to it.
        return None

    yes_total_cost = float(s.get("yes_total_cost_dollars") or 0)
    no_total_cost = float(s.get("no_total_cost_dollars") or 0)
    # Kalshi's settlement record already aggregates fees across all
    # fills for this market (verified empirically). Fall back to the
    # per-fill sum only when settlement omits it.
    fee = float(s.get("fee_cost") or 0)
    if fee == 0 and fill_sum:
        fee = float(fill_sum.get("fee_sum_dollars") or 0)
    # Offset-closed: bot opened YES then closed via a NO-leg trade.
    # Kalshi books both legs as held to settlement; cost basis is the
    # sum of both legs and payout is $1 only on the winning side
    # (yields the real ~1¢ spread loss/gain rather than the +$0.36
    # phantom win the side_held-only formula produced).
    offset_closed = (yes_n_settle > 0 and no_n_settle > 0)
    if offset_closed:
        total_cost_basis = yes_total_cost + no_total_cost
        mr = s.get("market_result")
        yes_payout = yes_n_settle if mr == "yes" else 0.0
        no_payout = no_n_settle if mr == "no" else 0.0
        total_payout = yes_payout + no_payout
        # Display contracts = opened size (1 here), not yes+no (2).
        if fill_sum and fill_sum.get("contracts_filled"):
            contracts = float(fill_sum["contracts_filled"])
    else:
        total_cost_basis = (yes_total_cost if side_held == "yes"
                             else no_total_cost)
        won_side = (s.get("market_result") == side_held)
        total_payout = contracts if won_side else 0.0
    total_cost_with_fee = total_cost_basis + fee
    total_return = total_payout - total_cost_with_fee
    # "won" reflects net P&L sign (downstream "WIN/LOSS" badge + the
    # win-rate aggregate). For offset-closed rows with ~zero spread
    # the row will fall on the loss side of the badge due to fees,
    # which matches the real economic outcome.
    won = (total_return > 0)
    pct_return = (total_return / total_cost_with_fee
                   if total_cost_with_fee > 0 else 0.0)

    # Sim-state join for model prob + matchup + player name.
    closed = (sim_state or {}).get("closed_positions") or []
    # Primary join: side-specific ticker. Falls back to the event
    # ticker (= ``match_id`` in the upstream sim simulator, which
    # historically didn't persist the side-specific ticker) so the
    # 'My prob' / Predicted-winner columns aren't blank for the
    # 385-ish older sim rows recorded before the ticker field was
    # added. Side disambiguation uses ``side_player`` matching the
    # ticker's trailing 3-letter suffix when both records exist.
    sim_row = next((c for c in closed if c.get("ticker") == ticker), None)
    if sim_row is None and event_ticker:
        tri = ticker.rsplit("-", 1)[-1] if "-" in ticker else ""
        for c in closed:
            if c.get("match_id") != event_ticker:
                continue
            # If both legs of an event appear, prefer the row whose
            # side_player initials match the side-ticker's trailing
            # tricode (e.g. ``-NAV`` -> "Yunchaokete Bu" rejected,
            # "Cristian Garin" accepted when its tricode is NAV).
            sp = (c.get("side_player") or "").upper()
            if tri and sp and not any(part.startswith(tri[:2]) or part[:3] == tri
                                        for part in sp.split()):
                sim_row = sim_row or c  # weak match — keep as fallback
                continue
            sim_row = c
            break
    side_player = ""
    matchup = ""
    tournament = ""
    entry_model_prob = None
    if sim_row:
        side_player = sim_row.get("side_player") or ""
        pa = sim_row.get("player_a") or ""
        pb = sim_row.get("player_b") or ""
        matchup = (f"{pa} vs {pb}" if pa and pb
                    else (sim_row.get("event_title") or ""))
        tournament = sim_row.get("tournament") or ""
        entry_model_prob = sim_row.get("entry_model_prob")
    if not side_player:
        parts = ticker.rsplit("-", 1)
        if len(parts) == 2:
            side_player = parts[1]
    if not matchup:
        ev_parts = event_ticker.rsplit("-", 1)
        if len(ev_parts) == 2:
            matchup = ev_parts[1]

    tour = "ATP" if ticker.startswith("KXATPMATCH") else "WTA"
    sex = "Men" if tour == "ATP" else "Women"
    series_label = (f"{tournament} {sex} Singles" if tournament
                    else f"{tour} Singles")
    final_position_label = (
        f"{int(round(contracts))} {'Yes' if side_held == 'yes' else 'No'}"
    )

    # Fill-derived columns. When no fill is recorded (sim-state
    # phantom or canceled order), show em-dashes rather than fake
    # numbers.
    avg_price_dollars: float | None = None
    contracts_filled: float | None = None
    order_type_label = "—"
    fill_time_label = "—"
    if fill_sum:
        # avg_price_dollars is now the BUY-only average (None if there
        # are no recorded buy fills — edge case).
        raw_avg = fill_sum.get("avg_price_dollars")
        avg_price_dollars = float(raw_avg) if raw_avg is not None else None
        raw_n = fill_sum.get("contracts_filled")
        contracts_filled = float(raw_n) if raw_n else None
        taker_label = "Taker" if fill_sum.get("is_taker") else "Maker"
        if avg_price_dollars is not None:
            order_type_label = (
                f"Limit {int(round(avg_price_dollars*100))}¢ · {taker_label}"
            )
        fill_time_label = _format_fill_time(fill_sum.get("first_fill_time") or "")

    return {
        "ticker": ticker,
        "event_ticker": event_ticker,
        "settled_time": s.get("settled_time") or "",
        "side_player": side_player,
        "matchup": matchup,
        "series_label": series_label,
        "entry_model_prob": entry_model_prob,
        "final_position_label": final_position_label,
        "settlement_payout_dollars": float(s.get("value") or 0) / 100.0,
        "total_cost_dollars": total_cost_with_fee,
        "total_payout_dollars": total_payout,
        "total_return_dollars": total_return,
        "total_return_pct": pct_return,
        "won": won,
        "avg_price_dollars": avg_price_dollars,
        "contracts_filled": contracts_filled,
        "order_type_label": order_type_label,
        "fill_time_label": fill_time_label,
    }


def _merge_with_live_sim_state(primary: dict) -> dict:
    """Return a sim_state whose closed_positions is the union of the
    primary file's records and the live ``outputs-live/sim_state.json``
    snapshot. Used to give the SIM dashboard's Kalshi-driven history
    table access to the live-mode bot's stored per-bet model prob /
    player names — without that, real Kalshi trades render with
    'My prob' = '—' on the sim dashboard because the sim simulator
    has no record of them.

    Lookup is by ``ticker`` first, then ``match_id`` — primary wins
    on collision so the simulation's own paper-trade history still
    takes precedence in sim mode.
    """
    out = dict(primary or {})
    primary_closed = list(out.get("closed_positions") or [])
    seen_tickers = {c.get("ticker") for c in primary_closed if c.get("ticker")}
    seen_match_ids = {c.get("match_id") for c in primary_closed
                      if c.get("match_id")}
    try:
        live_path = Path("/root/tennis-forecast/data/outputs-live/sim_state.json")
        if not live_path.exists():
            return out
        with live_path.open("r", encoding="utf-8") as f:
            live = json.load(f) or {}
    except (OSError, json.JSONDecodeError):
        return out
    extra: list[dict] = []
    for c in (live.get("closed_positions") or []):
        if c.get("ticker") and c["ticker"] in seen_tickers:
            continue
        if c.get("match_id") and c["match_id"] in seen_match_ids:
            continue
        extra.append(c)
    if extra:
        out["closed_positions"] = primary_closed + extra
    return out


def build_tennis_history_for_page(sim_state: dict) -> list[dict]:
    """Return the enriched, phantom-filtered, settled-bet rows that
    drive the History tab. Caller can pass them to the page-level
    renderers (chart, attribution, summary cards) so every block on
    the History tab reflects the same source-of-truth data.

    Returned rows carry the fields ``_render_tennis_history_page``
    reads PLUS the legacy schema fields the existing chart /
    attribution / summary helpers expect, so callers can pass them
    straight through.
    """
    settlements = _fetch_tennis_settlements()
    fills_by_ticker = _summarize_fills(_fetch_tennis_fills())
    # Kalshi settlements are platform-wide (one set per account), but
    # the join with sim_state for per-bet model prob / matchup is
    # mode-specific: the SIM dashboard reads paper trades from
    # ``data/outputs/sim_state.json`` and never sees the live
    # executor's records. Merge in the live sim_state's closed_positions
    # so the sim history table can render the same prob / player names
    # the live page shows for real Kalshi trades. The primary sim_state
    # takes precedence on ticker collisions; live records only fill
    # rows the primary lacks.
    merged = _merge_with_live_sim_state(sim_state)
    rows = [_join_settlement_with_sim_state(s, merged, fills_by_ticker)
            for s in settlements]
    rows = [r for r in rows if r is not None]
    rows.sort(key=lambda r: r.get("settled_time") or "", reverse=True)
    # Decorate with legacy-shape fields so callers can pass the same
    # list to _render_history_chart / _render_history_attribution /
    # _render_summary_cards-via-rollup without further mapping.
    for r in rows:
        r["_bot_key"] = "tennis"
        r["_bot_name"] = "Tennis Forecast"
        r["exited_at"] = r.get("settled_time") or ""
        r["side"] = "YES"  # the live executor only ever buys YES
        r["realized_pnl_cents"] = int(round(
            float(r.get("total_return_dollars") or 0) * 100
        ))
        mp = r.get("entry_model_prob")
        avg_p = r.get("avg_price_dollars")
        if mp is not None and avg_p is not None:
            r["expected_ev_at_entry"] = float(mp) - float(avg_p)
        else:
            r["expected_ev_at_entry"] = None
    return rows


def compute_tennis_history_rollup(rows: list[dict]) -> dict:
    """Build the rollup dict ``_render_summary_cards`` expects, from
    the Kalshi-sourced tennis row list. Only counts CLOSED bets — the
    bot's open positions aren't sourced from settlements so the
    ``active_bets`` field is set to 0 and the caller can overlay the
    live open count if it has one."""
    wins = sum(1 for r in rows if r.get("won"))
    closed = len(rows)
    losses = closed - wins
    net_cents = sum(int(r.get("realized_pnl_cents") or 0) for r in rows)
    spent_cents = int(round(sum(
        float(r.get("total_cost_dollars") or 0) for r in rows
    ) * 100))
    gained_cents = int(round(sum(
        float(r.get("total_payout_dollars") or 0) for r in rows
    ) * 100))
    contracts_bought = int(round(sum(
        float(r.get("contracts_filled") or 0) for r in rows
    )))
    return {
        "period_net_pnl_cents": net_cents,
        "period_wins": wins,
        "period_losses": losses,
        "period_win_pct": (wins / closed) if closed > 0 else 0.0,
        "period_money_spent_cents": spent_cents,
        "period_money_gained_cents": gained_cents,
        "period_contracts_bought": contracts_bought,
        "active_bets": 0,  # caller overlays this from the live state
    }


def _render_tennis_history_page(sim_state: dict,
                                  sim_state_path: str | None = None,
                                  rows: list[dict] | None = None) -> str:
    """The tennis bot's History tab — sourced from Kalshi's
    /portfolio/settlements + /portfolio/fills (source of truth for
    closed bets) and enriched with sim_state's per-trade entry-model
    -prob + full player names.

    Column order (per latest spec):
      Last updated · Market · Player · My probability ·
      Avg price · Contracts filled · Order type ·
      Final position · Settlement payout · Total cost ·
      Total payout · Total return

    Player cell carries the player's country flag prepended (from the
    Sackmann matches_clean.csv → IOC lookup; unmapped players render
    name-only).

    Phantom settlements (canceled-order rows where we held 0 contracts
    at settle AND have no recorded fill) are silently dropped so the
    win/loss column is always honest about a position we actually had.
    """
    ioc_lookup = _player_country_map(sim_state_path)
    if rows is None:
        rows = build_tennis_history_for_page(sim_state)
    if not rows:
        return (
            "<div class='empty'>No Kalshi-settled tennis bets yet — once "
            "real positions on KX[ATP|WTA]MATCH tickers settle, they appear "
            "here directly from Kalshi.</div>"
        )

    out: List[str] = []
    out.append(
        "<table class='tennis-history'><thead><tr>"
        "<th title='ATP (men) or WTA (women) tour.'>Tour</th>"
        "<th title='When the order filled (ET).'>Last updated</th>"
        "<th>Market</th>"
        "<th title='The player our model picked — the YES contract "
        "we bought.'>Predicted winner</th>"
        "<th title='Whoever Kalshi resolved YES on — the player who "
        "actually won.'>Winner</th>"
        "<th class='num' title='Model probability for our predicted "
        "winner at order time.'>My prob</th>"
        "<th class='num' title='Contract-weighted average fill price.'>"
        "Avg price</th>"
        "<th class='num'>Contracts filled</th>"
        "<th class='num'>Final position</th>"
        "<th class='num' title='Total $ paid to enter (price × "
        "contracts + fees).'>Total cost</th>"
        "<th class='num' title='Total $ Kalshi returned at settle "
        "(1 dollar per contract for the winning side; $0 if we "
        "lost).'>Total payout</th>"
        "<th class='num' title='Total payout − total cost.'>"
        "Total return</th>"
        "</tr></thead><tbody>"
    )
    for r in rows:
        mp = r.get("entry_model_prob")
        ret = float(r.get("total_return_dollars") or 0)
        pct = float(r.get("total_return_pct") or 0)
        ret_cls = "green" if ret > 0 else ("red" if ret < 0 else "gray")
        ret_str = (f"{'+' if ret > 0 else ''}${ret:.2f} "
                    f"({'+' if pct > 0 else ''}{pct*100:.0f}%)")
        cost = float(r.get("total_cost_dollars") or 0)
        payout = float(r.get("total_payout_dollars") or 0)
        # Total payout: green if we got paid out, red if zero (lost).
        # Kalshi pays $1 × contracts on a win and $0 on a loss, so
        # binary colouring tracks the win/loss flag directly.
        payout_cls = "green" if payout > 0 else "red"
        avg_p = r.get("avg_price_dollars")
        avg_cell = (f"{int(round(float(avg_p)*100))}¢"
                    if avg_p is not None else "—")
        cf = r.get("contracts_filled")
        cf_cell = (f"{int(round(float(cf)))}"
                    if cf is not None else "—")
        # Predicted winner = the player we bet on (with flag).
        side_player_html = _player_with_flag(
            r.get("side_player") or "", ioc_lookup)
        # My prob = model % for that player at order time.
        my_prob_cell = (f"{float(mp)*100:.0f}%"
                         if mp is not None else "—")
        # Winner = actual match winner. We won → side_player; we lost
        # → the OTHER side in the matchup string.
        matchup_str = r.get("matchup") or ""
        side_player = r.get("side_player") or ""
        won = bool(r.get("won"))
        winner_name = side_player
        if not won and matchup_str and side_player:
            parts = [p.strip() for p in matchup_str.split(" vs ")]
            if len(parts) == 2:
                winner_name = parts[0] if parts[1] == side_player else parts[1]
        winner_html = _player_with_flag(winner_name, ioc_lookup)
        tour_badge = _tour_badge(r.get("ticker") or "")
        out.append(
            "<tr>"
            f"<td>{tour_badge}</td>"
            f"<td class='small gray'>{html.escape(r.get('fill_time_label') or '—')}</td>"
            f"<td><div>{html.escape(r.get('series_label') or '')}</div>"
            f"<div class='small gray'>{html.escape(matchup_str)}</div></td>"
            f"<td>{side_player_html}</td>"
            f"<td>{winner_html}</td>"
            f"<td class='num'>{my_prob_cell}</td>"
            f"<td class='num'>{avg_cell}</td>"
            f"<td class='num'>{cf_cell}</td>"
            f"<td class='num'>{html.escape(r.get('final_position_label') or '')}</td>"
            f"<td class='num'>${cost:.2f}</td>"
            f"<td class='num {payout_cls}'>${payout:.2f}</td>"
            f"<td class='num {ret_cls}'>{ret_str}</td>"
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
    # New (2026-06-02): the multi-model trainer ships ensemble_weights
    # in metrics and ensemble_components in model_coefficients.json.
    # Render them when present; fall back silently when an older
    # bundle is loaded (the trainer-on-droplet might not have run yet).
    ensemble_weights = metrics.get("ensemble_weights") or {}
    components = coefficients.get("ensemble_components") or []
    per_model = metrics.get("per_model") or {}

    out: List[str] = []
    # The tennis renderer is reused by the table-tennis bot. Detect the
    # sport from the elo-only feature names in the coefficients file and
    # swap a few sport-specific phrases so the narrative reads correctly
    # regardless of which bot the page is being rendered for.
    elo_feats = ((coefficients.get("logistic") or {}).get("features") or [])
    is_table_tennis = "diff_style_elo_pre" in elo_feats
    elo_blurb = ("Elo (overall + style)" if is_table_tennis
                  else "Elo (overall + surface)")
    # Live-adjustment blurb. Tennis (2026-06-02): the in-match model is
    # paused while we rebuild the pre-match ensemble; trades fire on
    # the calibrated pre-match prob with no in-match nudge.
    if is_table_tennis:
        live_blurb = (
            "transparent rules layer (score-state momentum, point "
            "streaks, deuce / game-point / match-point volatility, "
            "closing-game flags)"
        )
    elif components:
        live_blurb = (
            "currently DISABLED — the bot trades on the calibrated "
            "pre-match probability only while we rebuild the in-match "
            "adjustment on top of the new mixed ensemble"
        )
    else:
        live_blurb = (
            "transparent rules layer (score-state, serve %, momentum, "
            "tiebreak / decider / medical flags)"
        )
    out.append(
        "<p class='small gray'>Pre-match probability blends a "
        f"logistic regression on {elo_blurb} with a calibrated "
        f"<b>mixed ensemble</b> (HGB + GBM + Random Forest + Extra "
        f"Trees + full-feature logistic, weights chosen by SLSQP on "
        f"validation log-loss). Live adjustment is {live_blurb}. "
        "Signals only fire when model and market disagree by more "
        "than the configured edge floor.</p>"
    )
    out.append("<h3 class='subhead'>Component breakdown</h3>")
    out.append("<table><thead><tr>"
               "<th>Component</th><th>Accuracy</th><th>Brier</th>"
               "<th>Log loss</th></tr></thead><tbody>")
    ensemble_label = "Mixed ensemble" if components else "GBT ensemble"
    for name, mm in [("Elo-only logistic", elo_only),
                      (ensemble_label, ens),
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

    # Per-base-model component table (only when the trainer emitted
    # ``ensemble_components`` — older bundles skip this block).
    if components:
        # Pretty names for the base estimators.
        pretty = {
            "hgb": "HistGradientBoosting",
            "gbm": "GradientBoosting (sklearn)",
            "rf": "Random Forest (300, depth 8)",
            "et": "Extra Trees (300, depth 8)",
            "lr_full": "Logistic (full features)",
            "xgb": "XGBoost",
        }
        out.append("<h3 class='subhead'>Mixed-ensemble base models</h3>")
        out.append(
            "<p class='small gray'>Each base classifier is calibrated "
            "by sigmoid on its own training-tail. Weights are chosen by "
            "SLSQP minimising log-loss on a held-out 20% validation "
            "slice — a weight of 0 means the optimiser found the model "
            "didn't add signal beyond the others.</p>"
        )
        out.append("<table><thead><tr>"
                   "<th>Base model</th><th>Weight</th>"
                   "<th>Standalone accuracy</th>"
                   "<th>Standalone Brier</th>"
                   "<th>Standalone log loss</th>"
                   "</tr></thead><tbody>")
        # Sort by weight desc so the most-relied-on model is on top.
        comps_sorted = sorted(
            components,
            key=lambda c: -float(c.get("weight") or 0.0),
        )
        for c in comps_sorted:
            nm = c.get("name", "")
            w = float(c.get("weight") or 0.0)
            mm = c.get("metrics") or per_model.get(nm) or {}
            out.append(
                f"<tr><td>{html.escape(pretty.get(nm, nm))}</td>"
                f"<td class='num'>{w*100:.1f}%</td>"
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
            elif n == "diff_style_elo_pre":
                interp = f"+1 style-Elo point on player_a {sign} P(A wins) marginally"
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
# Models tab                                                                  #
# --------------------------------------------------------------------------- #

def _tennis_confidence(rows_test: int) -> dict:
    """Tennis-equivalent of dashboard._holdout_confidence — the same
    sample-size tiering applied to the held-out match count the
    trainer reports in metrics.json. Tennis ships a much larger
    holdout than the macro bots (thousands of completed matches),
    so this almost always lands in the high-confidence tier; the
    card is here mainly to flag the rare case where a re-trained
    model has shrunk its test horizon.
    """
    n = int(rows_test or 0)
    if n == 0:
        return {"tier": "none", "color": "#8b949e",
                "label": "No held-out data",
                "reason": ("metrics.json is missing or didn't record "
                           "a test-set row count — confidence in the "
                           "metrics on this page can't be quantified."),
                "n": 0}
    if n < 100:
        return {"tier": "low", "color": "#f85149",
                "label": "Low confidence",
                "reason": (f"Only {n:,} held-out matches — the "
                           "headline accuracy / ROC AUC are noisy "
                           "and may swing several pts across "
                           "retrains."),
                "n": n}
    if n < 500:
        return {"tier": "moderate", "color": "#d29922",
                "label": "Moderate confidence",
                "reason": (f"{n:,} held-out matches — directionally "
                           "meaningful but per-decile calibration "
                           "bins still carry wide error bars."),
                "n": n}
    if n < 2000:
        return {"tier": "good", "color": "#3fb950",
                "label": "Good confidence",
                "reason": (f"{n:,} held-out matches — sample size "
                           "is large enough that the headline "
                           "metrics are stable to within ~1 pt."),
                "n": n}
    return {"tier": "high", "color": "#3fb950",
            "label": "High confidence",
            "reason": (f"{n:,} held-out matches — every calibration "
                       "decile carries enough data to read at face "
                       "value."),
            "n": n}


def _render_tennis_confidence_card(out: List[str], conf: dict) -> None:
    """Compact tennis-fallback confidence line — used when the trainer
    hasn't written a holdout_predictions.csv yet. The standard
    holdout-CSV path uses ``dashboard._render_confidence_card``;
    this is structurally identical but reads ``conf['n']`` from the
    tennis tier dict (which counts held-out matches via rows_test).
    """
    color = conf["color"]
    n = int(conf.get("n") or 0)
    label = html.escape(conf.get("label", ""))
    reason = html.escape(conf.get("reason", ""))
    if n <= 0:
        out.append(
            f"<p class='small gray' "
            f"style='margin:0 0 12px 0;' title='{reason}'>"
            f"Held-out test set: <span style='color:{color};"
            f"font-weight:600;'>{label}</span></p>"
        )
    else:
        out.append(
            f"<p class='small gray' "
            f"style='margin:0 0 12px 0;' title='{reason}'>"
            f"Held-out test set: <b style='color:#c9d1d9;'>"
            f"{n:,} matches</b> · "
            f"<span style='color:{color};font-weight:600;'>{label}</span>"
            f"</p>"
        )


def _render_tennis_models_page(metrics: dict, coefficients: dict,
                                sim_state: dict,
                                metrics_path: str | None = None) -> str:
    """Tennis Models tab. Mirrors the standard sim.db-bot model page
    section by section (confidence banner → top features → headline
    cards → model overview → ROC + confusion → historical calibration
    → live calibration), so a user navigating between bots sees the
    same structure regardless of bot type.
    """
    from pathlib import Path
    from .dashboard import (  # type: ignore
        _read_feature_importance, _read_holdout_predictions,
        _holdout_confidence, _render_confidence_card,
        _render_feature_source_table, _svg_roc_curve, _svg_confusion,
        _svg_calibration, roc_from_holdout, confusion_from_holdout,
        calibration_from_holdout,
    )
    out: List[str] = []

    # Holdout predictions drive the confidence tier (rendered next to
    # the ROC + Confusion grid below where the user looks for held-out
    # context) and the chart data. Tennis trainer dumps
    # holdout_predictions.csv into the same artifacts dir as
    # metrics.json — fall back to the metrics.json rows_test count for
    # older trainer outputs that don't write the CSV.
    artifacts_dir = (Path(metrics_path).parent if metrics_path
                     else None)
    holdout_pairs: List = []
    if artifacts_dir:
        holdout_path = artifacts_dir / "holdout_predictions.csv"
        if holdout_path.exists():
            holdout_pairs = _read_holdout_predictions(str(holdout_path))
    if holdout_pairs:
        conf = _holdout_confidence(holdout_pairs)
    else:
        rows_test = int((metrics or {}).get("rows_test") or 0)
        conf = _tennis_confidence(rows_test)

    # Feature artifacts are loaded once here so both the headline
    # metrics (above the fold) and the feature chart / table below
    # share the same ``feats`` list.
    feats: List[dict] = []
    if artifacts_dir:
        fi_path = artifacts_dir / "feature_importance.csv"
        if fi_path.exists():
            feats = _read_feature_importance(str(fi_path))

    # ── Headline metrics cards row — at the top, matching the Home /
    # Watchlist tabs. Surfaces the bottom-line numbers (accuracy / F1
    # / precision / recall / ROC AUC / feature count) up front before
    # the deep-dive chart and table.
    blended = (metrics or {}).get("blended") or {}
    if blended:
        out.append(
            "<div class='cards' "
            "style='display:grid;grid-template-columns:repeat(6, 1fr);"
            "gap:10px;width:100%;'>"
        )

        def _pct(v, decimals=0):
            try:
                return f"{float(v) * 100:.{decimals}f}%"
            except (TypeError, ValueError):
                return "—"

        n_features = len(feats) if feats else "—"
        cards = [
            ("Accuracy", _pct(blended.get("accuracy"), 1)),
            ("F1", _pct(blended.get("f1"), 0)),
            ("Precision", _pct(blended.get("precision"), 0)),
            ("Recall", _pct(blended.get("recall"), 0)),
            ("ROC AUC", _pct(blended.get("roc_auc"), 0)),
            ("Features", str(n_features)),
        ]
        for label, value in cards:
            out.append(
                f"<div class='card'><div class='label'>"
                f"{html.escape(label)}</div>"
                f"<div class='value'>{html.escape(str(value))}</div></div>"
            )
        out.append("</div>")

    # ── Top features — bars + readable table in one aligned panel ──
    if feats:
        out.append(_render_feature_source_table(feats))

    # ── Model overview (training-set provenance) ───────────────────
    last_retrain = "—"
    if artifacts_dir:
        bundle_path = artifacts_dir / "prematch_model.joblib"
        if bundle_path.exists():
            try:
                import datetime as _dt
                mt = _dt.datetime.fromtimestamp(
                    bundle_path.stat().st_mtime, tz=_dt.timezone.utc)
                last_retrain = mt.strftime("%Y-%m-%d %H:%M UTC")
            except (OSError, OverflowError):
                pass
    overview_items = [
        ("Last retrained", last_retrain),
        ("Training rows",
            f"{int((metrics or {}).get('rows_train') or 0):,}"),
        ("Held-out rows",
            f"{int((metrics or {}).get('rows_test') or 0):,}"),
        ("Train/test cutoff",
            (metrics or {}).get("cutoff_date") or "—"),
        ("Blend weights",
            "70% calibrated GBT + 30% logistic (ELO-only)"),
    ]
    out.append("<h3 class='subhead'>Model overview "
                "<span class='small gray'>(from training "
                "artifacts)</span></h3>")
    out.append(
        "<dl class='model-overview-dl' "
        "style='display:grid;grid-template-columns:auto 1fr;"
        "gap:6px 18px;margin:0 0 12px 0;font-size:13px;'>"
    )
    for label, value in overview_items:
        out.append(
            f"<dt class='gray' style='margin:0;'>"
            f"{html.escape(label)}</dt>"
            f"<dd style='margin:0;color:#c9d1d9;'>"
            f"{html.escape(str(value))}</dd>"
        )
    out.append("</dl>")

    # ── ROC curve + confusion matrix from the historical holdout ───
    if holdout_pairs:
        roc_points = roc_from_holdout(holdout_pairs)
        cm = confusion_from_holdout(holdout_pairs, threshold=0.5)
        n_pairs = len(holdout_pairs)
        auc_scalar = blended.get("roc_auc") if blended else None
        out.append(
            f"<p class='small gray' style='margin:0 0 6px 0;'>"
            f"Sourced from the trainer's held-out historical test "
            f"set ({n_pairs:,} match predictions vs ground-truth "
            f"outcomes). The model never saw this slice during "
            f"training.</p>"
        )
        # Compact held-out row count + confidence tier, surfaced next
        # to the plots it grades (moved out of the top-of-page banner).
        _render_confidence_card(out, conf)
    elif conf:
        # No held-out CSV but the tennis fallback tier is still
        # meaningful; show the confidence card alone. (The previous
        # version of this branch tried to also render an ROC curve
        # + confusion matrix, but those references — roc_points,
        # cm, n_pairs — only get defined in the ``if holdout_pairs:``
        # branch above, so it 500'd the moment a sport bot landed
        # here. NBA was the first bot in production to do so since
        # it has a model accuracy metric but no held-out CSV.)
        _render_tennis_confidence_card(out, conf)

        out.append("<h3 class='subhead'>Calibration "
                    "<span class='small gray'>(historical held-out "
                    "test set, predicted prob vs observed positive "
                    "rate)</span></h3>")
        cal_bins = calibration_from_holdout(holdout_pairs, n_bins=10)
        out.append(_svg_calibration(cal_bins))

    # ── Held-out metrics by component (the existing tennis-only
    # comparison row, kept underneath the standard sections so the
    # user can still see what each blend component contributes). ──
    components = [
        ("elo_only", "ELO baseline"),
        ("ensemble", "Gradient-boost ensemble"),
        ("blended", "Blended (final)"),
    ]
    rows = []
    for key, label in components:
        c = (metrics or {}).get(key) or {}
        if not c:
            continue
        rows.append((label, c))
    if rows:
        out.append("<h3 class='subhead'>Held-out metrics by component"
                    " <span class='small gray'>(test set, "
                    f"{int((metrics or {}).get('rows_test') or 0):,} rows)"
                    "</span></h3>")
        out.append("<table><thead><tr><th>Model</th>"
                    "<th class='num'>Accuracy</th>"
                    "<th class='num'>F1</th>"
                    "<th class='num'>Precision</th>"
                    "<th class='num'>Recall</th>"
                    "<th class='num'>ROC AUC</th>"
                    "<th class='num'>Brier</th>"
                    "<th class='num'>Log loss</th>"
                    "</tr></thead><tbody>")
        for label, c in rows:
            out.append(
                f"<tr><td>{html.escape(label)}</td>"
                f"<td class='num'>{_fmt_pct(c.get('accuracy'), 1)}</td>"
                f"<td class='num'>{_fmt_pct(c.get('f1'), 1)}</td>"
                f"<td class='num'>{_fmt_pct(c.get('precision'), 1)}</td>"
                f"<td class='num'>{_fmt_pct(c.get('recall'), 1)}</td>"
                f"<td class='num'>{_fmt_pct(c.get('roc_auc'), 1)}</td>"
                f"<td class='num'>{(c.get('brier') or 0):.4f}</td>"
                f"<td class='num'>{(c.get('log_loss') or 0):.4f}</td>"
                "</tr>"
            )
        out.append("</tbody></table>")

    # Full feature list — every coefficient the bot uses to score a
    # match (the user explicitly asked for "all features being used to
    # make decisions"). Sorted by absolute coefficient size so the
    # most-influential ones float to the top.
    coeffs = (coefficients or {}).get("coefficients") or {}
    if isinstance(coeffs, dict) and coeffs:
        items = sorted(coeffs.items(),
                        key=lambda kv: abs(float(kv[1] or 0)),
                        reverse=True)
        out.append(
            f"<h3 class='subhead'>Features the model uses to make decisions"
            f" <span class='small gray'>({len(items)} total)</span></h3>"
        )
        out.append("<p class='small gray' style='margin:0 0 8px 0;'>"
                    "Per-feature coefficient on the live blended logistic. "
                    "<span style='color:#3fb950;'>Green</span> bars push "
                    "toward player A winning; "
                    "<span style='color:#f85149;'>red</span> push toward "
                    "player B.</p>")
        max_abs = max((abs(float(v or 0)) for _, v in items), default=1.0) or 1.0
        out.append("<svg viewBox='0 0 760 "
                    f"{30 + len(items) * 18}' "
                    "style='width:100%;height:auto;display:block;"
                    "background:#0d1117;border:1px solid #21262d;"
                    "border-radius:6px;'>")
        for i, (name, val) in enumerate(items):
            try:
                v = float(val or 0)
            except (TypeError, ValueError):
                v = 0.0
            y = 16 + i * 18
            mid = 280
            bar_w = abs(v) / max_abs * 380
            color = "#3fb950" if v >= 0 else "#f85149"
            x = mid if v >= 0 else mid - bar_w
            display_name = name if len(name) <= 32 else name[:29] + "…"
            out.append(
                f"<g><title>{html.escape(name)} · coef {v:+.4f}</title>"
                f"<text x='270' y='{y + 4}' fill='#c9d1d9' font-size='11' "
                f"text-anchor='end' "
                f"font-family='ui-monospace,SFMono-Regular,monospace'>"
                f"{html.escape(display_name)}</text>"
                f"<line x1='{mid}' y1='{y - 6}' x2='{mid}' y2='{y + 6}' "
                f"stroke='#484f58'/>"
                f"<rect x='{x:.1f}' y='{y - 4}' width='{bar_w:.1f}' "
                f"height='8' fill='{color}' rx='1'/>"
                f"<text x='{(x + bar_w + 6) if v >= 0 else (x - 6):.1f}' "
                f"y='{y + 4}' fill='#8b949e' font-size='10' "
                f"text-anchor='{'start' if v >= 0 else 'end'}'>"
                f"{v:+.3f}</text></g>"
            )
        out.append("</svg>")
    else:
        out.append("<div class='empty'>Coefficients file not "
                    "available — feature list will populate after the "
                    "next retrain.</div>")

    # Live calibration on closed paper bets — same shape as the sim.db
    # bots' calibration plot, just sourced from sim_state instead of a
    # SQL table. Kept simple: bin by predicted side-prob at entry,
    # measure realized win rate.
    closed = list((sim_state or {}).get("closed_positions") or [])
    if closed:
        n_bins = 10
        bins = [{"lo": i / n_bins, "hi": (i + 1) / n_bins,
                 "n": 0, "wins": 0} for i in range(n_bins)]
        for c in closed:
            entry_p = c.get("entry_model_prob")
            if entry_p is None:
                continue
            try:
                p = float(entry_p)
            except (TypeError, ValueError):
                continue
            idx = min(n_bins - 1, max(0, int(p * n_bins)))
            bins[idx]["n"] += 1
            if (c.get("result") or "").upper() == "WIN":
                bins[idx]["wins"] += 1
        populated = [b for b in bins if b["n"] > 0]
        if populated:
            out.append("<h3 class='subhead'>Live calibration "
                        "<span class='small gray'>(closed paper bets, "
                        f"{len(closed)} total)</span></h3>")
            from .dashboard import _svg_calibration  # type: ignore
            out.append(_svg_calibration(bins))

    # Kalshi-bet calibration — the honest "how is the model doing on
    # real money?" panel. Sourced from
    # ``data/processed/artifacts/kalshi_calibration.json`` which the
    # daily retrain timer writes after refitting the model. Reads
    # the per-bet rows and presents the same metric trio as the
    # held-out card (Brier / accuracy / win rate) plus the
    # calibration gap (mean predicted prob − actual win rate).
    if artifacts_dir:
        kalshi_path = artifacts_dir / "kalshi_calibration.json"
        if kalshi_path.exists():
            try:
                with kalshi_path.open("r", encoding="utf-8") as f:
                    kcal = json.load(f)
            except (OSError, ValueError):
                kcal = None
            if kcal and (kcal.get("n") or 0) > 0:
                out.append(_render_kalshi_calibration_card(kcal))

    return "".join(out)


def _render_kalshi_calibration_card(kcal: dict) -> str:
    """Render the Kalshi-bet calibration section on the Models tab.
    Mirrors the held-out metrics card style so the user can eyeball
    Sackmann-eval vs real-money-eval side by side."""
    n = int(kcal.get("n") or 0)
    brier = kcal.get("brier")
    acc = kcal.get("accuracy")
    win_rate = kcal.get("win_rate")
    mean_pred = kcal.get("mean_predicted_prob")
    gap = kcal.get("calibration_gap")
    log_loss = kcal.get("log_loss")
    generated = (kcal.get("generated_at") or "")[:19].replace("T", " ")
    out: List[str] = []
    out.append(
        "<h3 class='subhead' style='margin-top:24px;'>"
        "Calibration on real Kalshi bets "
        f"<span class='small gray'>({n} settled bet"
        f"{'s' if n != 1 else ''}"
        f"{' · refreshed ' + html.escape(generated) if generated else ''}"
        f")</span></h3>"
    )
    out.append(
        "<p class='small gray' style='margin:0 0 10px 0;'>"
        "How the live model has performed on every Kalshi bet that has "
        "settled. Pulled from <code>/portfolio/settlements</code> + "
        "<code>/portfolio/fills</code>, joined with sim_state for the "
        "model's pre-bet probability. Refreshed by the daily retrain "
        "timer (05:00 UTC) and the standalone "
        "<code>scripts/sync_from_kalshi.py</code>."
        "</p>"
    )

    def _card(label: str, value: str, hint: str = "",
               cls: str = "") -> str:
        title_attr = (f" title='{html.escape(hint)}'" if hint else "")
        return (f"<div class='card'{title_attr}>"
                f"<div class='label'>{html.escape(label)}</div>"
                f"<div class='value {cls}'>{value}</div></div>")

    cards: List[str] = []
    cards.append(_card(
        "Brier",
        f"{brier:.4f}" if brier is not None else "—",
        "Mean squared error of probability predictions. Lower is "
        "better. 0.25 is the no-signal baseline; <0.20 is generally "
        "good for binary-outcome markets."
    ))
    cards.append(_card(
        "Accuracy",
        f"{acc*100:.1f}%" if acc is not None else "—",
        "Share of bets where the model's >50% prediction matched the "
        "actual outcome.",
        cls=("green" if acc is not None and acc >= 0.6 else
              "red" if acc is not None and acc < 0.5 else "")
    ))
    cards.append(_card(
        "Actual win rate",
        f"{win_rate*100:.1f}%" if win_rate is not None else "—",
        "Share of bets we won — what actually happened on real money.",
        cls=("green" if win_rate is not None and win_rate >= 0.5 else
              "red" if win_rate is not None and win_rate < 0.5 else "")
    ))
    cards.append(_card(
        "Mean predicted prob",
        f"{mean_pred*100:.1f}%" if mean_pred is not None else "—",
        "Average model probability on the side we bet. A well-"
        "calibrated bot's mean predicted prob ≈ actual win rate."
    ))
    cards.append(_card(
        "Calibration gap",
        (f"{gap*100:+.1f}pp" if gap is not None else "—"),
        "Mean predicted prob − actual win rate. Positive = model "
        "over-confident (says 65% but wins 55%); negative = under-"
        "confident. ±5pp is healthy noise on small samples.",
        cls=("red" if gap is not None and abs(gap) > 0.10 else
              "green" if gap is not None and abs(gap) <= 0.05 else "")
    ))
    cards.append(_card(
        "Log loss",
        f"{log_loss:.4f}" if log_loss is not None else "—",
        "Cross-entropy of the predictions. Penalizes confident wrong "
        "answers harder than Brier."
    ))
    out.append("<div class='row'>" + "".join(cards) + "</div>")

    # Reliability diagram — bucket-level mean predicted prob vs
    # actual win rate. Reuses the dashboard's _svg_calibration via the
    # same {lo, hi, n, wins} schema it accepts (we map our buckets
    # to the wins=actual*n form).
    buckets = kcal.get("buckets") or []
    if buckets:
        adapted = []
        for b in buckets:
            actual = float(b.get("actual_win_rate") or 0)
            n_b = int(b.get("n") or 0)
            adapted.append({
                "lo": float(b.get("bin_lo") or 0),
                "hi": float(b.get("bin_hi") or 1),
                "n": n_b,
                "wins": int(round(actual * n_b)),
            })
        if adapted:
            out.append(
                "<h4 class='subhead' style='margin-top:14px;'>Reliability "
                "<span class='small gray'>(predicted prob vs actual win "
                "rate)</span></h4>"
            )
            from .dashboard import _svg_calibration  # type: ignore
            out.append(_svg_calibration(adapted))

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
    active_tab = tab_key if tab_key in ("watchlist", "models") else "watchlist"
    # Bot dropdown sits above the tab bar so it applies uniformly
    # across tabs (matches the standard renderer's layout).
    out.append(_render_bot_dropdown(available_bots, current_bot_key))
    out.append(_render_tab_bar(active=active_tab))

    if active_tab == "models":
        # ── Models section ───────────────────────────────────────────
        out.append("<div class='section'><h2>Model</h2><div class='body'>")
        out.append(_render_tennis_models_page(metrics, coefficients,
                                                sim_state,
                                                metrics_path=metrics_path))
        out.append("</div></div>")
    else:
        # ── Watchlist section ────────────────────────────────────────
        # Order per user spec: Active paper bets at top, forecast graph
        # in the middle (interactive — click a ticker row to plot it),
        # ticker table at the bottom.
        out.append("<div class='section'><h2>Watchlist — model vs market</h2>"
                    "<div class='body'>")
        out.append(_render_current_prediction(metrics, sim_state,
                                                  sim_state_path=sim_state_path))

        # Active paper bets — render via the standard shared renderer
        # so the column shape (Opened | Ticker | Title | Contracts |
        # Side | Entry prob | Current prob | Entry cost | Potential
        # gain | Closes in) and styling match NBA / gas-prices /
        # cpi / claims exactly. The tennis adapter
        # ``active_bets_for_rollup`` already reshapes sim_state into
        # the standard row schema.
        out.append("<h3 class='subhead'>Active paper bets</h3>")
        from .dashboard import _render_active_bets_table  # type: ignore
        bets = active_bets_for_rollup(sim_state_path, watchlist_path)
        _active_buf: List[str] = []
        _render_active_bets_table(
            _active_buf, bets,
            empty_msg="No active paper bets right now.",
            show_bot=False,
        )
        out.append("".join(_active_buf))

        # Forecast graph + full watchlist table show ALL matches in the
        # configured Kalshi series. Rows are sorted with BUY-eligible
        # (top 10 by edge × EV — these are the rows the simulator
        # actually opens on) at the top of the table; the dedicated
        # "Top 10" section was dropped because the green-tinted rows in
        # the main table already surface them.
        out.append(_render_forecast_graph(rows))

        age = _last_updated_age(payload.get("generated_at"))
        with_contracts = sum(
            1 for r in rows
            if r.get("open_interest") is not None
            and float(r.get("open_interest") or 0) > 0
        )
        out.append(
            f"<h3 class='subhead'>Tradeable matches · {with_contracts} "
            f"<span class='small gray'>of {len(rows)} pulled from Kalshi · "
            f"generated {html.escape(age)}</span></h3>"
        )
        out.append(_render_watchlist_table(payload, sim_state=sim_state))

        out.append("</div></div>")  # /body /section

    out.append("</body></html>")
    return "".join(out)


# --------------------------------------------------------------------------- #
# Training Data tab                                                           #
# --------------------------------------------------------------------------- #

# Default location of the tennis training DB on the droplet — matches the
# trainer's ``upsert_training_panel`` path. ``Path.exists`` decides whether
# the dashboard shows a populated table or a "not initialised yet" stub.
_TRAINING_DB_PATH = Path("/root/tennis-forecast/data/training_history.db")

# All training-data table columns and their definitions. Each entry is
# (sql-column, short-label, full-definition). The short label goes in
# the table header; clicking it pops up a definition modal. ``None``
# label hides the column from rendering (e.g. internal IDs).
#
# The order here is the rendered column order.
_TRAINING_COLUMNS: list[tuple[str, str, str]] = [
    # ── Match identity ───────────────────────────────────────────────
    ("tourney_date", "Date",
     "Date of the match's tournament round, in YYYY-MM-DD."),
    ("tourney_name", "Tournament",
     "Tournament name from the official ATP/WTA tour calendar."),
    ("tour", "Tour",
     "ATP (men's) or WTA (women's). Determined by which Sackmann "
     "match file the row came from."),
    ("surface", "Surface",
     "Hard / Clay / Grass / Carpet — the playing surface."),
    ("level", "Level",
     "Tournament tier raw code: G = Grand Slam, M = Masters 1000, "
     "A = ATP 500/250 or WTA tier-equivalent, F = Tour Finals, "
     "D = Davis Cup, C = Challenger, S = ITF Futures."),
    ("round", "Round",
     "Round of the match: R128, R64, R32, R16, QF, SF, F (final), "
     "RR = round-robin, BR = bronze."),
    ("draw_size", "Draw",
     "Total number of players in the main draw. 128 for a Slam, 64 "
     "for most Masters, 32 for most 250s."),
    ("best_of", "BO",
     "Best-of-3 or best-of-5 sets. Slams + Davis Cup men's are "
     "best-of-5; everything else is best-of-3."),
    # ── Outcome ─────────────────────────────────────────────────────
    ("player_a", "Player A",
     "First player in the matchup. Each historical match is stored "
     "twice in the training panel — once with Player A = the actual "
     "winner and once with Player A = the actual loser — so the "
     "feature differences cancel and the trained model isn't biased "
     "toward putting the winner on either side."),
    ("player_b", "Player B",
     "Second player in the matchup. See Player A for the orientation "
     "note."),
    ("winner", "Winner",
     "Name of the player who actually won this match."),
    # ── Player A raw attributes ─────────────────────────────────────
    ("a_age", "A age",
     "Player A's age in years at match start."),
    ("a_height_cm", "A height",
     "Player A's listed height in centimetres."),
    ("a_hand", "A hand",
     "Player A's playing hand: R = right, L = left, U = unknown / "
     "ambidextrous."),
    ("a_country", "A country",
     "Player A's nationality (IOC 3-letter code)."),
    ("a_rank", "A rank",
     "Player A's ATP/WTA singles ranking at the time of the match. "
     "Lower number = better."),
    ("a_rank_points", "A pts",
     "Player A's ranking points entering the match."),
    ("a_seed", "A seed",
     "Player A's seeding in this draw, if seeded."),
    ("a_entry", "A entry",
     "How Player A entered the draw: Q = qualifier, WC = wild card, "
     "LL = lucky loser, SE = special exempt, ALT = alternate, "
     "PR = protected ranking."),
    # ── Player B raw attributes ─────────────────────────────────────
    ("b_age", "B age", "Player B's age in years at match start."),
    ("b_height_cm", "B height", "Player B's listed height in cm."),
    ("b_hand", "B hand",
     "Player B's playing hand: R = right, L = left, U = unknown."),
    ("b_country", "B country",
     "Player B's nationality (IOC 3-letter code)."),
    ("b_rank", "B rank",
     "Player B's ATP/WTA singles ranking at match time."),
    ("b_rank_points", "B pts",
     "Player B's ranking points entering the match."),
    ("b_seed", "B seed",
     "Player B's seeding in this draw, if seeded."),
    ("b_entry", "B entry",
     "Player B's entry route into the draw (Q, WC, LL, SE, ALT, PR)."),
    # ── Engineered features the model actually trains on ─────────────
    ("diff_elo_pre", "Elo Δ",
     "Player A's pre-match overall Elo minus Player B's. Computed "
     "rolling through the historical match panel. Larger positive = "
     "Player A is the all-surface favourite. ⚙ MODEL FEATURE."),
    ("diff_surface_elo_pre", "Surface Elo Δ",
     "Surface-specific Elo difference (A − B) for this match's "
     "surface. Captures the fact that some players are clay "
     "specialists, others grass specialists, etc. ⚙ MODEL FEATURE."),
    ("diff_form_last5", "Form 5 Δ",
     "Win-rate over the last 5 matches (A − B). Captures short-term "
     "momentum / cold streaks. ⚙ MODEL FEATURE."),
    ("diff_form_last10", "Form 10 Δ",
     "Win-rate over the last 10 matches (A − B). Smoother form "
     "signal that's less reactive to a single bad day. ⚙ MODEL "
     "FEATURE."),
    ("diff_avg_serve_pts_won_10", "Serve % Δ",
     "Average serve points won % over the last 10 matches (A − B). "
     "A direct measure of who's holding serve better recently. "
     "⚙ MODEL FEATURE."),
    ("diff_avg_return_pts_won_10", "Return % Δ",
     "Average return points won % over the last 10 matches (A − B). "
     "Captures who's been breaking serve / pressuring the opponent's "
     "delivery. ⚙ MODEL FEATURE."),
    ("diff_avg_bp_saved_10", "BP saved Δ",
     "Average break-points saved % over the last 10 matches (A − B). "
     "Clutch-on-serve indicator. ⚙ MODEL FEATURE."),
    ("diff_days_rest", "Days rest Δ",
     "Days since each player's last match (A − B). Positive = "
     "Player A had more rest. Top permutation-importance feature in "
     "the current model. ⚙ MODEL FEATURE."),
    ("h2h_a_wins_minus_b_wins", "H2H Δ",
     "Career head-to-head record up to (not including) this match: "
     "A's wins minus B's wins. ⚙ MODEL FEATURE."),
    ("rank_diff", "Rank Δ",
     "B's ranking minus A's ranking (so positive = A is higher-"
     "ranked / better). ⚙ MODEL FEATURE."),
    ("level_rank", "Level rank",
     "Tournament tier as a numeric code: Grand Slam = 4, Masters / "
     "WTA 1000 = 3, ATP 500 / WTA 500 / ATP 250 / WTA 250 = 2, "
     "Davis Cup / Challenger / other = 1. ⚙ MODEL FEATURE."),
    ("round_rank", "Round rank",
     "Round depth as a numeric code: R128 = 1, R64 = 2, R32 = 3, "
     "R16 = 4, QF = 5, SF = 6, F = 8. ⚙ MODEL FEATURE."),
    # ── Derived / candidate features (NOT currently selected) ───────
    ("age_diff", "Age Δ",
     "Player A's age minus B's. Negative = Player A is younger. "
     "Computed but not currently in the selected feature list — "
     "tracked here in case it surfaces signal in a future search."),
    ("height_diff_cm", "Height Δ",
     "Player A's height minus B's, in cm. Taller players historically "
     "have an edge on fast surfaces. Candidate feature, not yet "
     "selected."),
    ("rank_points_diff", "Rank pts Δ",
     "Player A's ranking points minus B's. A continuous version of "
     "the rank diff that's more sensitive to the gap between top-5 "
     "and top-20. Candidate feature."),
    ("seed_diff", "Seed Δ",
     "Player B's seed minus A's. Positive = A is higher-seeded. "
     "Candidate feature."),
    ("hand_match", "Same hand?",
     "1 if both players are right-handed or both left-handed, "
     "0 if one is left and one is right (the lefty advantage case), "
     "blank when at least one is unknown. Candidate feature."),
    ("same_country", "Same flag?",
     "1 if both players share the same IOC country code, else 0. "
     "Captures the rare same-country matchup. Candidate feature."),
]

# Backwards-compat alias used in older render code paths. Maps the
# columns the rendered table previously knew about to the new
# (label, definition) pair the modal reads.
_FEATURE_LABELS: Dict[str, Tuple[str, str]] = {
    sql: (label, definition)
    for sql, label, definition in _TRAINING_COLUMNS
}


def _open_training_db():
    """Try to import the trainer-side training_db module to reuse its
    pagination helpers. Returns ``None`` if tennis-forecast's ``src/``
    isn't on sys.path (which happens at dashboard startup before the
    bots' upstream packages are loaded — the alias registered by
    bots/tennis.py adds ``src.data`` to sys.modules)."""
    try:
        # The tennis bot loads its upstream package under the alias
        # ``tennis_src`` (see bots/_base.load_upstream_as_alias) and
        # registers ``src`` -> ``tennis_src`` aliases for joblib unpickle.
        # That alias chain also makes ``src.data.training_db`` reachable.
        import importlib
        return importlib.import_module("src.data.training_db")
    except ImportError:
        try:
            return importlib.import_module("tennis_src.data.training_db")
        except Exception:  # noqa: BLE001
            return None


def render_training_data_panel(*, current_bot: str | None,
                                  page: int = 1, page_size: int = 50,
                                  tour_filter: str | None = None,
                                  split_filter: str | None = None,
                                  current_tab: str = "training",
                                  period_key: str = "all") -> str:
    """Render the Training Data tab. Tennis-only — the panel reads the
    rows the trainer wrote to ``training_history.db`` and paginates
    over them. Shows the 12 engineered features the model trains on,
    the binary winner label, and which train/val/test split slice the
    row belongs to.

    Other bots see a brief explanation: this DB is tennis-specific
    until their trainers also adopt the pattern.
    """
    if current_bot and current_bot != "tennis":
        return (
            "<section class='card'><div class='body'>"
            "<h2>Training Data</h2>"
            "<p class='small gray'>The Training Data tab is currently "
            "tennis-only. Each row shown reflects what the model trained "
            "on — engineered features and the binary winner label. "
            "Other bots haven't been wired into the training database "
            "yet.</p></div></section>"
        )
    db_mod = _open_training_db()
    if db_mod is None or not _TRAINING_DB_PATH.exists():
        return (
            "<section class='card'><div class='body'>"
            "<h2>Training Data — tennis</h2>"
            "<p class='small gray'>The training database hasn't been "
            "populated yet. It's written by the daily tennis-forecast "
            "retrain (see <code>src/models/train_prematch_model.py</code>'s "
            "<code>upsert_training_panel</code> call). Run the trainer "
            "once and the rows appear here.</p></div></section>"
        )

    # Whitelist filter values so a malformed query string can't be
    # forwarded into the SQL.
    tour_filter = tour_filter if tour_filter in ("ATP", "WTA") else None
    split_filter = split_filter if split_filter in ("train", "val",
                                                       "test") else None

    # Kalshi-only rows are matches the bot recorded on Kalshi that
    # the Sackmann panel doesn't have yet (the source updates on a
    # lag). They're date-newer than every historical training row, so
    # they sit at the top of the sort.
    kalshi_only_all = _build_kalshi_only_rows(tour_filter)
    n_kalshi = len(kalshi_only_all)
    n_historical = db_mod.count_training_matches(
        _TRAINING_DB_PATH, tour=tour_filter, split=split_filter,
    )
    total = n_kalshi + n_historical
    total_pages = max(1, (total + page_size - 1) // page_size)
    page = min(max(1, page), total_pages)

    # Slice the combined window. The combined sort order is:
    #   indices [0,           n_kalshi)               -> Kalshi-only rows
    #   indices [n_kalshi,    n_kalshi + n_historical) -> historical
    start = (page - 1) * page_size
    end = start + page_size

    kalshi_only_rows: List[Dict[str, Any]] = []
    if start < n_kalshi:
        kalshi_only_rows = kalshi_only_all[start:min(end, n_kalshi)]

    rows: List[Dict[str, Any]] = []
    if end > n_kalshi and n_historical > 0:
        hist_offset = max(0, start - n_kalshi)
        hist_limit = end - max(start, n_kalshi)
        if hist_limit > 0:
            rows = db_mod.fetch_training_matches(
                _TRAINING_DB_PATH,
                offset=hist_offset, limit=hist_limit,
                tour=tour_filter, split=split_filter,
            )

    # Derive Winner from label for every training row (label=1 means
    # player_a won; label=0 means player_b won). Done here rather
    # than in SQL because the column is computed only at render time.
    for r in rows:
        label_v = r.get("label")
        if label_v is None:
            r["winner"] = None
        else:
            r["winner"] = (r.get("player_a") if int(label_v) == 1
                            else r.get("player_b"))


    out: List[str] = []
    out.append("<section class='card'><div class='body'>")
    out.append("<h2>Training Data — tennis</h2>")
    out.append(
        f"<p class='small gray'>All matches across the historical "
        f"training panel and the live Kalshi calendar — "
        f"<b>{total:,}</b> from the trainer's most recent fit, plus "
        f"<b>{len(kalshi_only_rows):,}</b> recorded on Kalshi but not "
        f"yet in the Sackmann panel (it updates on a lag). Each row "
        f"is a single match with the winner and every candidate "
        f"feature; rows duplicate when the same match exists in both "
        f"sources. Sorted newest first; columns flagged "
        f"⚙ MODEL FEATURE in their definition are the ones the model "
        f"actually trains on.</p>"
    )

    # Tour / split filter pills. Hand-rolled query-string preservation
    # so the pagination links below also keep the active filter.
    def _filter_link(key: str, value: str | None, label: str,
                      active: bool) -> str:
        params = [("tab", current_tab)]
        if current_bot:
            params.append(("bot", current_bot))
        if period_key and period_key != "all":
            params.append(("period", period_key))
        # Preserve the OTHER filter dim
        if key != "tour" and tour_filter:
            params.append(("tour", tour_filter))
        if key != "split" and split_filter:
            params.append(("split", split_filter))
        if value is not None:
            params.append((key, value))
        qs = "&".join(f"{k}={v}" for k, v in params)
        cls = "tab-pill" + (" tab-pill-active" if active else "")
        return f"<a class='{cls}' href='?{qs}'>{html.escape(label)}</a>"

    out.append("<div class='tab-bar' style='margin-top:8px;'>")
    out.append("<span class='small gray' style='margin-right:8px;'>Tour:</span>")
    out.append(_filter_link("tour", None, "All", tour_filter is None))
    out.append(_filter_link("tour", "ATP", "ATP", tour_filter == "ATP"))
    out.append(_filter_link("tour", "WTA", "WTA", tour_filter == "WTA"))
    out.append("</div>")
    out.append(
        "<p class='small gray' style='margin-top:8px;'>"
        "Click any column header for its definition. Columns marked "
        "<b>⚙ MODEL FEATURE</b> in the definition are the ones the "
        "current model actually trains on; everything else is a "
        "candidate feature carried for review.</p>"
    )

    # Table — every column is a clickable header that opens a modal
    # with the column's definition (see the inline script below).
    out.append("<div style='overflow-x:auto;margin-top:12px;'>")
    out.append("<table class='training-data-table'><thead><tr>")
    for sql, label, _ in _TRAINING_COLUMNS:
        # Numeric columns get .num for right-alignment; player names /
        # categorical attrs stay left-aligned for readability.
        is_num = sql not in {
            "tourney_date", "tourney_name", "tour", "surface", "level",
            "round", "player_a", "player_b", "a_hand", "a_country",
            "a_entry", "b_hand", "b_country", "b_entry",
        }
        cls = " class='num'" if is_num else ""
        out.append(
            f"<th{cls}><button type='button' class='col-def-btn' "
            f"data-col='{html.escape(sql)}'>"
            f"{html.escape(label)}</button></th>"
        )
    out.append("</tr></thead><tbody>")
    if not rows and not kalshi_only_rows:
        out.append(
            f"<tr><td colspan='{len(_TRAINING_COLUMNS)}' "
            f"class='empty'>No rows for the selected filter.</td></tr>"
        )

    def _fmt_cell(sql: str, v: Any) -> str:
        if v is None or v == "":
            return "—"
        if sql in {"a_hand", "b_hand"}:
            return html.escape(str(v))
        if sql in {"hand_match", "same_country"}:
            return "Yes" if int(v) == 1 else "No"
        if isinstance(v, float):
            # Diffs render with a sign; raw stats render plain.
            if sql.endswith("_diff") or sql.startswith("diff_") or \
                    sql == "h2h_a_wins_minus_b_wins":
                return f"{v:+.3f}"
            return f"{v:.3f}" if abs(v) < 1000 else f"{int(v):,}"
        if isinstance(v, int):
            return f"{v:,}" if abs(v) >= 1000 else str(v)
        return html.escape(str(v))

    # Render Kalshi-only rows first (most recent, page 1 only), then
    # the paginated historical rows below.
    for r in kalshi_only_rows + rows:
        out.append("<tr>")
        for sql, _, _ in _TRAINING_COLUMNS:
            v = r.get(sql)
            is_num = sql not in {
                "tourney_date", "tourney_name", "tour", "surface", "level",
                "round", "player_a", "player_b", "a_hand", "a_country",
                "a_entry", "b_hand", "b_country", "b_entry",
            }
            cls = " class='num'" if is_num else ""
            out.append(f"<td{cls}>{_fmt_cell(sql, v)}</td>")
        out.append("</tr>")
    out.append("</tbody></table></div>")

    # Pagination — Prev | page N of M | Next + jump-to dropdown
    def _page_link(p: int) -> str:
        params = [("tab", current_tab)]
        if current_bot:
            params.append(("bot", current_bot))
        if period_key and period_key != "all":
            params.append(("period", period_key))
        if tour_filter:
            params.append(("tour", tour_filter))
        if split_filter:
            params.append(("split", split_filter))
        params.append(("page", str(p)))
        return "?" + "&".join(f"{k}={v}" for k, v in params)

    out.append("<div class='small' style='margin-top:14px;display:flex;"
                "align-items:center;gap:12px;'>")
    if page > 1:
        out.append(f"<a class='tab-pill' href='{_page_link(page - 1)}'>← Prev</a>")
    else:
        out.append("<span class='tab-pill tab-pill-disabled'>← Prev</span>")
    out.append(
        f"<span>Page <b>{page:,}</b> of <b>{total_pages:,}</b> "
        f"<span class='gray'>({total:,} rows)</span></span>"
    )
    # Jump-to dropdown — covers up to 1000 pages cheaply; beyond that
    # rendering options gets heavy, so fall back to a small text input.
    if total_pages <= 1000:
        out.append("<form method='get' style='display:inline;'>")
        # Preserve filters as hidden fields so the dropdown's submit
        # navigates to the right URL.
        out.append(f"<input type='hidden' name='tab' value='{html.escape(current_tab)}'>")
        if current_bot:
            out.append(f"<input type='hidden' name='bot' value='{html.escape(current_bot)}'>")
        if period_key and period_key != "all":
            out.append(f"<input type='hidden' name='period' value='{html.escape(period_key)}'>")
        if tour_filter:
            out.append(f"<input type='hidden' name='tour' value='{html.escape(tour_filter)}'>")
        if split_filter:
            out.append(f"<input type='hidden' name='split' value='{html.escape(split_filter)}'>")
        out.append("<label class='gray' style='margin-right:6px;'>Jump:</label>")
        out.append("<select name='page' onchange='this.form.submit()'>")
        for p in range(1, total_pages + 1):
            sel = " selected" if p == page else ""
            out.append(f"<option value='{p}'{sel}>{p}</option>")
        out.append("</select></form>")
    else:
        out.append(
            "<form method='get' style='display:inline;'>"
            f"<input type='hidden' name='tab' value='{html.escape(current_tab)}'>"
            f"<input type='hidden' name='bot' value='{html.escape(current_bot or '')}'>"
            "<label class='gray' style='margin-right:6px;'>Jump to:</label>"
            f"<input type='number' name='page' min='1' max='{total_pages}' "
            f"value='{page}' style='width:80px;'><button type='submit'>Go</button>"
            "</form>"
        )
    if page < total_pages:
        out.append(f"<a class='tab-pill' href='{_page_link(page + 1)}'>Next →</a>")
    else:
        out.append("<span class='tab-pill tab-pill-disabled'>Next →</span>")
    out.append("</div>")

    # ── Column-definition modal + inline JS ──────────────────────────
    # Builds a small JS map of every column's full definition and pops
    # up a centred panel when any column header is clicked. Self-
    # contained so the existing dashboard CSS / JS doesn't need to know
    # this panel exists.
    import json as _json
    defs = {sql: {"label": label, "def": definition}
             for sql, label, definition in _TRAINING_COLUMNS}
    out.append("<dialog id='col-def-modal' class='col-def-modal'>"
                "<form method='dialog'>"
                "<h3 id='col-def-title'></h3>"
                "<p id='col-def-body'></p>"
                "<button type='submit'>Close</button>"
                "</form></dialog>")
    out.append(
        "<style>"
        ".col-def-btn { background:none; border:0; color:inherit; "
        "font:inherit; cursor:pointer; padding:0; "
        "text-decoration:underline dotted; }"
        ".col-def-btn:hover { color:#79c0ff; }"
        ".col-def-modal { max-width:520px; padding:16px 20px; "
        "border:1px solid #30363d; background:#0d1117; color:#c9d1d9; }"
        ".col-def-modal::backdrop { background:rgba(0,0,0,.6); }"
        ".col-def-modal h3 { margin:0 0 8px; }"
        ".col-def-modal p { margin:0 0 12px; line-height:1.5; }"
        ".training-data-table { font-size:12px; }"
        ".training-data-table th { white-space:nowrap; }"
        ".training-data-table td { white-space:nowrap; }"
        "</style>"
    )
    out.append(
        "<script>(function(){"
        "var defs = " + _json.dumps(defs) + ";"
        "var modal = document.getElementById('col-def-modal');"
        "if (!modal || !modal.showModal) return;"
        "document.querySelectorAll('.col-def-btn').forEach(function(b){"
        "  b.addEventListener('click', function(){"
        "    var d = defs[b.dataset.col];"
        "    if (!d) return;"
        "    document.getElementById('col-def-title').textContent = d.label;"
        "    document.getElementById('col-def-body').textContent = d['def'];"
        "    modal.showModal();"
        "  });"
        "});"
        "})();</script>"
    )

    out.append("</section>")
    return "".join(out)


# Path to the live executor's sim_state — used to enrich Kalshi-only
# rows with full player names + tournament metadata recorded at order-
# placement time (Kalshi's /portfolio/settlements doesn't carry those).
_LIVE_SIM_STATE_PATH = Path(
    "/root/tennis-forecast/data/outputs-live/sim_state.json"
)


def _load_event_ticker_enrichment() -> Dict[str, Dict[str, Any]]:
    """Build an ``event_ticker -> match_metadata`` lookup from the
    live executor's sim_state.json closed_positions. The bot
    recorded each closed position with full player names, tournament,
    and surface at order-placement time — exactly the fields the
    Kalshi outcomes table lacks.

    Cheap to call per render: file is ~500KB and we read once. Falls
    back to an empty dict if the file isn't present (e.g. local dev).
    """
    if not _LIVE_SIM_STATE_PATH.exists():
        return {}
    try:
        with _LIVE_SIM_STATE_PATH.open("r", encoding="utf-8") as f:
            state = json.load(f) or {}
    except (OSError, json.JSONDecodeError):
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for p in (state.get("closed_positions") or []):
        mid = p.get("match_id") or ""
        if not mid:
            continue
        # Prefer richer records: a later close on the same match_id
        # may have player names that an earlier one was missing.
        existing = out.get(mid) or {}
        merged = {
            "player_a": p.get("player_a") or existing.get("player_a"),
            "player_b": p.get("player_b") or existing.get("player_b"),
            "side_player": p.get("side_player") or existing.get("side_player"),
            "tournament": (p.get("tournament")
                            or existing.get("tournament")),
            "surface": p.get("surface") or existing.get("surface"),
            "event_title": (p.get("event_title")
                              or existing.get("event_title")),
        }
        out[mid] = merged
    return out


def _build_kalshi_only_rows(tour_filter: str | None) -> List[Dict[str, Any]]:
    """Return Kalshi bets that don't have a matching training_matches
    row, shaped to fit the same column layout. Used on page 1 so the
    combined table includes the bot's live activity even when the
    underlying Sackmann panel hasn't caught up yet.

    For each unmatched Kalshi outcome:
      * Date is decoded from the ``YYMMMDD`` segment of event_ticker
      * Tour from the ticker prefix (KX(ATP|WTA)MATCH)
      * Player names default to the 3-letter tricodes since we don't
        have full names without the original watchlist; opens the
        door to "Player A: KAS / Player B: KES" rendering
      * Bet columns populated from the outcome record itself
      * Everything else (raw attrs, engineered features) stays None
        and renders as ``—`` — the model didn't see these matches
    """
    if not _TRAINING_DB_PATH.exists():
        return []
    import sqlite3
    from datetime import datetime
    out_rows: List[Dict[str, Any]] = []
    enrichment = _load_event_ticker_enrichment()
    try:
        conn = sqlite3.connect(str(_TRAINING_DB_PATH),
                                check_same_thread=False)
        try:
            cur = conn.execute(
                "SELECT ticker, event_ticker, side_player, other_player, "
                "market_result, settle_value, won, entry_price, "
                "settle_price, realized_pnl, fee_cost, closed_at "
                "FROM kalshi_outcomes ORDER BY closed_at DESC"
            )
            cols = [c[0] for c in cur.description]
            kalshi_records = [dict(zip(cols, r)) for r in cur.fetchall()]
            # Get the set of (date, tricodes) already covered by training
            # rows so we don't duplicate when both are present.
            cur2 = conn.execute(
                "SELECT tourney_date, player_a, player_b FROM training_matches"
            )
            covered: set[tuple[str, frozenset]] = set()
            for date, pa, pb in cur2.fetchall():
                if not date or not pa or not pb:
                    continue
                covered.add((date, frozenset({
                    _player_tricode(pa), _player_tricode(pb)
                })))
        finally:
            conn.close()
    except sqlite3.OperationalError:
        return []

    for ko in kalshi_records:
        ev = ko.get("event_ticker") or ""
        if "-" not in ev:
            continue
        prefix, tail = ev.split("-", 1)
        if len(tail) < 7:
            continue
        try:
            dt = datetime.strptime(tail[:7], "%y%b%d")
        except ValueError:
            continue
        date = dt.strftime("%Y-%m-%d")
        # Derive tour from prefix.
        tour = ("ATP" if "ATPMATCH" in prefix
                else "WTA" if "WTAMATCH" in prefix else None)
        if tour_filter and tour != tour_filter:
            continue
        sp = ko.get("side_player") or ""
        op = ko.get("other_player") or ""
        pair = frozenset({sp, op})
        if (date, pair) in covered:
            # The training data has this match — it'll be decorated
            # with the Kalshi bet info in the main loop below.
            continue
        # Resolve the actual winner from Kalshi's settled result. Our
        # side is ``side_player``; ``won == 1`` means our side won,
        # which means the side_player tricode won. Non-binary results
        # (scalar / void / walkover) leave the winner as unknown
        # rather than guessing.
        won_v = ko.get("won")
        market_result = (ko.get("market_result") or "").lower()
        if market_result not in ("yes", "no"):
            winner = None
        elif won_v == 1:
            winner = sp
        elif won_v == 0:
            winner = op
        else:
            winner = None
        # Pull the bot's recorded metadata if we have it. Maps the
        # tricodes back to full player names + tournament + surface
        # so the table shows e.g. "Frances Tiafoe" / "Hard" / "Roland
        # Garros" instead of just "TIA" / "—" / "—".
        meta = enrichment.get(ev, {}) or {}
        full_a, full_b = meta.get("player_a"), meta.get("player_b")
        side_full = meta.get("side_player") or ""
        # Decide which side is A vs B by matching side_player full
        # name to the recorded side_player tricode.
        if full_a and full_b:
            a_last_initial = _player_tricode(full_a)
            if side_full and side_full == full_a:
                # full_a is the side we bet on (sp)
                player_a, player_b = full_a, full_b
            elif side_full and side_full == full_b:
                player_a, player_b = full_b, full_a
            elif a_last_initial == sp:
                player_a, player_b = full_a, full_b
            else:
                player_a, player_b = full_b, full_a
        else:
            player_a, player_b = sp, op
        # Recompute winner against the resolved full names.
        if market_result not in ("yes", "no"):
            winner = None
        elif won_v == 1:
            winner = player_a if side_full == player_a or sp == _player_tricode(
                player_a) else (
                player_a if sp != _player_tricode(player_b) else player_b
            )
            # Simpler: side_player (our side) won. Map sp tricode to
            # whichever of (player_a, player_b) has that initial.
            if _player_tricode(player_a) == sp:
                winner = player_a
            elif _player_tricode(player_b) == sp:
                winner = player_b
            else:
                winner = player_a  # safe default — won_v=1 means our side won
        elif won_v == 0:
            # Our side lost → the OTHER side won.
            if _player_tricode(player_a) == sp:
                winner = player_b
            elif _player_tricode(player_b) == sp:
                winner = player_a
            else:
                winner = player_b
        else:
            winner = None
        out_rows.append({
            "tourney_date": date,
            "tourney_name": (meta.get("tournament")
                              or meta.get("event_title")
                              or ev),
            "tour": tour,
            "surface": meta.get("surface"),
            "level": None,
            "round": None,
            "draw_size": None,
            "best_of": None,
            "player_a": player_a,
            "player_b": player_b,
            "winner": winner,
        })
    return out_rows


def _player_tricode(full_name: str | None) -> str:
    """3-letter uppercase code used in Kalshi's ticker encoding.
    Built from the player's last name (the final whitespace-separated
    token). Returns ``""`` for empty / unknown inputs so the lookup
    dict's get() falls through cleanly."""
    if not full_name:
        return ""
    parts = str(full_name).strip().split()
    if not parts:
        return ""
    return parts[-1][:3].upper()


