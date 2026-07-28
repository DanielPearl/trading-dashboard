"""Home / summary / active-bets / seasons / history panel renderers."""
from __future__ import annotations

import html
import json
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from typing import List
from typing import Tuple
from .data import bot_regime_status
from .fmt import (
    _ev_status,
    _market_date_label,
    _match_text_from_ticker,
    _side_tricode_from_ticker,
    _sport_event_label,
    cents_or_dash,
    fmt_signed_cents,
    fmt_underlying,
    kalshi_fee_cents,
    minutes_to_close_from_ticker,
    question_str,
    ticker_cell_html,
    time_to_close_str,
)


# --------------------------------------------------------------------------- #
# Section helpers
# --------------------------------------------------------------------------- #

PERIOD_OPTIONS = [
    ("day", "Day", 1),
    ("week", "Week", 7),
    ("month", "Month", 30),
    ("year", "Year", 365),
    ("all", "All-time", None),
]


def _period_days(period_key: str) -> int | None:
    """Map ``?period=X`` query value to the rolling-window day count
    used by the SQL filters. Unknown / missing → None (lifetime).
    """
    for key, _label, days in PERIOD_OPTIONS:
        if key == period_key:
            return days
    return None


def _render_period_filter(out: List[str], period_key: str,
                            current_bot: str = "",
                            tab_key: str = "home") -> None:
    """Period filter dropdown (Day · Week · Month · Year · All-time).
    Uses the same wrapper + select styling as the Watchlist tab's bot
    selector so the three filters across the dashboard read as one
    consistent UI control.
    """
    select_id = f"period-select-{html.escape(tab_key)}"
    out.append("<div class='bot-filter-bar'>")
    out.append(f"<label for='{select_id}' class='filter-label'>"
               f"Period</label>")
    out.append(
        f"<select id='{select_id}' class='bot-select' "
        f"data-period-select>"
    )
    for key, label, _days in PERIOD_OPTIONS:
        bot_qs = (f"&bot={html.escape(current_bot)}"
                  if current_bot else "")
        tab_qs = (f"&tab={html.escape(tab_key)}"
                  if tab_key and tab_key != "home" else "")
        href = f"?period={key}{bot_qs}{tab_qs}"
        sel = " selected" if key == period_key else ""
        out.append(
            f"<option value='{html.escape(href)}'{sel}>"
            f"{html.escape(label)}</option>"
        )
    out.append("</select>")
    out.append("</div>")


def _week_change_pct(rollup: dict) -> Tuple[str, str]:
    """Return (text, css_class) for the Home tab's "Week change" card.

    Compares lifetime net P&L now to where it stood seven days ago
    (now − this_week_pnl). Positive = account is up vs. last week.
    Returns ('—', 'gray') when there's no baseline to compare against
    (week-ago P&L was zero) so the card doesn't show a misleading ∞%.
    """
    this_week = rollup.get("this_week_pnl_cents", 0) or 0
    lifetime = rollup.get("net_pnl_cents", 0) or 0
    week_ago = lifetime - this_week
    if week_ago == 0:
        return ("—", "gray")
    pct = (this_week / abs(week_ago)) * 100.0
    cls = "green" if pct > 0 else ("red" if pct < 0 else "gray")
    sign = "+" if pct > 0 else ("−" if pct < 0 else "")
    return (f"{sign}{abs(pct):.1f}%", cls)


def _render_home_summary_cards(out: List[str], rollup: dict) -> None:
    """Home tab headline cards (user spec 2026-07-20, left→right):
    Cash · Predictions (portfolio incl. positions) · Change (24h) ·
    Unrealized return · Money spent · Potential gain. The first four
    come straight from the Kalshi account (same numbers the Kalshi
    app shows); the last two mirror the Active-bets column totals.
    """
    from .kalshi_client import get_portfolio_overview
    try:
        ov = get_portfolio_overview()
    except Exception:  # noqa: BLE001 — cards degrade to em-dashes
        ov = {}

    def _dollars(c) -> str:
        return f"${c/100:,.2f}" if c is not None else "—"

    potential = rollup.get("potential_gain_cents", 0)
    chg = ov.get("change_24h_cents")
    chg_cls = ("green" if (chg or 0) > 0
               else ("red" if (chg or 0) < 0 else "gray"))
    chg_txt = fmt_signed_cents(chg) if chg is not None else "—"
    unreal = ov.get("unrealized_cents")
    unreal_cls = ("green" if (unreal or 0) > 0
                  else ("red" if (unreal or 0) < 0 else "gray"))
    out.append("<div class='row'>")
    out.append(f"<div class='card'><div class='label' "
               f"title='Kalshi account cash balance — money not "
               f"currently riding on any position.'>"
               f"Cash</div>"
               f"<div class='value' id='card-cash'>"
               f"{_dollars(ov.get('cash_cents'))}</div></div>")
    out.append(f"<div class='card'><div class='label' "
               f"title='Total portfolio value: cash + every open "
               f"position at its current market price. Matches the "
               f"Kalshi app.'>"
               f"Predictions</div>"
               f"<div class='value' id='card-portfolio'>"
               f"{_dollars(ov.get('portfolio_cents'))}</div></div>")
    chg_tip = ("Portfolio value now vs ~24 hours ago (from the "
               "rolling snapshot log)." if not ov.get("change_24h_estimated")
               else "Net realized P&amp;L settled in the last 24 hours "
                    "(snapshot baseline still accumulating; switches to "
                    "full portfolio change after a day).")
    out.append(f"<div class='card'><div class='label' "
               f"title='{chg_tip}'>"
               f"Change (24h)</div>"
               f"<div class='value {chg_cls}' id='card-change-24h'>"
               f"{chg_txt}</div></div>")
    out.append(f"<div class='card'><div class='label' "
               f"title='Kalshi&apos;s own valuation of every open "
               f"position (the portfolio_value field on "
               f"/portfolio/balance) — Predictions minus Cash.'>"
               f"Position</div>"
               f"<div class='value' id='card-position'>"
               f"{_dollars(ov.get('positions_value_cents'))}</div></div>")
    out.append(f"<div class='card'><div class='label' "
               f"title='Open positions marked to market minus their "
               f"cost basis — profit that exists on paper but has not "
               f"settled.'>"
               f"Unrealized return</div>"
               f"<div class='value {unreal_cls}' id='card-unrealized'>"
               f"{fmt_signed_cents(unreal) if unreal is not None else chr(8212)}"
               f"</div></div>")
    out.append(f"<div class='card'><div class='label' "
               f"title='Sum of the Potential gain column across the "
               f"Active bets table ((100 - entry) x contracts - "
               f"entry fee) — what settles if every open position "
               f"wins.'>"
               f"Potential gain</div>"
               f"<div class='value green' id='card-potential-earnings'>"
               f"+{fmt_signed_cents(potential).lstrip('+')}</div></div>")
    out.append("</div>")


def _render_summary_cards(out: List[str], rollup: dict,
                           id_suffix: str = "",
                           show_closed_contracts: bool = False) -> None:
    """History headline cards (user spec 2026-07-20, left→right):
    Cash · Predictions · Total bets · Money spent · P&L · Win %.

    The first two are the live Kalshi account (identical to the Home
    cards, never affected by the period filter). The last four are
    computed from the period-filtered ledger the caller passes in
    ``rollup`` — the filter above these cards drives them.
    """
    from .kalshi_client import get_portfolio_overview
    try:
        ov = get_portfolio_overview()
    except Exception:  # noqa: BLE001
        ov = {}

    def _dollars(c) -> str:
        return f"${c/100:,.2f}" if c is not None else "—"

    total_bets = rollup.get("total_closed", 0)
    money_spent_c = int(round((rollup.get("total_staked") or 0) * 100))
    net_c = int(round((rollup.get("total_realized_pnl") or 0) * 100))
    pnl_cls = "green" if net_c > 0 else ("red" if net_c < 0 else "gray")
    win_rate = rollup.get("win_rate")
    win_cls = ("green" if (win_rate or 0) > 0.5
               else ("red" if win_rate is not None and total_bets
                     and win_rate < 0.5 else "gray"))
    win_str = (f"{win_rate*100:.0f}%"
               if win_rate is not None and total_bets else "—")
    out.append("<div class='row compact'>")
    out.append(f"<div class='card'><div class='label' "
               f"title='Kalshi account cash balance right now — not "
               f"affected by the period filter.'>"
               f"Cash</div>"
               f"<div class='value' id='card-cash{id_suffix}'>"
               f"{_dollars(ov.get('cash_cents'))}</div></div>")
    out.append(f"<div class='card'><div class='label' "
               f"title='Total portfolio value: cash + open positions "
               f"at market. Not affected by the period filter.'>"
               f"Predictions</div>"
               f"<div class='value' id='card-portfolio{id_suffix}'>"
               f"{_dollars(ov.get('portfolio_cents'))}</div></div>")
    out.append(f"<div class='card'><div class='label' "
               f"title='Settled bets in the selected period.'>"
               f"Total bets</div>"
               f"<div class='value' id='card-closed-bets{id_suffix}'>"
               f"{total_bets}</div></div>")
    out.append(f"<div class='card'><div class='label' "
               f"title='Total cost basis of the period&apos;s settled "
               f"bets (entry price x contracts).'>"
               f"Money spent</div>"
               f"<div class='value' id='card-money-spent{id_suffix}'>"
               f"{fmt_signed_cents(-money_spent_c)}</div></div>")
    out.append(f"<div class='card'><div class='label' "
               f"title='Net realized profit and loss on the "
               f"period&apos;s settled bets, after fees.'>"
               f"P&amp;L</div>"
               f"<div class='value {pnl_cls}' id='card-net-pnl{id_suffix}'>"
               f"{fmt_signed_cents(net_c)}</div></div>")
    out.append(f"<div class='card'><div class='label' "
               f"title='Wins divided by settled bets in the selected "
               f"period.'>"
               f"Win %</div>"
               f"<div class='value {win_cls}' id='card-win-pct{id_suffix}'>"
               f"{win_str}</div></div>")
    out.append("</div>")


def _render_summary(out: List[str], rollup: dict, active_bets: List[dict],
                    history: List[dict],
                    period_key: str = "all",
                    current_bot: str = "",
                    available_bots: List[dict] | None = None,
                    hedge_cfg: dict | None = None) -> None:
    """Section 1 — global cross-bot summary. The dropdown above the
    headline cards is a Bot navigator: selecting any bot jumps to its
    Watchlist tab so the user can dive into per-bot detail without
    hunting through the bot-card grid below. The summary cards
    themselves stay scoped to All-time totals (the period filter
    moved off Home — the History tab still has its period filter).
    """
    period_label = next(
        (lbl for k, lbl, _ in PERIOD_OPTIONS if k == period_key),
        "All-time",
    )
    bets_made = rollup.get("period_bets_made", 0)
    net = rollup.get("period_net_pnl_cents", 0)
    pnl_cls = "green" if net > 0 else ("red" if net < 0 else "gray")
    win_pct = rollup.get("period_win_pct", 0.0)
    has_closed = (rollup.get("period_wins", 0)
                  + rollup.get("period_losses", 0)) > 0
    win_cls = ("green" if win_pct > 0.5
               else ("red" if has_closed and win_pct < 0.5 else "gray"))
    win_pct_str = f"{win_pct*100:.0f}%" if has_closed else "—"

    out.append("<div class='section'><h2>Summary — across all bots</h2>"
               "<div class='body summary-body'>")

    # Bot-jump dropdown moved above the tab bar (per user request) so
    # it applies to every tab in one place.

    # ── Headline cards ────────────────────────────────────────────────
    _render_home_summary_cards(out, rollup)

    # Active bets list — same table used in the per-bot view below.
    # Same circle-i info button as on the Watchlist tab — opens the
    # shared rules popup ("What does the bot need before it'll buy?")
    # using window.__BUY_CRITERIA__ as the data source.
    out.append(
        "<h3 class='subhead' "
        "style='display:flex;align-items:center;gap:8px;'>"
        "Active bets "
        "<button type='button' class='criteria-rules-btn' "
        "title=\"What does the bot need before it'll buy?\">i</button>"
        "</h3>"
    )
    # Scroll container — keeps the Summary's active-bets table from
    # pushing the bot-card grid off-screen when many bots have
    # positions open at once. Max-height was picked so ~6 rows are
    # visible before the user has to scroll; matches the watchlist
    # scroll idiom used elsewhere on the page.
    out.append("<div class='summary-active-scroll'>")
    _render_active_bets_table(out, active_bets,
                                empty_msg="No active bets right now.",
                                hedge_cfg=hedge_cfg,
                                sport_style=True)
    out.append("</div>")

    out.append("</div></div>")


def _render_notifications_panel(out: List[str],
                                  limit: int = 5) -> None:
    """Recent auto-pause notifications from the regime monitor.

    Reads the tail of ``data/regime_notifications.jsonl`` and renders
    a compact panel above the bot card grid when there's at least one
    entry. Silent (no rendering) when the file is empty or missing —
    a no-news-is-good-news posture so the Home tab stays calm.
    """
    from . import regime_monitor
    notes = regime_monitor.read_notifications(limit=limit)
    if not notes:
        return
    out.append("<div class='notifications-panel'>")
    out.append(
        "<div class='notifications-head'>"
        "<span class='notifications-title'>Recent auto-pauses</span>"
        "<span class='small gray'>The regime monitor auto-disabled "
        "these bots after 3 consecutive 30-day windows of negative "
        "P&amp;L. Use the bot card toggle to resume.</span></div>"
    )
    out.append("<ul class='notifications-list'>")
    for n in notes:
        ts = (n.get("ts") or "")[:19].replace("T", " ")
        bot_name = n.get("bot_name") or n.get("bot_key") or "—"
        reason = n.get("reason") or ""
        out.append(
            f"<li><span class='notification-ts'>{html.escape(ts)}</span>"
            f"<span class='notification-bot'>"
            f"{html.escape(str(bot_name))}</span>"
            f"<span class='notification-reason'>"
            f"{html.escape(reason)}</span></li>"
        )
    out.append("</ul></div>")


def _render_bot_cards(out: List[str], rollup: dict,
                        bot_models: List[dict] | None,
                        period_label: str) -> None:
    """Per-bot card grid for the Performance tab. Compact, clickable —
    each card is an anchor to the bot's Watchlist tab. Cards align on
    a fixed grid (auto-fit minmax 280px) so they share row + column
    edges. Contract rules live on the Watchlist tab to keep these
    cards skimmable.

    Top-right slot carries an on/off toggle (active / paused state).
    Bot name + current underlying value + series_ticker stack down
    the left so the most-glanceable info (which bot, what's its
    current quote, which Kalshi series) sits together. Paused cards
    get a dimmed style + PAUSED badge so disabled bots are visually
    distinct from the running ones.
    """
    if not bot_models:
        out.append("<div class='empty'>No bot data yet.</div>")
        return

    # Pull the latest per-bot enable state so the toggles render in
    # their current position. Defaults to enabled = True for bots
    # without a stored state.
    from . import bot_state
    bot_states = bot_state.get_all_states()

    # Per-bot perf rows — used for the period-scoped Gain/loss cell.
    perf_by_name = {name: s for name, s in (rollup.get("per_bot") or [])}

    def _fmt_pct(v, decimals=0):
        if v is None:
            return "—"
        try:
            return f"{float(v)*100:.{decimals}f}%"
        except (TypeError, ValueError):
            return "—"

    out.append("<div class='bot-cards-grid'>")
    for entry in bot_models:
        b = entry.get("bot") or {}
        m = entry.get("model") or {}
        name = b.get("name", "—")
        bot_key = b.get("key", "")
        # Upper-right meta slot. For Kalshi sim.db bots this is the
        # series_ticker prefix (e.g. "KXNBAGAME"). JSON-source sport bots
        # (tennis, table-tennis) read the same series_ticker field from
        # config — set it to the bot's label of choice (e.g.
        # "BASELINEBREAK" for tennis, "TABLETENNIS" for table-tennis).
        series_ticker = b.get("series_ticker") or "—"
        # Period-scoped net P&L from this bot's per-bot summary row.
        perf = perf_by_name.get(name, {})
        gain_loss = perf.get("period_net_pnl_cents", 0) or 0
        gl_cls = ("green" if gain_loss > 0
                   else ("red" if gain_loss < 0 else "gray"))
        gl_str = fmt_signed_cents(gain_loss)
        # Each card is a link to the bot's Models tab — the deeper
        # per-bot view (feature importance, calibration, confusion
        # matrix, all features used to make decisions). Tennis routes
        # through its own page since its model has its own renderer.
        if not bot_key:
            href = "#"
        elif b.get("dashboard_type") in ("sport", "survivor", "billboard", "reality"):
            href = f"?bot={html.escape(bot_key)}&tab=models"
        else:
            href = f"?tab=models&bot={html.escape(bot_key)}"

        # Compute drift-badge HTML (if any) up-front so it can be
        # rendered inline with the bot name in the card header.
        # Drift = |training accuracy − live actual-win-%| > 10pp on
        # n ≥ 10 closed bets.
        # Tennis is exempt from the drift badge — its paper-trade
        # ledger settles probabilistically and the user explicitly
        # asked not to surface drift on the tennis card.
        ACTUAL_WIN_MIN_N = 10
        DRIFT_PP_THRESHOLD = 0.10
        drift_html = ""
        # Natural-gas is also exempt per user request — the bot's
        # "training accuracy" is a per-strike grid average that doesn't
        # line up apples-to-apples with the live actual-win-%, so the
        # drift badge fires spuriously on every load.
        _drift_exempt = b.get("dashboard_type") in ("sport", "survivor", "billboard", "reality") \
            or bot_key in ("natural-gas", "hormuz")
        if m and not _drift_exempt:
            a_wins_pre = int(m.get("actual_wins") or 0)
            a_losses_pre = int(m.get("actual_losses") or 0)
            a_total_pre = a_wins_pre + a_losses_pre
            try:
                acc_train_pre = float(m.get("classifier_accuracy") or 0)
            except (TypeError, ValueError):
                acc_train_pre = 0.0
            if a_total_pre >= ACTUAL_WIN_MIN_N and acc_train_pre > 0:
                a_pct_pre = a_wins_pre / a_total_pre
                if abs(acc_train_pre - a_pct_pre) > DRIFT_PP_THRESHOLD:
                    gap_pp = int(round(abs(acc_train_pre - a_pct_pre) * 100))
                    drift_html = (
                        f"<span class='drift-badge' "
                        f"title='Training accuracy ({acc_train_pre*100:.0f}%) "
                        f"and live actual-win-% ({a_pct_pre*100:.0f}%) differ "
                        f"by {gap_pp}pp on {a_total_pre} closed bets — model "
                        f"may have drifted; a retrain is likely overdue.'"
                        f">⚠ drift</span>"
                    )
        # Regime status pill — rolling edge-health check sourced from
        # the bot's last 90 days of closed bets. Sits inline with the
        # bot name so the user can scan the grid for which bots are
        # currently making money. Tennis-style adapters don't have
        # the schema we need, so they get no pill (rather than a
        # misleading "no data" badge on every load).
        regime_html = ""
        if b.get("dashboard_type") not in ("sport", "survivor", "billboard", "reality"):
            regime = bot_regime_status(b.get("db_path") or "")
            if regime.get("status") and regime["status"] != "gray":
                regime_html = (
                    f"<span class='regime-pill regime-{regime['status']}' "
                    f"title='{html.escape(regime.get('reason') or '')}'>"
                    f"{html.escape(regime.get('label') or '')}</span>"
                )

        # Forecast-staleness badge — flags when the bot's stored
        # ``current_gas_price`` model snapshot has drifted away from
        # the live Kalshi-implied spot (50¢-crossover strike on the
        # series' most-imminent event). Triggered above a $0.20
        # gap, which is roughly 1σ for the natgas residual and well
        # outside normal noise. Catches the "EIA feed lag" failure
        # mode where the bot's price-input is days behind reality
        # and its forecast (and every model_prob_yes) is anchored
        # to a stale level.
        #
        # Only applies to sim.db-style bots that record a scalar
        # underlying — same exclusion as the regime pill above.
        staleness_html = ""
        # Hormuz is exempt: its stored current_gas_price is a *forecast*
        # (the predicted weekly peak), not a current observed level, so
        # the "forecast vs market-implied" gap the badge measures is the
        # bot's intended edge — not a stale upstream feed. (Same spirit
        # as the natural-gas drift-badge exemption above.)
        if (b.get("dashboard_type") not in
                ("sport", "survivor", "billboard", "reality", "whale", "rules-parser")
                and bot_key != "hormuz"
                and m and m.get("current_gas_price") is not None
                and b.get("series_ticker")):
            try:
                bot_price = float(m["current_gas_price"])
                divisor = float((b.get("display") or {}).get("divisor", 1.0)) or 1.0
                bot_price_in_market_units = bot_price / divisor
                from . import kalshi_client as _kc
                implied, _err = _kc.get_implied_spot(b["series_ticker"])
            except Exception:  # noqa: BLE001
                bot_price_in_market_units = None
                implied = None
            if implied is not None and bot_price_in_market_units is not None:
                # Use $0.20 absolute threshold for natgas-shape series;
                # this is roughly 1σ of the bot's residual and well
                # outside normal intra-day noise. For markets where
                # the scalar isn't a $/MMBTU price (e.g. jobless claims
                # in thousands), the same absolute gap reads as a
                # different number of "units" — acceptable as a v1
                # heuristic, can be refined per-bot later.
                gap = abs(implied - bot_price_in_market_units)
                if gap >= 0.20:
                    fmt = lambda v: f"{v:.2f}"  # noqa: E731
                    tip = (
                        f"Bot's stored current value: "
                        f"{fmt(bot_price_in_market_units)} · "
                        f"Live Kalshi-implied: {fmt(implied)} · "
                        f"Gap: {gap:+.2f} — model may be reading a "
                        f"stale upstream data feed."
                    )
                    staleness_html = (
                        f"<span class='stale-badge' "
                        f"title='{html.escape(tip)}'>⚠ stale</span>"
                    )
        # Toggle state for this bot — defaults to enabled = True.
        bot_state_entry = bot_states.get(bot_key) or {}
        bot_enabled = bool(bot_state_entry.get("enabled", True))
        card_classes = ["bot-card"]
        if not bot_enabled:
            card_classes.append("bot-card-paused")
        # Card header layout per user spec: bot name on top, ticker
        # directly below. The previous underlying-value "price" line
        # was dropped — surfacing $4.50 / 211K / 0.38pp on a model-
        # performance card was confusing (it's not the bot's *score*,
        # it's the upstream market value), and the bots that don't
        # track a scalar (tennis / survivor) were forced to render
        # "—" there anyway.
        paused_badge = (
            "<span class='paused-badge' title='Bot is paused — toggle "
            "on to resume taking bets.'>PAUSED</span>"
            if not bot_enabled else ""
        )
        out.append(
            f"<a class='{' '.join(card_classes)}' href='{href}' "
            f"data-bot-key='{html.escape(bot_key)}'>"
        )
        out.append("<div class='bot-card-head'>")
        out.append("<div class='bot-card-head-left'>")
        out.append(
            f"<div class='bot-name'>{html.escape(name)}{regime_html}"
            f"{drift_html}{staleness_html}{paused_badge}</div>"
        )
        out.append(
            f"<div class='bot-meta'>{html.escape(series_ticker)}</div>"
        )
        out.append("</div>")
        # Toggle switch — click is intercepted by JS so the parent
        # anchor's navigation doesn't fire. data-* attributes hold the
        # mutable state the JS flips.
        toggle_attrs = (
            f"data-bot-key='{html.escape(bot_key)}' "
            f"data-enabled='{'1' if bot_enabled else '0'}' "
            f"aria-pressed='{'true' if bot_enabled else 'false'}' "
            "type='button' onclick='toggleBotState(event, this)'"
        )
        out.append(
            f"<button class='bot-toggle' {toggle_attrs}>"
            f"<span class='bot-toggle-track'>"
            f"<span class='bot-toggle-knob'></span>"
            f"</span>"
            f"</button>"
        )
        out.append("</div>")

        if not m:
            out.append("<dl><dt class='gray'>Model</dt>"
                       "<dd class='gray' style='grid-column:span 3;text-align:left;'>"
                       "no snapshot yet</dd></dl>")
        else:
            a_wins = int(m.get("actual_wins") or 0)
            a_losses = int(m.get("actual_losses") or 0)
            a_total = a_wins + a_losses
            # Show the real percentage at any sample size (per user
            # request). n=0 still shows "—" to distinguish "no data
            # yet" from "0%". The drift-badge logic above keeps its
            # n ≥ 10 guard since drift needs a meaningful sample.
            a_pct = a_wins / a_total if a_total > 0 else None
            if a_total > 0:
                a_str = f"{a_pct*100:.0f}%"
                a_cls = ("green" if a_pct > 0.55
                         else ("red" if a_pct < 0.45 else ""))
            else:
                a_str = "—"
                a_cls = "gray"
            features = int(m.get("feature_count") or 0)
            # Sample sizes: training-set rows the model fit on, and the
            # held-out test rows the headline metrics were measured on.
            # Both come from model_snapshots (sqlite bots) or metrics.json
            # (tennis-style adapters via fetch_latest_model). Cell reads
            # "—" when a bot hasn't been retrained since the schema added
            # the column.
            def _fmt_n(v):
                try:
                    return f"{int(v):,}" if v else "—"
                except (TypeError, ValueError):
                    return "—"
            train_str = _fmt_n(m.get("rows_train"))
            test_str = _fmt_n(m.get("rows_test"))
            out.append("<dl>")
            out.append(f"<dt>Accuracy</dt><dd>{_fmt_pct(m.get('classifier_accuracy'), 1)}</dd>"
                        f"<dt>F1</dt><dd>{_fmt_pct(m.get('training_f1'))}</dd>")
            out.append(f"<dt>Precision</dt><dd>{_fmt_pct(m.get('training_precision'))}</dd>"
                        f"<dt>ROC AUC</dt><dd>{_fmt_pct(m.get('training_roc_auc'))}</dd>")
            out.append(f"<dt>Recall</dt><dd>{_fmt_pct(m.get('training_recall'))}</dd>"
                        f"<dt>Features</dt><dd>{features}</dd>")
            out.append(f"<dt title='Training-set size — number of historical observations the model fit on. More rows = more market regimes covered.'>Train rows</dt>"
                        f"<dd>{train_str}</dd>"
                        f"<dt title='Held-out test-set size — observations the headline metrics were measured on.'>Test rows</dt>"
                        f"<dd>{test_str}</dd>")
            out.append(f"<dt>Actual win %</dt><dd class='{a_cls}'>{a_str}</dd>"
                        f"<dt>P&amp;L</dt><dd class='{gl_cls}'>{gl_str}</dd>")
            # Data source — pulled from dashboard.yaml. Spans the full
            # row width so long descriptions don't crowd a metric cell.
            ds = b.get("data_source")
            if ds:
                out.append(
                    f"<dt title='Where the model's training data comes "
                    f"from. Real public source — never synthetic.'>Source</dt>"
                    f"<dd style='grid-column:span 3;text-align:left;font-size:0.85em;color:var(--muted);'>"
                    f"{html.escape(str(ds))}</dd>"
                )
            out.append("</dl>")

        # Footer hints at the click affordance — same idiom as the
        # ticker cells in the watchlist (subtle "go here" signal).
        out.append("<div class='bot-card-foot'>"
                   "<span>View model</span>"
                   "<span class='arrow'>›</span>"
                   "</div>")
        out.append("</a>")  # /bot-card
    out.append("</div>")  # /bot-cards-grid


def _render_active_bets_table(out: List[str], bets: List[dict],
                              empty_msg: str = "No active bets.",
                              show_bot: bool = True,
                              chart_link: bool = False,
                              hedge_cfg: dict | None = None,
                              hide_settled: bool = True,
                              watchlist: List[dict] | None = None,
                              event_title: str | None = None,
                              is_sport_bot: bool = False,
                              display: dict | None = None,
                              sport_style: bool = False) -> None:
    """Shared renderer used by both Section 1 (cross-bot summary) and
    the per-bot view inside the Watchlist tab. Columns:
        Opened | [Bot] | Ticker | Question | Contracts | Side
        | Entry cost | Current | Potential gain | Closes in
    The Bot column is skipped when ``show_bot`` is False (per-bot view
    where the bot is implied by the surrounding section). Entry cost /
    Current / Potential gain are in dollars (per-position totals).

    ``hedge_cfg`` is accepted for parity with callers but the table no
    longer renders a per-row hedge column — the actual hedge
    execution lives in ``hedge_monitor.py`` which closes any position
    that crosses the configured profit-lock / stop-loss thresholds.
    Once closed, the position drops out of this table and shows up
    on the History tab with ``exit_reason='hedge'``.

    ``hide_settled=True`` (default) filters out positions whose
    Kalshi-ticker-encoded settlement date is more than 1 hour in
    the past — zombie open positions whose bot didn't record the
    settle event. The hedge daemon closes them on its next tick
    so they appear on History; this renderer hides them in the
    interim so the active-bets table reflects only positions the
    bot is actually exposed on.

    ``chart_link=True`` makes each row clickable and stamps the
    chart-overlay attributes (``data-ticker``, ``data-strike``,
    ``data-yes-prob``) so the watchlist hero chart can draw a
    threshold line at the bet's strike (or entry probability for
    sport bots) — the same affordance the strike-ladder rows have.
    """
    # Drop already-settled positions when requested (default). The
    # Summary already pre-filters before calling us; per-bot
    # Watchlist views and the standard renderer rely on this guard.
    if hide_settled:
        bets = [
            b for b in bets
            if (
                (b.get("minutes_to_close")
                 if b.get("minutes_to_close") is not None
                 else minutes_to_close_from_ticker(b.get("ticker"))) or 0
            ) >= -60
        ]
    if not bets:
        out.append(f"<div class='empty'>{html.escape(empty_msg)}</div>")
        return
    # Look-up of the ticker table underneath so the active-bets row
    # mirrors the same title + side text the watchlist shows for the
    # same ticker. Per-bot views pass `watchlist`; the cross-bot
    # Summary tab leaves it None and falls back to the legacy fields
    # stored on each position.
    wl_by_ticker: dict = {
        (w.get("ticker") or ""): w for w in (watchlist or [])
    }
    use_event_title = bool((display or {}).get("watchlist_title_use_event")
                           and event_title)
    bot_th = "<th>Bot</th>" if show_bot else ""
    # Column layout: ``Title`` carries Kalshi's published contract title
    # (the YES question text). ``Side`` carries the YES / NO badge.
    # No separate ``Question`` column — Title already names the
    # contract, and on sport rows it would just restate the matchup.
    tbody_attrs = " id='wl-active-tbody' data-chart-link='1'" if chart_link else ""
    if sport_style:
        # Cross-bot summary in the same structure as the per-bot sport
        # watchlist Active bets table (user 2026-07-10: "the active
        # bets on the homepage should be structured like the active
        # bets on the individual watchlist pages like the tennis
        # watchlist page"). Rules / Total-contracts columns are
        # omitted — position records carry neither the rules text nor
        # market-wide volume.
        # 2026-07-15 (user): mirror the per-bot Active bets column set.
        # Order: Date | [Bot] | Event | Title | Side
        #      | My contracts | Cost | Payout
        #      | Model entry % | Kalshi entry % | Model live % | Kalshi live %
        #      | Closes in
        # Edge / EV columns dropped (Model entry % vs Model live %
        # comparison replaces them for held positions).
        out.append(
            "<table><thead><tr>"
            "<th title='Market date parsed from the Kalshi ticker.'>Date</th>"
            f"{bot_th}"
            "<th title='Competition (MLB, NBA Summer League, PDC, ...)'>Event</th>"
            "<th>Title</th>"
            "<th>Side</th>"
            "<th class='num' title='Number of contracts in this position.'>My contracts</th>"
            "<th class='num' title='Kalshi total cost — entry price × contracts + Kalshi entry fee.'>Cost</th>"
            "<th class='num' title='Potential earnings if this side wins — $1 × contracts settlement payout minus (entry + fee).'>Payout</th>"
            "<th class='num' title='Model % at ENTRY — Pinnacle&apos;s prob for our side at the moment we opened. Static once the trade is on.'>Model entry %</th>"
            "<th class='num' title='Kalshi entry % — the implied probability for our side at the price we paid. Static.'>Kalshi entry %</th>"
            "<th class='num' title='Model % NOW — today&apos;s Pinnacle prob for our side. Compare to Model entry % to see how the line has moved.'>Model live %</th>"
            "<th class='num' title='Live Kalshi market price for our side — updates continuously.'>Kalshi live %</th>"
            "<th class='num' title='Time until the contract resolves'>Closes in</th>"
            "<th></th>"
            f"</tr></thead><tbody{tbody_attrs}>")
    else:
        out.append("<table><thead><tr>"
               f"<th>Opened</th>{bot_th}<th>Ticker</th>"
               "<th>Title</th>"
               "<th>Side</th>"
               "<th class='num' title='Number of contracts in this position — the size of your bet.'>My contracts</th>"
               "<th class='num' title='Model probability for our side at entry — what the model thought before we bet.'>Model prob</th>"
               "<th class='num' title='Implied probability of our side at entry (= entry price in ¢).'>Entry prob</th>"
               "<th class='num' title='Implied probability of our side right now, taken from the market mid.'>Current prob</th>"
               "<th class='num' title='Entry prob × contracts + Kalshi entry fee — total cash out at open'>Entry cost</th>"
               "<th class='num' title='(100¢ − entry) × contracts − entry fee — gross profit if our side wins'>Potential gain</th>"
               "<th class='num' title='Time until the contract resolves'>Closes in</th>"
               "<th></th>"
               f"</tr></thead><tbody{tbody_attrs}>")
    for b in bets:
        opened = (b.get("opened_at") or "")[:19].replace("T", " ")
        side = (b.get("side") or "").upper()
        badge_cls = "badge-yes" if side == "YES" else "badge-no"
        entry = b.get("entry_price_cents") or 0
        contracts = b.get("contracts", 0) or 0
        bot_name = b.get("_bot_name", "—")
        # Question — rendered in the bot's native display units
        # ($/gal, K claims, $/MMBtu) when display config is attached.
        floor = b.get("floor_strike")
        cap = b.get("cap_strike")
        try:
            strike_low = float(floor) if floor is not None else None
        except (TypeError, ValueError):
            strike_low = None
        try:
            strike_high = float(cap) if cap is not None else None
        except (TypeError, ValueError):
            strike_high = None
        direction = "between" if (strike_low is not None
                                   and strike_high is not None) else "above"
        question = question_str(direction, strike_low, strike_high,
                                  display=b.get("_display"))
        # Probability columns — entry prob is just entry_price_cents
        # (1 cent = 1% implied probability for the side bet on).
        # Current prob is the market's view of "this side wins" right
        # now, derived from the mid where available with graceful
        # fallbacks for bots that don't write position_marks.
        entry_prob_pct = entry  # cents == percent
        # Compute mid for the YES side first, then flip for NO bets.
        mid_yes = b.get("mark_mid")
        if mid_yes is None:
            ya = b.get("mark_yes_ask")
            yb = b.get("mark_yes_bid")
            if ya is not None and yb is not None:
                mid_yes = (int(ya) + int(yb)) / 2.0
            elif ya is not None:
                mid_yes = int(ya)
            else:
                # Derive from the opposing side's ask (no_ask ≈ 100−yes)
                na = b.get("mark_no_ask")
                if na is not None:
                    mid_yes = max(0, 100 - int(na))
        if mid_yes is None:
            current_prob_pct = None
        else:
            current_prob_pct = (float(mid_yes) if side == "YES"
                                 else 100.0 - float(mid_yes))
        # Dollar columns — all incorporate Kalshi trading fees:
        #   Entry cost     = entry prob × contracts + entry fee
        #   Potential gain = (100 − entry) × contracts − entry fee
        #                    (entry fee already paid; settlement at
        #                     100¢ has zero exit fee)
        entry_fee_c = kalshi_fee_cents(entry, contracts)
        entry_cost_base = entry * contracts / 100.0
        entry_fee_dollars = entry_fee_c / 100.0
        potential_gain = ((100 - entry) * contracts - entry_fee_c) / 100.0
        # Entry-cost cell shows base − fee inline so the user reads
        # both pieces as cash outflows (a positive fee is still cash
        # leaving the account at open). Tooltip explains.
        entry_cost_cell = (
            f"<td class='num red' title='Entry prob × contracts + "
            f"Kalshi entry fee — total cash out at open'>"
            f"−${entry_cost_base:.2f}"
            f"<span class='entry-fee'> − ${entry_fee_dollars:.2f}</span>"
            f"</td>"
        )
        # Probability cells — both rendered in the default white text;
        # the user can compare entry vs current at a glance without
        # the color cue (which was tracking direction of market move).
        entry_prob_cell = f"<td class='num'>{entry_prob_pct}%</td>"
        if current_prob_pct is None:
            current_prob_cell = "<td class='num gray'>—</td>"
        else:
            current_prob_cell = (
                f"<td class='num' title='Market mid for our side right "
                f"now. Compare to Entry prob to see how the market has "
                f"moved.'>{current_prob_pct:.0f}%</td>"
            )
        mtc = b.get("minutes_to_close")
        # Universal fallback: parse the settlement date out of the
        # ticker (Kalshi encodes ``YYMMMDD`` after the series prefix).
        # Catches tennis paper bets — those sim positions don't record
        # an expected_expiration_time, so the previous tennis-adapter
        # lookup against live_state.json missed them once the match
        # rolled off the live state.
        if mtc is None:
            mtc = minutes_to_close_from_ticker(b.get("ticker"))
        # Sign / color logic for potential gain — usually positive
        # (winning side pays $1 minus entry minus fees), but very
        # high entry prices on extreme strikes can flip negative.
        pg_sign = "+" if potential_gain >= 0 else "−"
        pg_cls  = "green" if potential_gain >= 0 else "red"
        # Bot cell: link to that bot's watchlist tab so the user can
        # jump from a row in the cross-bot active-bets summary into
        # the per-bot detail view in one click. Tennis routes through
        # its own page; the rest land on the standard watchlist tab.
        bot_key = b.get("_bot_key") or ""
        bot_dt = b.get("_dashboard_type") or "standard"
        if bot_key:
            if bot_dt == "sport":
                href = f"?bot={html.escape(bot_key)}&tab=watchlist"
            else:
                href = f"?tab=watchlist&bot={html.escape(bot_key)}"
            bot_link = (f"<a href='{href}' class='bot-link'>"
                        f"{html.escape(bot_name)}</a>")
        else:
            bot_link = html.escape(bot_name)
        bot_td = (f"<td>{bot_link}</td>" if show_bot else "")
        # Model % display — prefer today's fresh Pinnacle value if
        # the rollup builder attached it (``current_model_prob_yes``),
        # else fall back to the entry-time stamped prob. This keeps
        # the Home page's Active bets in sync with the per-bot
        # Watchlist page's Model %, which uses the fresh line
        # (Chiba Marines vs Seibu Lions 2026-07-15: entry-time 68%
        # drifted to fresh Pinnacle 56%; the two pages showed
        # different numbers for the same match until this joined
        # them on the fresh side).
        # The criteria modal ("why was this bet chosen") is a
        # decision snapshot, so it uses the ENTRY-TIME prob directly
        # via ``m_yes_entry`` below.
        m_yes_now = b.get("current_model_prob_yes")
        m_yes_entry = b.get("model_yes_prob_at_entry")
        m_yes = m_yes_now if m_yes_now is not None else m_yes_entry
        k_yes = b.get("kalshi_yes_prob_at_entry")
        # Backfill from decision_json for bots whose schema doesn't
        # have dedicated columns (natural-gas stashes both probs
        # inside the JSON payload). Same fallback fetch_bet_history
        # applies to closed rows.
        if (m_yes is None or k_yes is None) and b.get("decision_json"):
            try:
                _dj = json.loads(b["decision_json"]) if isinstance(
                    b["decision_json"], str) else b["decision_json"]
                if isinstance(_dj, dict):
                    if m_yes is None and _dj.get("model_prob") is not None:
                        m_yes = _dj["model_prob"]
                    if k_yes is None and _dj.get("kalshi_implied_prob") is not None:
                        k_yes = _dj["kalshi_implied_prob"]
            except (TypeError, ValueError, json.JSONDecodeError):
                pass
        # Selected-side probabilities — YES bet uses model_yes / kalshi_yes
        # directly; NO bet uses 1 - model_yes / 1 - kalshi_yes.
        if m_yes is not None:
            try:
                m_yes_f = float(m_yes)
                model_p = m_yes_f if side == "YES" else (1.0 - m_yes_f)
            except (TypeError, ValueError):
                model_p = None
        else:
            model_p = None
        if k_yes is not None:
            try:
                k_yes_f = float(k_yes)
                kalshi_p = k_yes_f if side == "YES" else (1.0 - k_yes_f)
            except (TypeError, ValueError):
                kalshi_p = None
        else:
            kalshi_p = None
        edge_pts = (
            (model_p - kalshi_p) * 100.0
            if (model_p is not None and kalshi_p is not None) else None
        )
        # Criteria modal uses the ENTRY-TIME prob (a decision-snapshot
        # view), independent of what fresh Pinnacle says now.
        if m_yes_entry is not None:
            try:
                _me_f = float(m_yes_entry)
                model_p_entry = _me_f if side == "YES" else (1.0 - _me_f)
            except (TypeError, ValueError):
                model_p_entry = None
        else:
            model_p_entry = None
        edge_pts_entry = (
            (model_p_entry - kalshi_p) * 100.0
            if (model_p_entry is not None and kalshi_p is not None) else None
        )
        criteria = {
            "ticker": b.get("ticker"),
            "side": side,
            "entry": entry,
            "contracts": contracts,
            "question": question,
            "bot": bot_name if show_bot else b.get("_bot_name", ""),
            "model_p": model_p_entry,
            "kalshi_p": kalshi_p,
            "edge_pts": edge_pts_entry,
            "entry_ev": b.get("expected_ev_at_entry"),
            "break_even": b.get("break_even_probability"),
            "opened": opened,
        }
        criteria_json = html.escape(json.dumps(
            criteria, separators=(",", ":"), default=str))
        # Title cell: mirror the watchlist row underneath for the same
        # ticker so the two tables always show identical text. Falls
        # through to the position's stored title and finally to a
        # derived question when no watchlist match is available (the
        # cross-bot Summary tab doesn't supply a watchlist).
        wl_row = wl_by_ticker.get(b.get("ticker") or "")
        if use_event_title:
            title_text = event_title
        elif wl_row and wl_row.get("title"):
            title_text = wl_row.get("title")
        else:
            title_text = b.get("_title") or b.get("title") or ""
        if not title_text:
            match_text = b.get("_match") or _match_text_from_ticker(b.get("ticker"))
            side_player = b.get("_side_player")
            if side_player:
                title_text = (f"{match_text} — bet on {side_player}"
                               if match_text else side_player)
            elif match_text:
                tri = _side_tricode_from_ticker(b.get("ticker"), side)
                title_text = (f"{match_text} — bet on {tri}"
                               if tri else match_text)
            else:
                title_text = question
        # Closes in: for tennis paper bets, _minutes_to_close is provided
        # by the tennis adapter (derived from expected_expiration_time);
        # for Kalshi bots the simulator already supplies minutes_to_close.
        # Chart-overlay data attrs (only emitted when ``chart_link``) —
        # the watchlist hero chart's row-click hook reads these and
        # draws a horizontal threshold line at the bet's strike (non-
        # sport) or entry probability (sport).
        if chart_link:
            tr_attrs = f" data-ticker='{html.escape(b.get('ticker') or '')}'"
            try:
                if strike_low is not None:
                    tr_attrs += f" data-strike='{float(strike_low):.6f}'"
            except (TypeError, ValueError):
                pass
            try:
                # YES bet's entry price = implied YES probability; NO
                # bet's entry price implies (100 - entry)% YES.
                yes_prob = (entry / 100.0 if side == "YES"
                             else (100 - entry) / 100.0)
                tr_attrs += f" data-yes-prob='{yes_prob:.4f}'"
            except (TypeError, ValueError):
                pass
        else:
            tr_attrs = ""
        # Model prob cell — renders the side-adjusted LIVE model
        # probability (Pinnacle NOW, from current_model_prob_yes).
        # Tooltip surfaces the implied edge (model − Kalshi) when
        # both are available.
        if model_p is None:
            model_prob_cell = "<td class='num gray'>—</td>"
        else:
            tip = ""
            if edge_pts is not None:
                tip = (f" title='Model edge {edge_pts:+.1f}pp vs entry "
                       f"price'")
            model_prob_cell = (
                f"<td class='num'{tip}>{model_p*100:.0f}%</td>"
            )
        # Model ENTRY % cell — Pinnacle's prob for our side at the
        # moment we opened. Static once the trade is on.
        if model_p_entry is None:
            model_entry_cell = "<td class='num gray'>—</td>"
        else:
            model_entry_cell = (
                f"<td class='num'>{model_p_entry*100:.0f}%</td>"
            )
        # Kalshi ENTRY % cell — implied prob of our side at the
        # price we actually paid. entry_price_cents is already
        # scoped to the side we bought (tennis convention).
        kalshi_entry_cell = (
            f"<td class='num'>{entry}%</td>" if entry
            else "<td class='num gray'>—</td>"
        )
        # Side cell: for sport bots, mirror the watchlist row underneath
        # (team tricode on top, "vs opponent" beneath). The team we're
        # actually rooting for sits on top — on a NO bet that's the
        # team the YES side is *against* — so the user reads the bet
        # the same way Kalshi's market page reads it. The badge color
        # (green YES / red NO) is preserved as a left-edge accent so
        # the bet direction stays visible at a glance. Non-sport bots
        # keep the legacy YES/NO badge — the watchlist's third column
        # is a different field (Question) over there.
        if is_sport_bot:
            # Sport rows prefer pre-supplied labels (tennis carries
            # _yes_label / _no_label = player names); fall back to the
            # NBA tricode parser for KXNBAGAME tickers.
            yes_label = b.get("_yes_label") or _side_tricode_from_ticker(
                b.get("ticker"), "YES")
            no_label = b.get("_no_label") or _side_tricode_from_ticker(
                b.get("ticker"), "NO")
            if side == "YES":
                side_team, opp_team = yes_label, no_label
            else:
                side_team, opp_team = no_label, yes_label
            if side_team:
                # No badge_cls colour on the player name — Side reads
                # as identity, not direction.
                side_cell = (
                    f"<td class='active-side-team'>"
                    f"<strong>{html.escape(str(side_team))}</strong>"
                    f"<br><span class='small gray'>vs "
                    f"{html.escape(str(opp_team))}</span></td>"
                )
            else:
                side_cell = (
                    f"<td><span class='badge {badge_cls}'>{side}</span></td>"
                )
        else:
            side_cell = (
                f"<td><span class='badge {badge_cls}'>{side}</span></td>"
            )
        # In-game model pill — only renders when the live model has a
        # confident view. Sits inline with the Side cell so it reads
        # as additional context on what we're holding.
        in_game = b.get("_in_game") or {}
        ig_action = (in_game.get("action") or "").lower()
        ig_pill = ""
        if in_game and ig_action in {"exit_now", "let_run", "hold"}:
            cls_map = {
                "exit_now": "ig-red",
                "let_run": "ig-green",
                "hold": "ig-yellow",
            }
            label_map = {
                "exit_now": "EXIT",
                "let_run": "RUN",
                "hold": "HOLD",
            }
            ig_pill = (
                f"<span class='in-game-pill {cls_map[ig_action]}' "
                f"title='{html.escape(in_game.get('reason') or '')}'>"
                f"{label_map[ig_action]}</span>"
            )
        # Inject the pill into the side cell so the table doesn't
        # gain a column.
        if ig_pill:
            side_cell = side_cell.replace("</td>", f" {ig_pill}</td>", 1)

        if sport_style:
            date_label = _market_date_label(b.get("ticker"),
                                            b.get("rules_primary"))
            event_label = _sport_event_label(
                b.get("rules_primary"), title_text,
                b.get("_tournament"))
            # Side cell mirrors the sport pages: the team/player we
            # hold on top, opponent beneath.
            sp = b.get("_side_player") or ""
            if sp:
                match_txt = b.get("_match") or ""
                opp = ""
                if match_txt and sp in match_txt:
                    opp = match_txt.replace(sp, "").replace(
                        " vs ", " ").strip()
                side_cell_s = (
                    f"<td class='active-side-team'>"
                    f"<strong>{html.escape(sp)}</strong>"
                    + (f"<br><span class='small gray'>vs "
                       f"{html.escape(opp)}</span>" if opp else "")
                    + "</td>")
            else:
                side_cell_s = side_cell
            # 2026-07-15 layout mirrors the per-bot Active bets table:
            # Date | [Bot] | Event | Title | Side
            #   | My contracts | Cost | Payout
            #   | Model entry % | Kalshi entry % | Model live % | Kalshi live %
            #   | Closes in | why-button
            # Edge / EV dropped — the entry-vs-live comparison across
            # Model + Kalshi shows the same information in a more
            # actionable form.
            total_cost = entry_cost_base + entry_fee_dollars
            _payout_cls = "num green" if potential_gain >= 0 else "num red"
            out.append(
                f"<tr{tr_attrs}>"
                f"<td>{html.escape(date_label or '—')}</td>"
                f"{bot_td}"
                f"<td>{html.escape(event_label or '—')}</td>"
                f"<td>{html.escape(title_text)}</td>"
                f"{side_cell_s}"
                f"<td class='num'>{contracts}</td>"
                f"<td class='num red' title='Kalshi total cost — "
                f"{entry}¢ × {contracts} contracts + "
                f"${entry_fee_dollars:.2f} entry fee = "
                f"${total_cost:.2f} total cash out at open.'>"
                f"−${total_cost:.2f}</td>"
                f"<td class='{_payout_cls}' title='Potential earnings "
                f"if this side wins — $1 × {contracts} contracts − "
                f"${total_cost:.2f} paid = net gain.'>"
                f"{pg_sign}${abs(potential_gain):.2f}</td>"
                f"{model_entry_cell}"
                f"{kalshi_entry_cell}"
                f"{model_prob_cell}"
                f"{current_prob_cell}"
                f"<td class='num'>{time_to_close_str(mtc)}</td>"
                f"<td><button type='button' class='criteria-btn' "
                f"title='Why was this bet chosen?' "
                f"data-criteria='{criteria_json}'>i</button></td>"
                f"</tr>"
            )
            continue
        out.append(
            f"<tr{tr_attrs}><td>{html.escape(opened)}</td>"
            f"{bot_td}"
            f"<td class='mono'>{ticker_cell_html(b.get('ticker'))}</td>"
            f"<td>{html.escape(title_text)}</td>"
            f"{side_cell}"
            f"<td class='num'>{contracts}</td>"
            f"{model_prob_cell}"
            f"{entry_prob_cell}"
            f"{current_prob_cell}"
            f"{entry_cost_cell}"
            f"<td class='num {pg_cls}' title='"
            f"(100¢ − {entry}¢) × {contracts} contracts − ${entry_fee_dollars:.2f} fee = "
            f"${(100 - entry) * contracts / 100.0:.2f} − ${entry_fee_dollars:.2f} = "
            f"${potential_gain:.2f}. Entry fee already paid; settlement "
            f"at 100¢ or 0¢ has zero exit fee.'>"
            f"{pg_sign}${abs(potential_gain):.2f}</td>"
            f"<td class='num'>{time_to_close_str(mtc)}</td>"
            f"<td><button type='button' class='criteria-btn' "
            f"title='Why was this bet chosen?' "
            f"data-criteria='{criteria_json}'>i</button></td>"
            f"</tr>"
        )
    out.append("</tbody></table>")


def _parse_season_dt(value: str | None) -> datetime | None:
    """Best-effort ISO-8601 → aware datetime. Accepts the ``Z`` suffix
    that PyYAML / config files commonly use; returns None on parse
    failure so a broken season block just hides the affected card
    rather than 500-ing the page."""
    if not value:
        return None
    s = str(value).strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _humanize_countdown(delta_seconds: float) -> str:
    """Server-side initial value for the countdown cells, in the same
    ``Xd Xh Xm Xs`` shape the JS tick() function renders. Floors to 0
    on negatives so the placeholder never reads as a negative duration
    if the JS hasn't run yet."""
    s = max(0, int(delta_seconds))
    days, rem = divmod(s, 86400)
    hours, rem = divmod(rem, 3600)
    mins, secs = divmod(rem, 60)
    return f"{days}d {hours}h {mins}m {secs}s"


def _humanize_duration(delta_seconds: float) -> str:
    """Render a duration as ``Xd`` or ``Xw Yd`` for season-length cells.
    Used as the static "Length" value on the Seasons tab; the live
    countdown values are rendered client-side in JS."""
    if delta_seconds <= 0:
        return "—"
    days = int(delta_seconds // 86400)
    if days < 14:
        return f"{days}d"
    weeks, rem_days = divmod(days, 7)
    if rem_days == 0:
        return f"{weeks}w"
    return f"{weeks}w {rem_days}d"


def _render_seasons_panel(out: List[str], available_bots: List[dict]) -> None:
    """One card per league. A bot can declare multiple ``seasons:``
    entries when its bot trades multiple Kalshi competitions
    (tennis = ATP + WTA tours, darts = Premier League + PDC World
    Championship, etc.) — each entry renders as its own card.
    Cards whose end time has already passed are hidden so the tab
    stays focused on what's actually trading; the countdown only
    flips between "Starts in …" and "Ends in …". Live leagues sort
    above upcoming ones."""
    now = datetime.now(timezone.utc)
    one_year_out = now + timedelta(days=365)

    cards: List[tuple[dict, dict, datetime, datetime]] = []
    for bot in available_bots:
        for season in (bot.get("seasons") or []):
            start = _parse_season_dt(season.get("start"))
            end = _parse_season_dt(season.get("end"))
            if not start or not end:
                continue
            if end < now:
                # Season already wrapped up; update the YAML to the
                # next iteration to bring this league back to the tab.
                continue
            if start > one_year_out:
                continue
            cards.append((bot, season, start, end))

    # Live leagues first (soonest end), then upcoming (soonest start).
    def _sort_key(item):
        _, _, s, e = item
        return (0, e) if s <= now <= e else (1, s)
    cards.sort(key=_sort_key)

    out.append(
        "<div class='section'><h2>Seasons</h2><div class='body'>"
    )
    if not cards:
        out.append(
            "<div class='empty'>No leagues have an active or "
            "upcoming season configured. Update a "
            "<code>seasons:</code> block in "
            "<code>config/dashboard.yaml</code> to bring a card "
            "back.</div>"
        )
        out.append("</div></div>")
        return

    out.append(
        "<p class='small gray' style='margin: 0 0 14px 0;'>"
        "One card per league. Live leagues "
        "(<span class='green'>Ends in</span>) sit above upcoming "
        "ones (<span class='yellow'>Starts in</span>)."
        "</p>"
    )
    out.append("<div class='season-grid'>")
    for bot, season, start, end in cards:
        bot_label = bot.get("name") or bot.get("key")
        season_name = season.get("name") or bot_label
        length = _humanize_duration((end - start).total_seconds())
        start_ms = int(start.timestamp() * 1000)
        end_ms = int(end.timestamp() * 1000)
        start_str = start.strftime("%b %-d, %Y")
        end_str = end.strftime("%b %-d, %Y")
        if now < start:
            init_status, init_color = "Upcoming", "yellow"
            init_label, init_value = "Starts in", _humanize_countdown(
                (start - now).total_seconds())
        else:
            init_status, init_color = "In season", "green"
            init_label, init_value = "Ends in", _humanize_countdown(
                (end - now).total_seconds())
        total = (end - start).total_seconds()
        if now <= start or total <= 0:
            init_pct = 0.0
        else:
            init_pct = max(0.0, min(100.0,
                ((now - start).total_seconds() / total) * 100.0))
        bot_href = f"?bot={html.escape(bot['key'])}&tab=watchlist"
        out.append(
            f"<div class='season-card' data-season-card "
            f"data-start='{start_ms}' data-end='{end_ms}'>"
            f"<div class='season-card-head'>"
            f"<a class='season-bot' href='{bot_href}'>"
            f"{html.escape(bot_label)}</a>"
            f"<span class='status-pill {init_color}' "
            f"data-season-status>{html.escape(init_status)}</span>"
            f"</div>"
            f"<div class='season-name'>{html.escape(season_name)}</div>"
            f"<div class='season-countdown'>"
            f"<div class='season-countdown-label' "
            f"data-season-countdown-label>"
            f"{html.escape(init_label)}</div>"
            f"<div class='season-countdown-value {init_color}' "
            f"data-season-countdown-value>"
            f"{html.escape(init_value)}</div>"
            f"</div>"
            f"<div class='season-progress'>"
            f"<div class='season-progress-fill' "
            f"data-season-progress-fill "
            f"style='width: {init_pct:.2f}%;'></div>"
            f"</div>"
            f"<div class='season-meta'>"
            f"<div><span class='season-meta-label'>Start</span>"
            f"<span class='season-meta-value'>"
            f"{html.escape(start_str)}</span></div>"
            f"<div><span class='season-meta-label'>End</span>"
            f"<span class='season-meta-value'>"
            f"{html.escape(end_str)}</span></div>"
            f"<div><span class='season-meta-label'>Length</span>"
            f"<span class='season-meta-value'>"
            f"{html.escape(length)}</span></div>"
            f"</div>"
            f"</div>"
        )
    out.append("</div>")  # /season-grid
    out.append("</div></div>")  # /body /section


def _render_history_chart(out: List[str], history: List[dict],
                            period_key: str = "all",
                            current_bot: str = "") -> None:
    """Daily net P&L line chart for the History tab. The closed-bet
    ledger is embedded as JSON on the SVG node; the JS buckets bets
    by UTC day and draws the line client-side. The period dropdown
    sits inline above the chart as its toolbar.
    """
    points: List[list] = []
    for h in history:
        ts_str = h.get("exited_at")
        pnl = h.get("realized_pnl_cents")
        if not ts_str or pnl is None:
            continue
        # Same idiom as the bet-history rows: chop to 19 chars to drop
        # fractional seconds + tz suffix, then parse as UTC. Timestamps
        # in sim.db are recorded in UTC.
        try:
            dt = datetime.fromisoformat(ts_str[:19])
            epoch = int(dt.replace(tzinfo=timezone.utc).timestamp())
        except (TypeError, ValueError):
            continue
        try:
            pnl_int = int(pnl)
        except (TypeError, ValueError):
            continue
        points.append([epoch, pnl_int])
    points_payload = html.escape(
        json.dumps(points, separators=(",", ":")),
        quote=True,
    )
    out.append("<div class='history-chart-section'>")
    # Toolbar row: chart title on the left, period selector on the
    # right — visually anchors the filter to the chart it controls.
    out.append("<div class='history-chart-toolbar'>")
    out.append("<div class='history-chart-title'>Daily net P&amp;L</div>")
    out.append("</div>")
    out.append(
        "<div class='history-chart-wrap'>"
        f"<svg data-history-chart data-points='{points_payload}' "
        "width='100%' height='260' viewBox='0 0 800 260' "
        "preserveAspectRatio='none' style='display:block'></svg>"
        "</div>"
    )
    out.append("</div>")  # /history-chart-section


def _render_history_attribution(out: List[str],
                                  history: List[dict]) -> None:
    """P&L attribution panels for the History tab — small breakdown
    tables that slice the closed-bet ledger four ways: by bot, by
    month, by side (YES/NO), and by predicted-EV bucket. Each panel
    tries to answer "where is the P&L coming from?" so the user can
    spot whether profit is broad (likely real edge) or concentrated
    in one slice (likely a quirk).

    Respects whatever period filter the History tab is currently on
    — ``history`` is already the period-scoped list the caller passes
    into the chart and ledger renderers below.
    """
    if not history:
        return  # Empty state already covered by the ledger block.

    def _row(label: str, bets: List[dict]) -> dict:
        n = len(bets)
        total = sum((b.get("realized_pnl_cents") or 0) for b in bets)
        wins = sum(1 for b in bets if (b.get("realized_pnl_cents") or 0) > 0)
        return {"label": label, "n": n, "total_cents": total,
                "win_pct": (wins / n) if n else 0.0}

    def _emit_table(title: str, hint: str, rows: List[dict]) -> None:
        hint_html = (f" <span class='small gray'>{html.escape(hint)}</span>"
                     if hint else "")
        out.append(
            f"<div class='attribution-panel'>"
            f"<h3 class='subhead'>{html.escape(title)}{hint_html}</h3>"
        )
        if not rows:
            out.append("<div class='empty'>No data in this slice.</div>"
                       "</div>")
            return
        out.append(
            "<table><thead><tr>"
            "<th>Bucket</th>"
            "<th class='num'>Bets</th>"
            "<th class='num'>P&amp;L</th>"
            "<th class='num'>Win %</th>"
            "</tr></thead><tbody>"
        )
        for r in rows:
            pnl_cls = ("green" if r["total_cents"] > 0
                        else ("red" if r["total_cents"] < 0 else "gray"))
            win = r["win_pct"]
            win_cls = ("green" if win > 0.5
                        else ("red" if r["n"] > 0 and win < 0.5 else "gray"))
            win_str = f"{win*100:.0f}%" if r["n"] > 0 else "—"
            dollars = r["total_cents"] / 100.0
            sign = "+" if r["total_cents"] > 0 else (
                "−" if r["total_cents"] < 0 else "")
            out.append(
                f"<tr><td>{html.escape(r['label'])}</td>"
                f"<td class='num'>{r['n']}</td>"
                f"<td class='num {pnl_cls}'>{sign}${abs(dollars):.2f}</td>"
                f"<td class='num {win_cls}'>{win_str}</td></tr>"
            )
        out.append("</tbody></table></div>")

    # ── Slice: by bot ───────────────────────────────────────────────
    by_bot: dict[str, List[dict]] = {}
    for h in history:
        by_bot.setdefault(h.get("_bot_name") or "—", []).append(h)
    bot_rows = sorted(
        (_row(name, bets) for name, bets in by_bot.items()),
        key=lambda r: r["total_cents"], reverse=True,
    )

    # ── Slice: by month (YYYY-MM) ───────────────────────────────────
    by_month: dict[str, List[dict]] = {}
    for h in history:
        ts = (h.get("exited_at") or "")[:7]  # YYYY-MM
        if ts:
            by_month.setdefault(ts, []).append(h)
    month_rows = [_row(m, bets) for m, bets in
                  sorted(by_month.items(), reverse=True)]

    # ── Slice: by entry-price bucket ────────────────────────────────
    # Replaces the earlier YES/NO breakdown (which was ~always YES on
    # the sport bots, so it collapsed to a single row and told the user
    # nothing). Bucketing by the actual paid price surfaces where the
    # P&L pattern lives — deep-underdog (15-25¢) buys are the tail the
    # miscalibration reviews keep flagging, so isolating that bucket
    # from the moderate-favourite (50-65¢) bucket is the useful cut.
    # 7 buckets aligned to the bot's 30-70¢ operating range (global
    # gates block buys below ~30¢ or above ~70¢). Two open-ended tails
    # flag any trade that slipped outside the band — a nonzero count
    # there is worth investigating on its own — and five equal 8¢
    # slices carve the 30-70¢ zone finely enough to see where the P&L
    # pattern lives. Untagged catches rows without a recorded entry
    # price and stays empty-hidden.
    # Bucket edges per user 2026-07-10: <35 / 35–39.99 / 40–44.99 /
    # 45–49.99 / 50–54.99 / 55–59.99 / 60–64.99 / ≥65. Entry prices
    # are integer cents, so each bucket is [lo, hi).
    price_buckets = [
        ("< 35¢",     None, 35),
        ("35–39¢",    35,   40),
        ("40–44¢",    40,   45),
        ("45–49¢",    45,   50),
        ("50–54¢",    50,   55),
        ("55–59¢",    55,   60),
        ("60–64¢",    60,   65),
        ("≥ 65¢",     65,   None),
        ("untagged",  None, None),
    ]
    price_rows: List[dict] = []
    for label, lo, hi in price_buckets:
        bucket_bets: List[dict] = []
        for h in history:
            price = h.get("entry_price_cents")
            if lo is None and hi is None:
                if price is None:
                    bucket_bets.append(h)
                continue
            if price is None:
                continue
            try:
                p = int(price)
            except (TypeError, ValueError):
                continue
            if lo is not None and p < lo:
                continue
            if hi is not None and p >= hi:
                continue
            bucket_bets.append(h)
        if bucket_bets:
            price_rows.append(_row(label, bucket_bets))

    # ── Slice: by entry edge (model − market, percentage points) ─────
    # Replaced the predicted-EV buckets (user 2026-07-10): the edge is
    # the number the buy decision actually keys on — for the benchmark
    # bots it's "Pinnacle fair prob − price paid" — so bucketing on it
    # answers "do bigger edges actually realize more P&L?" directly.
    # Side-adjusted: a NO bet's edge is (1 − model_yes) − price paid.
    def _entry_edge_pp(h: dict) -> float | None:
        model_yes = h.get("model_yes_prob_at_entry")
        price = h.get("entry_price_cents")
        if model_yes is None or price is None:
            return None
        try:
            m = float(model_yes)
            p = float(price) / 100.0
        except (TypeError, ValueError):
            return None
        side = (h.get("side") or "YES").upper()
        model_side = m if side == "YES" else (1.0 - m)
        return (model_side - p) * 100.0

    # No "untagged" bucket (user 2026-07-10): settlements are joined
    # back to the bot ledgers for their model prob (both the paper and
    # executor sim_states are indexed — see _load_sim_state_enrichment),
    # and the rare row that still has no recorded model prob (manual
    # trades from before any bot tracked the series) is simply
    # excluded from this panel; it still counts in the other three.
    edge_buckets = [
        ("< 0pp",    -100.0, 0.0),
        ("0–3pp",     0.0,   3.0),
        ("3–5pp",     3.0,   5.0),
        ("5–7pp",     5.0,   7.0),
        ("7–10pp",    7.0,   10.0),
        ("10–15pp",   10.0,  15.0),
        ("15pp+",     15.0,  100.0),
    ]
    edge_rows: List[dict] = []
    for label, lo, hi in edge_buckets:
        bucket_bets: List[dict] = []
        for h in history:
            edge_pp = _entry_edge_pp(h)
            if edge_pp is None:
                continue
            if edge_pp < lo or edge_pp >= hi:
                continue
            bucket_bets.append(h)
        if bucket_bets:
            edge_rows.append(_row(label, bucket_bets))

    out.append(
        "<h3 class='subhead' style='margin-top:14px;'>"
        "P&amp;L attribution</h3>"
    )
    out.append("<div class='attribution-grid'>")
    # Short titles per user 2026-07-10 — the hint text moved into
    # each table's Bucket-column semantics and is self-evident.
    _emit_table("By bot", "", bot_rows)
    _emit_table("By month", "", month_rows)
    _emit_table("By price", "", price_rows)
    _emit_table("By edge", "", edge_rows)
    out.append("</div>")


def _render_bet_history_block(out: List[str], history: List[dict],
                               heading: str = "Historical bets — closed",
                               shown_initially: int = 5) -> None:
    """Subsection: closed bets with entry/exit/P&L. Used inline under
    Section 1 (Summary) and Section 5 (Active bet) so each view shows
    the lifetime trade ledger directly under its active-bets table.

    Uses HTML <details>/<summary> so the first ``shown_initially`` rows
    are visible and the rest are collapsible — no JS. Pass an empty
    ``heading`` to render the table without a subhead — useful when the
    enclosing section's title already names the period.
    """
    if heading:
        out.append(f"<h3 class='subhead'>{html.escape(heading)}</h3>")
    if not history:
        out.append("<div class='empty'>No closed bets yet.</div>")
        return

    head = (
        "<table><thead><tr>"
        "<th title='Date the contract was opened (UTC).'>Opened</th>"
        "<th title='Date the contract was closed (UTC).'>Closed</th>"
        "<th>Bot</th>"
        "<th>Title</th>"
        "<th title='Who the bot bet on — the side this position pays out for.'>Projected winner</th>"
        "<th title='Player / team that actually won the match. Derived from settlement outcome + which side we bet on.'>Winner</th>"
        "<th>Side</th>"
        "<th class='num' title='Model % for the side we bet on at the moment the contract was bought — static once opened, same definition as the Active bets Model entry % column.'>Model entry %</th>"
        "<th class='num' title='Implied probability at the price we "
        "paid — static once opened, same definition as the Active "
        "bets Kalshi entry % column.'>Kalshi entry %</th>"
        "<th class='num'>Contracts</th>"
        "<th class='num' title='Net EV per contract at entry: (model_p − entry_price) − half-spread. "
        "Positive = +EV trade.'>Entry EV</th>"
        "<th class='num'>P&amp;L</th>"
        "<th>Outcome</th>"
        "</tr></thead><tbody>"
    )

    def render_row(b):
        # Both timestamps to ISO YYYY-MM-DD HH:MM:SS (UTC). Slicing at
        # 19 chops the fractional seconds + tz offset for compactness.
        opened = (b.get("opened_at") or "")[:19].replace("T", " ")
        closed = (b.get("exited_at") or "")[:19].replace("T", " ")
        side = (b.get("side") or "").upper()
        badge_cls = "badge-yes" if side == "YES" else "badge-no"
        entry = b.get("entry_price_cents")
        exit_c = b.get("exit_price_cents")
        contracts = b.get("contracts", 0) or 0
        pnl = b.get("realized_pnl_cents") or 0
        pnl_cls_ = "green" if pnl > 0 else ("red" if pnl < 0 else "gray")
        outcome = "WON" if pnl > 0 else ("LOST" if pnl < 0 else "FLAT")
        # Value at close — uses the bot's display config so jobless
        # renders "189K" (no decimals) and gas/natgas render "$2.79".
        display = b.get("_display") or {}
        value_at_close = b.get("gas_price_at_close")
        value_str = (fmt_underlying(value_at_close, display)
                     if value_at_close is not None else "—")
        bot_name = b.get("_bot_name", "—")
        # Question — rendered in the bot's native units when display
        # config is attached. Strikes pulled via market_views subquery
        # in fetch_bet_history.
        floor = b.get("floor_strike")
        cap = b.get("cap_strike")
        try:
            strike_low = float(floor) if floor is not None else None
        except (TypeError, ValueError):
            strike_low = None
        try:
            strike_high = float(cap) if cap is not None else None
        except (TypeError, ValueError):
            strike_high = None
        direction = ("between" if (strike_low is not None
                                    and strike_high is not None)
                     else "above")
        question = question_str(direction, strike_low, strike_high,
                                  display=display)
        # Selected-side model prob: YES bet = model_yes_prob; NO bet = 1 - that.
        m_yes = b.get("model_yes_prob_at_entry")
        if m_yes is not None:
            p_sel = float(m_yes) if side == "YES" else (1.0 - float(m_yes))
            mp_str = f"{p_sel*100:.0f}%"
        else:
            mp_str = "—"
        # Kalshi entry % (implied prob at fill) + Entry cost (price x
        # contracts, dollars) — user 2026-07-20 column spec.
        kalshi_entry_pct_str = (f"{int(entry)}%" if entry is not None
                                 else "—")
        # Country flags (user 2026-07-21) — resolved from the ticker's
        # league or the tennis nationality map; '' when unknown.
        from .flags import flag_for as _flag_for
        from .flags import flag_matchup as _flag_matchup
        _tk = b.get("ticker") or ""
        ev = b.get("expected_ev_at_entry")
        if ev is None or round(float(ev), 2) == 0:
            ev_str = "0"
            ev_cls = "gray"
        else:
            ev_sign = "+" if ev > 0 else "−"
            ev_str = f"{ev_sign}${abs(ev):.2f}"
            ev_cls = _ev_status(ev)[0]
        # Title cell: Kalshi-published contract title; falls back to a
        # derived "matchup — bet on X" or the strike question text
        # when no Kalshi title is recorded on this row.
        title_text = b.get("_title") or b.get("title") or ""
        if not title_text:
            match_text = b.get("_match") or _match_text_from_ticker(b.get("ticker"))
            side_player = b.get("_side_player")
            if side_player:
                title_text = (f"{match_text} — bet on {side_player}"
                               if match_text else side_player)
            elif match_text:
                tri = _side_tricode_from_ticker(b.get("ticker"), side)
                title_text = (f"{match_text} — bet on {tri}"
                               if tri else match_text)
            else:
                title_text = question
        # Bot cell: link to the bot's watchlist (same idiom as the
        # active-bets table).
        bot_key = b.get("_bot_key") or ""
        bot_dt = b.get("_dashboard_type") or "standard"
        if bot_key:
            if bot_dt == "sport":
                bot_href = f"?bot={html.escape(bot_key)}&tab=watchlist"
            else:
                bot_href = f"?tab=watchlist&bot={html.escape(bot_key)}"
            bot_cell = (f"<td><a href='{bot_href}' class='bot-link'>"
                         f"{html.escape(bot_name)}</a></td>")
        else:
            bot_cell = f"<td>{html.escape(bot_name)}</td>"
        # ``merged_trade_count > 1`` flags a history row that collapses
        # multiple flap-trades (open + close + re-open on the same
        # match/strike) into one. Surface a small "×N" badge next to
        # the Title so the user can tell a merged row from a single
        # trade — and the P&L column makes sense as the *net* across
        # those N trades. Moved from the Ticker cell 2026-07-11 when
        # that column was retired.
        merged_n = int(b.get("merged_trade_count") or 1)
        if merged_n > 1:
            merged_badge = (
                f"<span class='merged-badge' "
                f"title='Net P&L across {merged_n} trades on this same "
                f"ticker (bot re-opened the position after each close).'"
                f">×{merged_n}</span>"
            )
        else:
            merged_badge = ""
        # Winner cell — derive who won the match from the settlement
        # outcome + which side we bet on. Sport rows carry _match
        # ("Player A vs Player B") and _side_player (whichever one we
        # took); a positive P&L means our side won, a negative one
        # means the opponent won. Non-sport rows (gas / claims / cpi)
        # don't have a "match winner" concept — they render "—".
        _match_text = b.get("_match") or ""
        _side_player = b.get("_side_player") or ""
        # Matchup titles render on TWO lines — "X" over "vs Y" — so
        # long name pairs stop stretching the Title column (user
        # 2026-07-21). Non-matchup titles pass through unchanged.
        if " vs " in title_text:
            # Each line STARTS with its flag (user 2026-07-21); the
            # "vs" hangs off the first line in muted small type so the
            # stacked names stay the visual anchor.
            _ta, _, _tb = title_text.partition(" vs ")
            # Drop the legacy "— bet on X" suffix: the Projected
            # winner column says it now.
            _tb = _tb.split(" — bet on ")[0].strip()
            # Two stacked lines, each STARTING with its flag; "vs"
            # rides muted at the end of the TOP line (user
            # 2026-07-21). ``_match_text`` carries the FULL names, so
            # last-name title lines still resolve nationalities.
            _title_cell = (
                f"<span style='white-space:nowrap'>"
                f"{_flag_for(_ta, _tk, _match_text)}{html.escape(_ta)}"
                f" <span class='small gray'>vs</span></span><br>"
                f"<span style='white-space:nowrap'>"
                f"{_flag_for(_tb, _tk, _match_text)}"
                f"{html.escape(_tb)}</span>")
        else:
            _title_cell = _flag_matchup(html.escape(title_text), _tk)
        winner_str = "—"
        if _side_player and _match_text and pnl != 0:
            if pnl > 0:
                winner_str = _side_player
            else:
                opp = (_match_text
                       .replace(_side_player, "")
                       .replace(" vs ", " ")
                       .strip(" -"))
                winner_str = opp or "—"
        return (f"<tr><td>{html.escape(opened)}</td>"
                f"<td>{html.escape(closed)}</td>"
                f"{bot_cell}"
                f"<td>{_title_cell}{merged_badge}</td>"
                f"<td>{_flag_for(_side_player, _tk, _match_text)}"
                f"{html.escape(_side_player or chr(8212))}</td>"
                f"<td>{_flag_for(winner_str, _tk, _match_text)}"
                f"{html.escape(winner_str)}</td>"
                f"<td><span class='badge {badge_cls}'>{side}</span></td>"
                f"<td class='num'>{mp_str}</td>"
                f"<td class='num'>{kalshi_entry_pct_str}</td>"
                f"<td class='num'>{contracts}</td>"
                f"<td class='num {ev_cls}'>{ev_str}</td>"
                f"<td class='num {pnl_cls_}'>{fmt_signed_cents(pnl)}</td>"
                f"<td class='{pnl_cls_}'>{outcome}</td></tr>")

    # All rows go into a single table — the History tab's
    # max-height: 640px scroll container handles overflow. The old
    # ``Show N more`` collapsible details was confusing when the
    # scroll already implies "everything's in here".
    # ``shown_initially`` is retained on the function signature for
    # back-compat with callers (Section 5 / Home use a smaller
    # window) but the History tab passes the full list through.
    out.append(head)
    for b in history:
        out.append(render_row(b))
    out.append("</tbody></table>")


# Bot-filter whitelist (user 2026-07-13): only surface bots that are
# actively trading against a sharp benchmark. Macro / commentary bots
# (gas / claims / cpi / natural-gas / survivor) stay out of the
# dropdown even when their sim.db exists. Strict list — a URL
# that lands on an off-whitelist bot renders the bot's page but the
# picker only shows these. Baseball added same-day per user ("include
# the baseball forecast in the contracts filter too") — it trades
# real money via the armed live executor, same as NBA / WNBA.
# Billboard added 2026-07-15 per user after the KXTOPSONG retarget —
# paper-only, but its watchlist / model / training pages are live.
BOT_FILTER_KEYS = {
    "mlb", "nba", "wnba", "tennis", "table-tennis", "darts",
    "world-cup", "billboard", "reality-leaks", "hormuz",
}


def default_filter_bot(bots: List[dict]) -> dict | None:
    """The topmost bot in the filter dropdown — the first entry of the
    registry (display order) whose key is whitelisted. This is what a
    bot-less URL resolves to, so opening Contracts → Watchlist lands
    on the same bot the dropdown shows first (user 2026-07-13).
    Returns None when no registered bot is whitelisted."""
    for b in bots:
        if b.get("key") in BOT_FILTER_KEYS:
            return b
    return None


def _render_bot_filter(out: List[str], available_bots: List[dict],
                       current_bot: str,
                       period_key: str = "all",
                       select_id: str = "bot-select",
                       include_all_option: bool = False,
                       tab_key: str = "watchlist") -> None:
    """Bot selector dropdown — used on both the Home tab (as the
    "jump to a bot's watchlist" navigator) and on the per-bot
    Watchlist tab. Native <select> for keyboard-friendliness.

    Each instance gets its own ``select_id`` so multiple dropdowns
    coexist on the same DOM (the page's tab panels all live in one
    document). The page's onchange JS finds them via
    ``[data-bot-select]`` so it doesn't need to know the id.

    ``include_all_option`` adds a leading "All bots" entry that lands
    on ``/`` — used on Home so the dropdown can return the user to
    the cross-bot summary view after browsing into a watchlist.
    """
    period_qs = (f"&period={html.escape(period_key)}"
                 if period_key and period_key != "all" else "")
    _bots_for_dropdown = [b for b in available_bots
                          if b.get("key") in BOT_FILTER_KEYS]
    out.append("<div class='bot-filter-bar'>")
    out.append(f"<label for='{html.escape(select_id)}' "
               f"class='filter-label'>Bot</label>")
    out.append(
        f"<select id='{html.escape(select_id)}' class='bot-select' "
        f"data-bot-select>"
    )
    if include_all_option:
        # "All bots" returns to the cross-bot home page.
        all_url = f"/?period={html.escape(period_key)}" if period_key and period_key != "all" else "/"
        sel = " selected" if not current_bot else ""
        out.append(
            f"<option value='{html.escape(all_url)}'{sel}>All bots</option>"
        )
    for b in _bots_for_dropdown:
        avail = b.get("available", True)
        suffix = "" if avail else " (no data)"
        sel = " selected" if b["key"] == current_bot else ""
        # Tennis routes through its own renderer; both branches build
        # the same URL shape so the dropdown jumps the user into the
        # right per-bot tab regardless of dashboard_type.
        href = (f"?bot={html.escape(b['key'])}"
                f"&tab={html.escape(tab_key)}{period_qs}")
        out.append(
            f"<option value='{html.escape(href)}'{sel}>"
            f"{html.escape(b['name'])}{html.escape(suffix)}</option>"
        )
    out.append("</select>")
    # Diagnosis button — opens the shared daily-droplet-diagnosis modal.
    # Status dot is set by the JS once /api/diagnosis/latest resolves
    # (healthy / has_issues / stale). Render uses ``data-diagnosis-trigger``
    # rather than an id so the multiple bot-filter-bar instances on a
    # page (home + tab views) can each carry a clickable trigger.
    out.append(
        "<button type='button' class='diagnosis-btn' "
        "data-diagnosis-trigger "
        "title='Daily droplet diagnosis report'>"
        "<span class='diagnosis-dot' data-diagnosis-dot></span>"
        "Diagnosis</button>"
    )
    out.append("</div>")


def _render_bot_unavailable(out: List[str], bot_key: str) -> None:
    out.append("<div class='section'><h2>Bot data unavailable</h2><div class='body'>"
               f"<div class='empty'>The <b>{html.escape(bot_key)}</b> bot is registered "
               f"but has no data on this host yet. Switch to a different bot above, "
               f"or run that bot's service to populate <code>data/sim.db</code>.</div>"
               "</div></div>")
