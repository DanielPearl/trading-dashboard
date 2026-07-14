"""Watchlist tab — hero chart, model-vs-market and active tables."""
from __future__ import annotations

import html
import json
from datetime import datetime
from datetime import timezone
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from .fmt import (
    _ev_status,
    _market_date_label,
    _side_tricode_from_ticker,
    _sport_event_label,
    fmt_underlying,
    kalshi_fee_cents,
    minutes_to_close_from_ticker,
    question_str,
    svg_kalshi_chart,
    ticker_link_html,
    time_left_str,
    time_to_close_str,
)
from .panels import _render_active_bets_table

import logging
log = logging.getLogger("dashboard")


def _fmt_signed_underlying(value: float | None, display: dict) -> str:
    """Like fmt_underlying but with a leading +/- sign — for delta
    values (median_change). Respects the bot's divisor + decimals."""
    if value is None:
        return "—"
    divisor = float(display.get("divisor", 1.0)) or 1.0
    v = float(value) / divisor
    decimals = int(display.get("underlying_decimals", 2))
    unit = display.get("underlying_unit", "")
    pos = display.get("unit_position", "prefix")
    sign = "+" if v >= 0 else "−"
    n = f"{abs(v):,.{decimals}f}"
    if pos == "prefix":
        return f"{sign}{unit}{n}"
    if pos == "suffix":
        return f"{sign}{n}{unit}"
    return f"{sign}{n}"


def _render_current_prediction(out: List[str], model: dict | None,
                                 display: dict | None = None,
                                 contract_is_closed: bool = False) -> None:
    """Renders the 'Current prediction' card row.

    Lives at the top of the Watchlist section now (per user request) — it
    fits there better than under the Model section because it's the
    immediate context for reading the watchlist rows below it.

    Number formatting follows the bot's display config so unemployment
    shows "189K" / "+1K" instead of "$189000.00" / "+1000.00".

    When `contract_is_closed` is True (the current event has already
    settled and there's no live market to forecast for), every value
    cell is dashed out — a stale model snapshot from the previous
    contract isn't a forecast for the next one.
    """
    if not model:
        return
    display = display or {}
    # No subsec wrapper — match the Summary's plain `<div class='row'>`
    # structure so the bottom-of-cards → top-of-h3 spacing collapses
    # naturally and the Watchlist "Active bet" h3 sits at the same
    # offset Summary's "Active bets" h3 does.
    prob_up = float(model.get("prob_up") or 0)
    change = float(model.get("median_change") or 0)
    q05 = model.get("quantile_05")
    q95 = model.get("quantile_95")
    if contract_is_closed:
        cur_str = pred_str = chg_str = prob_str = q05_str = q95_str = "—"
    else:
        cur_str = html.escape(
            fmt_underlying(model.get("current_gas_price"), display))
        pred_str = html.escape(
            fmt_underlying(model.get("median_price"), display))
        chg_str = html.escape(_fmt_signed_underlying(change, display))
        prob_str = f"{prob_up:.0%}"
        q05_str = (html.escape(fmt_underlying(q05, display))
                   if q05 is not None else "—")
        q95_str = (html.escape(fmt_underlying(q95, display))
                   if q95 is not None else "—")
    # Cadence-aware labels driven by the bot's display config:
    #   • underlying_label gives the "Current X" card a bot-specific name
    #     ("Retail gas price" vs "Initial jobless claims" vs "Last
    #     realized Core CPI MoM").
    #   • prediction_period_label fills "Predicted ___" with the right
    #     cadence ("next week" / "next month") so monthly bots don't read
    #     as if they were weekly. Defaults preserve existing weekly bots.
    cur_label = (display.get("underlying_label") or "Current price") if display else "Current price"
    period_label = (display.get("prediction_period_label") or "next week") if display else "next week"
    out.append("<div class='row compact'>")
    out.append(f"<div class='card'><div class='label'>{html.escape(cur_label)}</div>"
               f"<div class='value'>{cur_str}</div></div>")
    out.append(f"<div class='card'><div class='label'>Predicted {html.escape(period_label)}</div>"
               f"<div class='value'>{pred_str}</div></div>")
    out.append(f"<div class='card'><div class='label'>Median change</div>"
               f"<div class='value'>{chg_str}</div></div>")
    out.append(f"<div class='card'><div class='label'>P(price goes up)</div>"
               f"<div class='value'>{prob_str}</div></div>")
    out.append(f"<div class='card'><div class='label'>Lower 5%</div>"
               f"<div class='value'>{q05_str}</div></div>")
    out.append(f"<div class='card'><div class='label'>Upper 95%</div>"
               f"<div class='value'>{q95_str}</div></div>")
    out.append("</div>")  # /row

def _render_watchlist_hero(out: List[str],
                            watchlist: List[dict],
                            model: dict | None,
                            underlying_history: List[dict],
                            display: dict,
                            latest_active: dict | None,
                            kalshi_history: List[dict] | None = None,
                            prob_history: List[dict] | None = None,
                            atm_market: dict | None = None,
                            contract_open_ts: float | None = None,
                            contract_close_ts: float | None = None,
                            event_title: str | None = None) -> None:
    """Kalshi-style hero block: current implied-underlying forecast,
    value delta, time-to-close on the soonest market, and a chart of
    the forecast over the contract life. The hero forecast value
    updates as the user scrubs the line — the JS swaps it for the
    value at the cursor's timestamp, then restores the live current
    when the cursor leaves the chart.
    """
    # Chart data source — prefer the probability series (one point per
    # bot poll, y-axis pinned 0..100¢) so the line reflects how the
    # tracked ticker's implied probability moved over time. Falls back
    # to the Kalshi implied-underlying when there's no probability
    # history yet (e.g. a freshly-registered bot with empty
    # market_views).
    prob_history = prob_history or []
    use_prob = bool(prob_history)
    if use_prob:
        chart_history = prob_history
        chart_display = {
            "divisor": 1.0,
            "underlying_decimals": 0,
            "underlying_unit": "%",
            "unit_position": "suffix",
        }
    else:
        chart_history = kalshi_history or []
        chart_display = display

    current: float | None = None
    earliest_value: float | None = None
    if chart_history:
        for r in reversed(chart_history):
            v = r.get("value")
            if v is None:
                continue
            try:
                current = float(v)
                break
            except (TypeError, ValueError):
                continue
        for r in chart_history:
            v = r.get("value")
            if v is None:
                continue
            try:
                earliest_value = float(v)
                break
            except (TypeError, ValueError):
                continue
    if not use_prob:
        # Fallbacks only meaningful for the implied-underlying view —
        # the probability series has no equivalent of model.current_gas_price.
        if current is None and model is not None and model.get("current_gas_price") is not None:
            try:
                current = float(model["current_gas_price"])
            except (TypeError, ValueError):
                current = None
        if earliest_value is None and underlying_history:
            for r in underlying_history:
                v = r.get("value")
                if v is None:
                    continue
                try:
                    earliest_value = float(v)
                    break
                except (TypeError, ValueError):
                    continue

    # Raw value delta over the visible chart window (e.g. "▼ 9.05K").
    # Replaces the prior percent-change indicator: the user wants to see
    # the actual underlying delta in native units, not a normalized %.
    value_change: float | None = None
    if current is not None and earliest_value is not None:
        value_change = current - earliest_value

    # Total Kalshi volume across the visible watchlist + soonest close.
    vols = [int(r.get("volume") or 0) for r in watchlist
            if r.get("volume") is not None]
    total_volume = sum(vols)
    mtc_values = [float(r.get("minutes_to_close")) for r in watchlist
                  if r.get("minutes_to_close") is not None
                  and float(r.get("minutes_to_close")) > 0]
    soonest_mtc = min(mtc_values) if mtc_values else None

    # Per-bot display formatting + active-strike overlay (if any).
    label = display.get("underlying_label", "Underlying") if display else "Underlying"
    # No live contract — dash out the forecast + change indicator. Two
    # cases collapse to the same display: (a) the current event has
    # already settled (close_ts is in the past), or (b) Kalshi returned
    # no active markets in this series at all (between events, or the
    # fetch errored). In either case the "current forecast" isn't
    # forecasting anything — Kalshi's last printed price is just the
    # settlement value of an expired market, and a fresh series might
    # not have started yet.
    now_ts = datetime.now(timezone.utc).timestamp()
    contract_is_closed = (
        contract_close_ts is None
        or contract_close_ts <= now_ts
    )
    if contract_is_closed:
        current_str = "—"
        change_body = "—"
        change_cls = ""
        value_change = None
    else:
        current_str = fmt_underlying(current, chart_display)
        # Format the raw delta in the bot's native units, then strip the
        # leading sign (the arrow already conveys direction).
        if value_change is None:
            change_body = "—"
            change_cls = ""
        else:
            signed = _fmt_signed_underlying(value_change, chart_display)
            # _fmt_signed_underlying emits "+" or "−" as the first char.
            change_body = signed.lstrip("+−-")
            change_cls = "pos" if value_change >= 0 else "neg"

    active_strike = None
    active_side = None
    if latest_active:
        # The positions table doesn't carry strike_low / strike_high
        # directly — those live on market_views. Look them up by ticker.
        # For "between" markets we plot the midpoint; for "above $X" we
        # plot the lower bound.
        wl_row = next(
            (w for w in watchlist
             if w.get("ticker") == latest_active.get("ticker")),
            None,
        )
        if wl_row is not None:
            sl = wl_row.get("strike_low")
            sh = wl_row.get("strike_high")
            if sl is not None and sh is not None:
                try:
                    active_strike = (float(sl) + float(sh)) / 2
                except (TypeError, ValueError):
                    pass
            elif sl is not None:
                try:
                    active_strike = float(sl)
                except (TypeError, ValueError):
                    pass
        active_side = (latest_active.get("side") or "").upper() or None

    # Header layout (per user request): top-left is the volume of the
    # contract the chart line represents (the ATM market) — the
    # forecast value + arrow/change indicator that used to live here
    # were removed because they duplicate the information the chart's
    # right-edge already conveys visually. Top-right keeps the
    # time-to-close on the soonest market.
    contract_volume: float | None = None
    if atm_market:
        v = (atm_market.get("volume_fp")
             if atm_market.get("volume_fp") is not None
             else atm_market.get("volume"))
        if v is not None:
            try:
                contract_volume = float(v)
            except (TypeError, ValueError):
                contract_volume = None
    if contract_volume is None and total_volume:
        # No specific chart contract — show the visible watchlist's
        # aggregate Kalshi volume instead. Keeps the panel populated
        # for bots where the ATM lookup didn't return a market (e.g.
        # JSON-source bots with no per-ticker volume field).
        contract_volume = float(total_volume)
    if contract_volume is None:
        volume_str = "—"
    elif contract_volume >= 1e6:
        volume_str = f"{contract_volume/1e6:.2f}M"
    elif contract_volume >= 1e3:
        volume_str = f"{contract_volume/1e3:.1f}K"
    else:
        volume_str = f"{int(contract_volume):,}"

    out.append("<div class='wl-hero'>")
    out.append("<div class='wl-hero-top'>")
    out.append("<div class='wl-hero-stats'>")
    # Static volume display — no hover-swap behaviour (the chart
    # hover JS only swaps elements when it finds them; the legacy
    # .wl-hero-price / .wl-hero-change selectors are gone so the JS
    # cleanly no-ops on the swap step).
    out.append(
        f"<div class='wl-hero-volume'>"
        f"<span class='wl-hero-volume-text'>{html.escape(volume_str)}</span>"
        f"<span class='wl-hero-volume-label'>volume</span>"
        f"</div>"
    )
    out.append("</div>")  # /wl-hero-stats
    out.append(f"<div class='wl-hero-mtc'>"
               f"<span class='label'>Closes in</span> "
               f"<span class='value'>{time_left_str(soonest_mtc)}</span>"
               f"</div>")
    out.append("</div>")  # /wl-hero-top

    # Probability mode: pin y-axis to 0..100 (the value range a Kalshi
    # binary contract can ever take) and reference the active bet's
    # entry probability (entry_price_cents) — which lives on the same
    # 0..100 scale. Otherwise auto-scale as before and reference the
    # active bet's strike value.
    if use_prob:
        if latest_active is not None:
            try:
                reference_strike = float(
                    latest_active.get("entry_price_cents"))
                strike_is_active = True
            except (TypeError, ValueError):
                reference_strike = None
                strike_is_active = False
        else:
            reference_strike = None
            strike_is_active = False
        strike_side = active_side
        y_pin_min, y_pin_max = 0.0, 100.0
    else:
        reference_strike = active_strike
        strike_side = active_side
        strike_is_active = active_strike is not None
        y_pin_min, y_pin_max = None, None

    # Chart plots the chosen series (probability when present, else
    # Kalshi's implied-underlying forecast). svg_kalshi_chart handles
    # the <2 datapoint case internally.
    out.append(svg_kalshi_chart(
        chart_history, chart_display,
        reference_strike=reference_strike,
        strike_side=strike_side,
        strike_is_active_bet=strike_is_active,
        contract_open_ts=contract_open_ts,
        contract_close_ts=contract_close_ts,
        total_volume=total_volume,
        y_min=y_pin_min,
        y_max=y_pin_max,
    ))
    out.append("</div>")


# Tennis-shape ``rules_primary`` sentences are boilerplate of the form
# "If X wins the {matchup} professional tennis match in the
# {YEAR} {Event} after a ball has been played, then the market
# resolves to Yes." The regex below captures the {Event} substring —
# the run of words between the year and the round marker.

def _render_watchlist(out: List[str], watchlist: List[dict],
                      model: dict | None,
                      underlying_history: List[dict] | None = None,
                      display: dict | None = None,
                      latest_active: dict | None = None,
                      bot_active_bets: List[dict] | None = None,
                      kalshi_history: List[dict] | None = None,
                      prob_history: List[dict] | None = None,
                      atm_market: dict | None = None,
                      contract_open_ts: float | None = None,
                      contract_close_ts: float | None = None,
                      event_title: str | None = None,
                      threshold_source: dict | None = None,
                      edge_cfg: dict | None = None,
                      validator_cfg: dict | None = None,
                      risk_caps: dict | None = None,
                      hedge_cfg: dict | None = None,
                      extra_cfg: dict | None = None,
                      available_bots: List[dict] | None = None,
                      current_bot: str = "",
                      period_key: str = "all") -> None:
    # Buy-criteria reference button — a small circle-i info icon that
    # opens the shared rules modal. Built up-front so the sport-bot
    # header can pin it to the top-right of the h2 row; non-sport bots
    # still use the same html further below (inline with the Active
    # bets h3).
    rules_payload = json.dumps({
        "edge": edge_cfg or {},
        "validators": validator_cfg or {},
        "risk": risk_caps or {},
        "hedge": hedge_cfg or {},
        "extra": extra_cfg or {},
        "_source": threshold_source or {"source": "fallback",
                                          "captured_at": None,
                                          "missing_keys": []},
    }, separators=(",", ":"), default=str)
    rules_icon_html = (
        "<button type='button' class='criteria-rules-btn' "
        f"data-rules='{html.escape(rules_payload)}' "
        f"title=\"What does this bot need before it'll buy?\">"
        "i</button>"
    )

    # Sport bots render as three top-level sections (Active bets ·
    # Model vs market · Kalshi rules — the third comes from
    # ``_render_contract_rules`` after this function returns). Non-
    # sport bots keep the legacy single-section layout.
    is_sport_bot = current_bot in {"nba", "wnba", "tennis", "table-tennis",
                                    "darts", "world-cup", "mlb"}
    is_billboard_bot = current_bot == "billboard"
    if not is_sport_bot:
        out.append("<div class='section'><h2>"
                   "Watchlist — model vs market</h2>"
                   "<div class='body'>")
        # Current-prediction card row (Current price, Predicted next
        # week, etc.) — model-based hero. Skipped for sport bots
        # because each match has its own book, so there's no unified
        # "current forecast" to show.
        contract_is_closed = (
            contract_close_ts is None
            or contract_close_ts <= datetime.now(timezone.utc).timestamp()
        )
        _render_current_prediction(out, model, display=display,
                                     contract_is_closed=contract_is_closed)

    # ── Build the held-tickers map (needed by the verdict cell + row
    # sort even when we don't render the Active bets section below).
    bets = list(bot_active_bets or [])
    if not bets and latest_active:
        bets = [latest_active]
    held_by_ticker = {b.get("ticker"): b for b in bets if b.get("ticker")}
    n_bets = len(bets)

    # ── Kalshi-truth held tickers ─────────────────────────────────────
    # Row highlighting on the Model-vs-market panel (the HOLDING badge,
    # the .row-bought colouring) uses this map instead of the paper-
    # positions map above, so the highlight only fires when Kalshi's
    # portfolio actually lists an open position on that ticker. Paper
    # positions on sim used to slip through as HOLDING which was
    # misleading — nothing was bought on real Kalshi. On failure /
    # missing creds, ``get_open_positions`` returns None and we treat
    # it as "no highlight" so a transient API error doesn't paint every
    # row as held.
    kalshi_held_by_ticker: Dict[str, Dict[str, Any]] = {}
    try:
        from .kalshi_client import get_open_positions as _get_open_positions
        _kalshi_pos, _err = _get_open_positions()
        if _kalshi_pos is not None:
            for _p in _kalshi_pos:
                _tk = _p.get("ticker")
                if not _tk:
                    continue
                try:
                    _pos_fp = float(_p.get("position_fp")
                                     or _p.get("position") or 0)
                except (TypeError, ValueError):
                    _pos_fp = 0.0
                # Kalshi reports YES-side count as +, NO-side as −.
                _side = "YES" if _pos_fp > 0 else "NO" if _pos_fp < 0 else ""
                _avg = _p.get("average_price_cents")
                if _avg is None:
                    _avg = _p.get("avg_price_cents")
                if _avg is None:
                    # /portfolio/positions carries dollar totals, not an
                    # avg-price field — derive it so the Active-bets
                    # entry %, entry cost and total cost cells populate
                    # for real positions.
                    try:
                        # market_exposure = cost basis of the CURRENT
                        # position. total_traded is GROSS volume (buys
                        # + sells), so after any partial exit it
                        # over-prices the entry (2026-07-14: FARWAW
                        # showed a 102% entry after a 78c sell).
                        _traded = float(_p.get("market_exposure_dollars")
                                         or _p.get("total_traded_dollars")
                                         or 0)
                        if _traded > 0 and _pos_fp:
                            _avg = round(_traded / abs(_pos_fp) * 100)
                    except (TypeError, ValueError):
                        _avg = None
                _paper = held_by_ticker.get(_tk) or {}
                _record = {
                    "ticker": _tk,
                    "side": _side,
                    "contracts": int(abs(_pos_fp)) or None,
                    "entry_price_cents": (int(_avg)
                                          if _avg is not None else
                                          _paper.get("entry_price_cents")),
                    "opened_at": _paper.get("opened_at"),
                }
                # Register under the FULL Kalshi ticker AND the base
                # match_id (ticker minus its final ``-<SIDE>`` segment)
                # because sport-bot watchlist rows carry the base
                # match_id as ``data-ticker`` — the side is expressed
                # via ticker_a / ticker_b on the row, not appended to
                # the identifier. Without the base-form key the
                # highlight would never fire on any real sport-bot
                # position.
                kalshi_held_by_ticker[_tk] = _record
                if "-" in _tk:
                    _base = _tk.rsplit("-", 1)[0]
                    kalshi_held_by_ticker.setdefault(_base, _record)
    except Exception:  # noqa: BLE001
        log.exception("kalshi held-tickers lookup failed; "
                       "Model-vs-market highlight will be empty")

    # Non-sport bots: full Active-bets section + hero chart, same as
    # before. Sport bots skip this block; their Active bets and
    # Model-vs-market tables are emitted as separate top-level
    # sections at the bottom of this function.
    if not is_sport_bot:
        label = ("Active bets" if n_bets > 1 else "Active bet")
        count_suffix = (f" <span class='small gray'>({n_bets})</span>"
                         if n_bets > 1 else "")
        out.append(
            "<h3 class='subhead' "
            "style='display:flex;align-items:center;gap:8px;'>"
            f"{label}{count_suffix} {rules_icon_html}</h3>"
        )
        if bets:
            enriched_rows: List[dict] = []
            for ab in bets:
                enriched = dict(ab)
                wl_match = next(
                    (w for w in (watchlist or [])
                     if w.get("ticker") == ab.get("ticker")),
                    None,
                )
                if wl_match:
                    enriched.setdefault("floor_strike", wl_match.get("strike_low"))
                    enriched.setdefault("cap_strike", wl_match.get("strike_high"))
                    enriched.setdefault("minutes_to_close",
                                          wl_match.get("minutes_to_close"))
                    if enriched.get("mark_yes_ask") is None:
                        enriched["mark_yes_ask"] = wl_match.get("yes_ask_cents")
                    if enriched.get("mark_no_ask") is None:
                        enriched["mark_no_ask"] = wl_match.get("no_ask_cents")
                enriched["_display"] = display or {}
                enriched_rows.append(enriched)
            enriched_rows.sort(key=lambda r: r.get("opened_at", ""), reverse=True)
            out.append("<div class='watchlist-active-scroll'>")
            _render_active_bets_table(
                out, enriched_rows, show_bot=False,
                chart_link=True, hedge_cfg=hedge_cfg,
                watchlist=watchlist,
                event_title=event_title,
                is_sport_bot=False,
                display=display)
            out.append("</div>")
        else:
            out.append("<div class='empty'>No active bets right now.</div>")

        _render_watchlist_hero(out, watchlist, model,
                               underlying_history or [],
                               display or {}, latest_active,
                               kalshi_history=kalshi_history,
                               prob_history=prob_history or [],
                               atm_market=atm_market,
                               contract_open_ts=contract_open_ts,
                               contract_close_ts=contract_close_ts,
                               event_title=event_title)

    if not watchlist:
        if is_sport_bot:
            # No section wrapper is open yet for sport bots — emit both
            # sections with empty states so the page still looks like
            # the three-section layout.
            out.append("<div class='section'><h2>Active bets</h2>"
                       "<div class='body'>"
                       "<div class='empty'>No active bets right now.</div>"
                       "</div></div>")
            out.append("<div class='section'>")
            out.append(
                "<div style='display:flex;align-items:center;"
                "justify-content:space-between;gap:12px;"
                "padding:14px 18px 0 18px;'>"
                "<h2 style='margin:0;'>Model vs market</h2>"
                "<div style='display:flex;align-items:center;gap:8px;'>"
                "<span class='small gray'>Buy criteria</span>"
                f"{rules_icon_html}"
                "</div></div>"
                "<div class='body'>"
                "<div class='empty'>No open markets right now.</div>"
                "</div></div>"
            )
        else:
            out.append("<div class='empty'>No open markets right now.</div>")
            out.append("</div></div>")
        return

    # ── Pre-pass: enrich each row with EV/BE numbers, then sort by best
    # EV. Sorting by EV (not by gap or by alphabetical ticker) puts the
    # genuinely-actionable opportunities at the top of the table.
    for v in watchlist:
        ya = v.get("yes_ask_cents")
        na = v.get("no_ask_cents")
        spread = v.get("spread_cents") or 0
        half_spread_d = (spread / 2.0) / 100.0
        # Reference prob for EV — Pinnacle when the sport bot ships it,
        # else the bot's own model. Same rule as the Edge column below,
        # so EV and Edge tell the user the same story about the same
        # reference book. Sort-by-best-EV downstream is what drives the
        # "top of the watchlist" ordering the user actually acts on;
        # basing that on Pinnacle (when we have it) keeps the ordering
        # honest.
        p_yes_blend = v.get("pinnacle_prob_yes")
        if p_yes_blend is None:
            p_yes_blend = v.get("model_prob_yes")
        be_yes = (ya / 100.0) if ya is not None else None
        be_no = (na / 100.0) if na is not None else None
        # Net-of-fee EV. The Kalshi entry fee (ceil(0.07 × p × (1−p))
        # per contract) is charged at open; settlement at 0¢ / 100¢ has
        # no exit fee, so on a held-to-settle bet the only deduction is
        # the entry fee. Per-$1-contract figure: divide cents by 100.
        fee_yes_d = kalshi_fee_cents(ya, 1) / 100.0 if ya is not None else 0.0
        fee_no_d = kalshi_fee_cents(na, 1) / 100.0 if na is not None else 0.0
        ev_yes = ((p_yes_blend - be_yes) - half_spread_d - fee_yes_d
                  if p_yes_blend is not None and be_yes is not None else None)
        ev_no = (((1.0 - p_yes_blend) - be_no) - half_spread_d - fee_no_d
                 if p_yes_blend is not None and be_no is not None else None)
        # Best side by EV (only among the sides we have prices for).
        candidates = [(s, e) for s, e in (("YES", ev_yes), ("NO", ev_no))
                      if e is not None]
        best_side, best_ev = (None, None)
        if candidates:
            best_side, best_ev = max(candidates, key=lambda x: x[1])
        v["_ev_yes"] = ev_yes
        v["_ev_no"] = ev_no
        v["_be_yes"] = be_yes
        v["_be_no"] = be_no
        v["_best_side"] = best_side
        v["_best_ev"] = best_ev
    # Filter to rows that have at least 1 open contract — markets with
    # zero open interest aren't tradeable and clutter the table. Rows
    # that set ``_skip_oi_filter`` (e.g. billboard markets that may
    # have null Kalshi-side OI early in the chart week but are still
    # the correct surface to show on the dashboard) opt out. Held
    # tickers are never filtered out: the user always needs visibility
    # into positions they actually own, regardless of current liquidity.
    # Sport bots additionally keep zero-OI rows that carry a live quote:
    # a freshly-listed game nobody has traded yet (OI 0) is still
    # tradeable — you'd just be the first fill — and hiding it made
    # next-day games vanish from Model vs market until someone else
    # traded (2026-07-09, WNBA DAL@TOR).
    watchlist = [r for r in watchlist
                 if r.get("_skip_oi_filter")
                 or (r.get("open_interest") or 0) > 0
                 or r.get("ticker") in held_by_ticker
                 or (is_sport_bot and r.get("yes_ask_cents") is not None)]
    # Sort: sport bots (one row per game / match) have no strike axis,
    # so order by is-held → actionability → |best EV| descending. Held
    # rows always sort to the top so the user immediately sees what's
    # open regardless of where today's EV places it; matches the
    # billboard bot's pattern already in use below. (is_billboard_bot
    # is computed at the top of the function.)
    if is_sport_bot:
        def _sport_sort_key(r: dict) -> Tuple[int, int, float]:
            is_held = 0 if r.get("ticker") in held_by_ticker else 1
            v = r.get("bot_verdict") or "SKIP"
            actionable = 0 if v in ("BUY_YES", "BUY_NO") else 1
            ev = r.get("_best_ev")
            try:
                ev_mag = -abs(float(ev)) if ev is not None else 0.0
            except (TypeError, ValueError):
                ev_mag = 0.0
            return (is_held, actionable, ev_mag)
        watchlist = sorted(watchlist, key=_sport_sort_key)
    elif is_billboard_bot:
        # Each event has ~40 rows, almost all SKIPs. Surface only the
        # rows the bot would actually buy, ordered best-EV first, and
        # cap at 10 so the table stays scannable. Held positions
        # always sort first so the user can see what's currently
        # owned regardless of where today's EV places it — without
        # this, a row whose entry edge has already played out can
        # drop below the truncate cutoff and disappear entirely.
        def _billboard_sort_key(r: dict) -> Tuple[int, int, float]:
            is_held = 0 if r.get("ticker") in held_by_ticker else 1
            v = r.get("bot_verdict") or "SKIP"
            actionable = 0 if v in ("BUY_YES", "BUY_NO") else 1
            ev = r.get("_best_ev")
            try:
                ev_neg = -float(ev) if ev is not None else 0.0
            except (TypeError, ValueError):
                ev_neg = 0.0
            return (is_held, actionable, ev_neg)
        watchlist = sorted(watchlist, key=_billboard_sort_key)[:10]
    else:
        watchlist = sorted(
            watchlist,
            key=lambda r: (r.get("strike_low")
                           if r.get("strike_low") is not None else 9_999.0,
                           r.get("ticker") or ""),
        )

    # Column layout (per user spec): Ticker | Question | Contracts |
    # Kalshi YES + NO grouped | My YES + NO grouped | EV YES + NO
    # grouped | Verdict (rightmost). Chance was redundant with Kalshi
    # YES (same midpoint of the bid/ask); volume and closes-in live in
    # the hero header instead of being repeated per row.
    # Sport bots (NBA + tennis-shape) show Title + Side: the Title
    # carries Kalshi's published YES question ("Will MIN win the
    # SAS vs MIN game?") and Side carries the team / player being
    # bet on. Non-sport bots (gas / CPI / jobless) keep the
    # strike-band Question column too. is_sport_bot was already
    # computed above to drive the sport-specific row sort.
    if is_sport_bot:
        head_cols = (
            "<th title='Kalshi-published contract title — the "
            "YES question shown on the market page.'>Title</th>"
            "<th title='Who the bot is betting will win.'>Side</th>"
        )
    elif is_billboard_bot:
        # Billboard layout: Title | Song | Artist. Title carries the
        # full Kalshi-published question (so the chart week is visible
        # per row), Song + Artist break out the two pieces of identity
        # that the long Title stem buries.
        head_cols = (
            "<th title='Kalshi-published contract title — the YES "
            "question shown on the market page.'>Title</th>"
            "<th title='Song title (the contract resolves YES if this "
            "song is top 10 on the Billboard Hot 100 for the listed "
            "chart week).'>Song</th>"
            "<th title='Recording artist.'>Artist</th>"
        )
    else:
        head_cols = (
            "<th title='Kalshi-published contract title — the "
            "YES question shown on the market page.'>Title</th>"
            "<th>Question</th>"
        )
    # Settled-row filter — applied on every bot, both sport and non-
    # sport, both Active-bets and Model-vs-market panes. A contract
    # counts as "settled" (and therefore is not visible on the
    # watchlist) when any of the following signals fire:
    #
    #   1. ``completed=True`` from the exporter — the authoritative
    #      flag Kalshi flips once ``status`` moves to
    #      ``closed`` / ``settled`` / ``finalized``.
    #   2. ``expected_expiration_time`` is more than 15 minutes in the
    #      past. Kalshi is slow to flip ``status`` on tennis matches
    #      that have already ended in real life (Juhas vs Klok 2026-
    #      07-09 was 90-min-past-expiration but ``completed`` still
    #      False), and a 15-minute buffer avoids false positives on
    #      matches expiring imminently but still tradeable.
    #   3. Either side's ask is at an extreme (≤ 1¢ or ≥ 99¢). Kalshi
    #      pins the winning contract at 99¢ / losing at 1¢ once the
    #      match resolves, well before the market status catches up.
    #   4. Every ask field we can see is None — both sides withdrawn.
    #
    # We look for asks under both name shapes because
    # ``build_standard_watchlist_rows`` collapses per-side ask fields
    # (``yes_ask_cents_a`` / ``_b``) into a single top/bottom pair
    # (``yes_ask_cents`` / ``no_ask_cents``) on tennis-shape bots.
    now_ts = datetime.now(timezone.utc).timestamp()

    def _is_settled(r: dict) -> bool:
        if r.get("completed"):
            return True
        exp = r.get("expected_expiration_time")
        if exp:
            try:
                if isinstance(exp, (int, float)):
                    exp_ts = float(exp)
                else:
                    # ISO 8601 with a trailing "Z" — normalise to +00:00
                    exp_ts = datetime.fromisoformat(
                        str(exp).replace("Z", "+00:00")
                    ).timestamp()
                if exp_ts <= now_ts - 15 * 60:
                    return True
            except (TypeError, ValueError):
                pass
        ask_fields = (
            "yes_ask_cents", "no_ask_cents",
            "yes_ask_cents_a", "yes_ask_cents_b",
        )
        asks = tuple(r.get(f) for f in ask_fields)
        # Any side clamped to an extreme → contract has priced-in the
        # decisive side. Only fire when the extreme is on YES / NO ask
        # cents, not on internal "confidence" fields; asks are ints
        # in [0, 100].
        for a in asks:
            if isinstance(a, (int, float)) and (a <= 1 or a >= 99):
                return True
        return all(a is None for a in asks)

    # Split rows for sport bots: held rows go into the Active-bets
    # section, everything else into Model-vs-market. Non-sport bots
    # keep a single ordered list. Each section context carries its
    # own tbody id (for live-updates), position-columns flag, and
    # optional section header. The emission loop after the row
    # renderer walks this list once per section.
    if is_sport_bot:
        _held_rows = [r for r in watchlist
                       if r.get("ticker") in held_by_ticker]
        # Model-vs-market shows every watchlist row (including the
        # ones we hold). Duplicating held rows across both sections
        # is intentional (user 2026-07-11): the Active-bets pane
        # answers "what are we in?"; the Model-vs-market pane
        # answers "what does the model like right now?" — and having
        # our positions surface alongside the fresh watch pane makes
        # it easy to see how a bet we own compares to the alternatives
        # the model is currently evaluating. The row-bought /
        # bought-yes / bought-no CSS classes downstream apply the
        # highlight so held rows stand out visually.
        _open_rows = list(watchlist)
        # Model-vs-market rows only show up when a benchmark line is
        # quoting the match (user 2026-07-11: "if there is no model %
        # from odds sources, don't show it in model vs market
        # section"). The 2026-07-11 held-row exemption was retired at
        # the same time — a held row without a sharp benchmark ends
        # up as a blank Model % cell either way, so it belongs in
        # Active bets alone.
        #
        # With Betfair Exchange added to the cascade, the WNBA / WC
        # rationale for exemption ("Pinnacle posts hours after Kalshi
        # opens") no longer holds — Betfair typically has a line from
        # opening on those two. NBA joined 2026-07-09 with the
        # benchmark rearchitecture. table-tennis is EXEMPT: Pinnacle
        # currently quotes no TT at all, so the filter would blank the
        # table permanently.
        # 2026-07-13 (final): Model-vs-market only shows games that
        # HAVE odds from the benchmark sources (The Odds API cascade +
        # Pinnacle guest feed) — "please get these odds from the
        # sources ... and not show games when no odds are available."
        # Rows appear automatically as books post lines (Summer League
        # ~1-2h before tip, MLB the day before). table-tennis stays
        # exempt: its rows display the upstream Elo model's odds, so
        # the "no odds shown" concern doesn't apply.
        if current_bot in {"tennis", "darts",
                            "wnba", "world-cup", "mlb", "nba"}:
            _open_rows = [r for r in _open_rows
                           if r.get("pinnacle_prob_yes") is not None]
        # Zero-volume filter (user 2026-07-13): a match where the pair's
        # total Kalshi volume is 0 hasn't traded at all — hide it from
        # Model-vs-market until someone trades. Applies to every sport
        # pane (table-tennis included — volume is Kalshi-side data, so
        # the no-benchmark exemption above doesn't carry over). Rows
        # with volume=None (bot doesn't stamp it) stay visible: unknown
        # is not zero.
        _open_rows = [r for r in _open_rows
                       if r.get("volume") is None
                       or (r.get("volume") or 0) > 0]
        # Settled-row filter on both panes — a resolved match
        # shouldn't sit in Active bets either (the position is
        # already locked in, the buy/sell watch state is stale).
        _held_rows = [r for r in _held_rows if not _is_settled(r)]
        _open_rows = [r for r in _open_rows if not _is_settled(r)]
        # Every held bet must render in Active bets even when its
        # market has already left the watchlist (game finished but the
        # position hasn't settled yet — exactly the overnight window
        # where the user most wants to see what's still on the book).
        # Synthesize a minimal standard-shape row from the bet record
        # for any held ticker without a live watchlist row; the
        # position cells (contracts / entry cost / total cost) come
        # from held_by_ticker as usual. Added AFTER the settled filter
        # on purpose: these rows live until the position leaves the
        # book, not until Kalshi's quote pins.
        # Dedupe against the event-ticker keys the watchlist uses AS
        # WELL AS the side-specific tickers Kalshi records on the
        # position. Without stripping the ``-<SIDE>`` suffix the synth
        # loop would double-count every held position that ALSO has a
        # matching row on the fresh watchlist — one row for the event
        # ticker and one for the full side ticker.
        _covered_held: set[str] = set()
        for r in _held_rows:
            t = r.get("ticker") or ""
            _covered_held.add(t)
            if "-" in t:
                _covered_held.add(t.rsplit("-", 1)[0])
        for _ab in bets:
            _t = _ab.get("ticker") or ""
            _t_base = _t.rsplit("-", 1)[0] if "-" in _t else _t
            if not _t or _t in _covered_held or _t_base in _covered_held:
                continue
            _covered_held.add(_t)
            _covered_held.add(_t_base)
            _mark = _ab.get("mark_mid")
            _held_rows.append({
                "ticker": _t,
                "direction": "yes",
                "strike_low": None,
                "strike_high": None,
                "yes_ask_cents": (int(round(_mark))
                                   if _mark is not None else None),
                "no_ask_cents": None,
                "spread_cents": None,
                "volume": None,
                "open_interest": None,
                "model_prob_yes": _ab.get("model_yes_prob_at_entry"),
                "raw_model_prob_yes": None,
                "pinnacle_prob_yes": None,
                "_skip_oi_filter": True,
                "bot_verdict": "HOLDING",
                "rejection_reason": "",
                "title": (_ab.get("_title") or _ab.get("title")
                           or _ab.get("_match") or _t),
                "minutes_to_close": _ab.get("minutes_to_close"),
                "_yes_label": _ab.get("_side_player") or "",
                "_no_label": "",
                "rules_primary": "",
                "tournament": _ab.get("_tournament") or "",
            })
        section_ctxs = [
            {
                "kind": "active",
                "rows": _held_rows,
                "include_position_cols": True,
                "tbody_id": "watchlist-tbody-active",
                "empty_msg": "No active bets right now.",
            },
            {
                "kind": "model-vs-market",
                "rows": _open_rows,
                "include_position_cols": False,
                "tbody_id": "watchlist-tbody",
                "empty_msg": "No open markets right now.",
            },
        ]
    else:
        # Non-sport bots: settled contracts (past expiration, extreme
        # ask, etc.) are equally uninformative in the strike-ladder
        # watchlist — the settlement price has already been decided,
        # the row can't be traded, and leaving it in place shifts the
        # user's eye away from the actionable strikes.
        section_ctxs = [{
            "kind": "single",
            "rows": [r for r in watchlist if not _is_settled(r)],
            "include_position_cols": True,
            "tbody_id": "watchlist-tbody",
            "empty_msg": "No open markets right now.",
        }]

    for _ctx in section_ctxs:
        _rows_to_emit = _ctx["rows"]
        include_position_cols = _ctx["include_position_cols"]
        _tbody_id = _ctx["tbody_id"]

        # Sport bots wrap each table in its own `<div class='section'>`
        # with its own h2 title. Non-sport bots share the single
        # section opened at the top of the function.
        if is_sport_bot:
            if _ctx["kind"] == "active":
                out.append("<div class='section'><h2>Active bets</h2>"
                           "<div class='body'>")
                if not _rows_to_emit:
                    out.append(
                        f"<div class='empty'>{_ctx['empty_msg']}</div>"
                    )
                    out.append("</div></div>")
                    continue
            else:  # model-vs-market
                out.append("<div class='section'>")
                out.append(
                    "<div style='display:flex;align-items:center;"
                    "justify-content:space-between;gap:12px;"
                    "padding:14px 18px 0 18px;'>"
                    "<h2 style='margin:0;'>Model vs market</h2>"
                    "<div style='display:flex;align-items:center;gap:8px;'>"
                    "<span class='small gray'>Buy criteria</span>"
                    f"{rules_icon_html}"
                    "</div></div><div class='body'>"
                )
                if not _rows_to_emit:
                    out.append(
                        f"<div class='empty'>{_ctx['empty_msg']}</div>"
                    )
                    out.append("</div></div>")
                    continue

        # Position columns. On the sport-bot Active bets table (user
        # 2026-07-10) we drop Kalshi entry cost and rename Total cost
        # → Kalshi total cost; the surviving cluster is My contracts +
        # Kalshi total cost + Closes in, ordered to the right of Total
        # contracts. Model-vs-market on sport bots keeps ``pos_head``
        # empty (include_position_cols=False); Non-sport single tables
        # keep the legacy 3-col layout so nothing regresses.
        is_active = _ctx.get("kind") == "active"
        pos_head = ""
        if include_position_cols and is_active:
            pos_head = (
                "<th class='num' title='Number of contracts bought on this row. Blank when no position is open.'>Contracts bought</th>"
                "<th class='num' title='Top: potential earnings — settlement pays $1 × contracts (no exit fee); the figure shown is that payout minus what you paid (entry + fee), i.e. the net earnings. Bottom: Kalshi total cost — entry price × contracts + Kalshi entry fee. Blank when no position is held on this row.'>Investment</th>"
            )
        elif include_position_cols:
            pos_head = (
                "<th class='num' title='Number of contracts held on this row. Blank when no position is open.'>My contracts</th>"
                "<th class='num' title='Kalshi entry cost — entry price × contracts, before fees. Blank when no position is held on this row.'>Kalshi entry cost</th>"
                "<th class='num' title='Total cash out at open — Kalshi entry cost + Kalshi entry fee. Blank when no position is held on this row.'>Total cost</th>"
            )
        # Event column — shown on both Active bets and Model-vs-market
        # for sport bots (user asked for Event visibility on Active
        # bets too so held positions carry the same competition context
        # as the watch pane). Derived at render time from each row's
        # Kalshi ``rules_primary`` text (see ``_tennis_event_label``),
        # so it's live without a bot redeploy.
        _show_event_col = (is_sport_bot
                            and _ctx.get("kind") in
                                ("model-vs-market", "active"))
        event_head = (
            "<th title='Competition parsed from the Kalshi rules text "
            "— tour · tournament for tennis (ATP · Wimbledon, "
            "ITF · M15 Tokyo), league for basketball (WNBA / NBA). "
            "Falls back to the bot&apos;s competition label when the "
            "rules don&apos;t match a known template.'>Event</th>"
            if _show_event_col else ""
        )
        # Date column — every table (Model-vs-market, the single
        # combined non-sport-bot table, and Active bets), immediately
        # right of Rules per user spec. Parsed from the ticker's
        # encoded YYMMMDD (rules-text date as fallback).
        _show_date_col = _ctx.get("kind") in (
            "model-vs-market", "single", "active",
        )
        date_head = (
            "<th title='Market date parsed from the Kalshi ticker "
            "(game day / settlement day). Blank when the ticker "
            "doesn&apos;t encode one.'>Date</th>"
            if _show_date_col else ""
        )
        # Header layout diverges on the Active bets table (user
        # 2026-07-10, updated same day: probability columns pushed to
        # the far right, Total contracts dropped, "Contracts bought"
        # naming, live price labelled explicitly):
        #   Rules | Date | Event | Title | Side
        #     | Verdict | Contracts bought | Kalshi total cost
        #     | Closes in
        #     | Model % | Kalshi entry % | Kalshi live %
        # Model-vs-market + non-sport single keep the legacy layout
        # (Edge / EV / no entry-% column / Closes in before Verdict).
        if is_active:
            header_middle = (
                f"{pos_head}"
                "<th class='num' title='Model probability — sharp devigged reference from the best available benchmark book (Pinnacle first, else Betfair Exchange UK / EU). Em-dash for matches no sharp book is quoting or when the Odds API key isn&apos;t set. YES on top, NO on bottom.'>Model %</th>"
                "<th class='num' title='Kalshi entry % — the implied probability for each side at the price we paid, expressed on the yes-axis. YES on top, NO on bottom.'>Kalshi entry %</th>"
                "<th class='num' title='Live Kalshi market price — updates continuously. YES on top (green), NO on bottom (red).'>Kalshi live %</th>"
                "<th class='num' title='Time until the contract settles. Parsed from the Kalshi ticker&apos;s encoded date.'>Closes in</th>"
            )
        else:
            header_middle = (
                "<th class='num' title='Open interest — total contracts currently held open across all traders on this strike.'>Total contracts</th>"
                "<th class='num' title='Model probability — sharp devigged reference from the best available benchmark book (Pinnacle first, else Betfair Exchange UK / EU). Em-dash for matches no sharp book is quoting or when the Odds API key isn&apos;t set. YES on top, NO on bottom.'>Model %</th>"
                "<th class='num' title='Live Kalshi market price — YES on top (green), NO on bottom (red). Each side&apos;s implied probability that side wins.'>Kalshi %</th>"
                "<th class='num' title='Edge = benchmark probability (Pinnacle / Betfair) − Kalshi price, per side. YES on top (green), NO on bottom (red).'>Edge</th>"
                "<th class='num'>EV"
                "<button type='button' class='ev-info-btn' "
                "title='How is EV calculated?' "
                "aria-label='How is EV calculated?'>i</button>"
                "</th>"
                "<th class='num' title='Time until the contract settles. Parsed from the Kalshi ticker&apos;s encoded date.'>Closes in</th>"
                "<th>Verdict</th>"
                f"{pos_head}"
            )
        out.append("<div class='watchlist-scroll'>"
                   "<table><thead><tr>"
                   "<th title='Kalshi resolution rule for this contract — click to read.'>Rules</th>"
                   f"{date_head}"
                   f"{event_head}"
                   f"{head_cols}"
                   f"{header_middle}"
                   f"</tr></thead><tbody id='{html.escape(_tbody_id)}'>")
        for v in _rows_to_emit:
            ticker = v.get("ticker", "")
            qstr = question_str(v.get("direction", ""), v.get("strike_low"),
                                 v.get("strike_high"), display=display)
            # Detect whether this row is a held position on the row's
            # underdog (NO) side per the adapter's favored-side flip.
            # When it is, re-orient the row so YES tracks the HELD side
            # across every downstream cell — Model % / Kalshi % /
            # Entry % / the Side player label. Applies to BOTH the
            # Active bets AND Model-vs-market tables (user 2026-07-11:
            # France vs Spain rendered France on top with a HOLDING
            # badge while the actual position was Spain — for a held
            # row, "what am I holding" beats "who's favoured").
            _held_probe = kalshi_held_by_ticker.get(ticker)
            flip_active = (
                _held_probe is not None
                and (_held_probe.get("ticker") or "")
                    == (v.get("_no_ticker") or "")
            )
            ya_c = v.get("yes_ask_cents"); na_c = v.get("no_ask_cents")
            spread_cents = v.get("spread_cents")
            # Volume still drives the "thin volume" row-suspect flag below
            # but the column itself is gone — the hero shows the watchlist
            # total instead.
            volume = v.get("volume")
            oi = v.get("open_interest")
            oi_str = f"{int(round(float(oi))):,}" if oi is not None else "—"
            # Derive missing side from the other when only one ask is
            # quoted — render as a plain number (no "~" prefix) so the
            # cell parses as a real percentage. The derivation is exact
            # for binary contracts (YES + NO must sum to 100¢), so the
            # tilde was just adding noise.
            if ya_c is not None:
                kyes_str = f"{ya_c}%"
            elif na_c is not None:
                kyes_str = f"{100 - na_c}%"
            else:
                kyes_str = "—"
            if na_c is not None:
                kno_str = f"{na_c}%"
            elif ya_c is not None:
                kno_str = f"{100 - ya_c}%"
            else:
                kno_str = "—"
            p = v.get("model_prob_yes")
            raw_p = v.get("raw_model_prob_yes")
            my_yes_str = f"{int(round(float(p)*100))}%" if p is not None else "—"
            my_no_str = f"{int(round((1-float(p))*100))}%" if p is not None else "—"
            # Tooltip exposes the un-blended raw model probability so a 1pt
            # blended display doesn't hide a 30pt raw disagreement (or
            # vice versa). Lets the user audit "is the bot's actual view
            # justified, or is the blend doing all the work?"
            my_yes_tt = ""
            my_no_tt = ""
            if raw_p is not None and p is not None:
                raw_yes_pct = int(round(float(raw_p) * 100))
                raw_no_pct = int(round((1 - float(raw_p)) * 100))
                blended_yes_pct = int(round(float(p) * 100))
                blended_no_pct = int(round((1 - float(p)) * 100))
                my_yes_tt = (f" title='Raw model: {raw_yes_pct}% · "
                             f"Blended (vs Kalshi, skill-weighted): {blended_yes_pct}%'")
                my_no_tt = (f" title='Raw model: {raw_no_pct}% · "
                            f"Blended: {blended_no_pct}%'")

            # Validation flags — surfaced via row dimming + tooltip only.
            # The Gap column was removed; EV YES + EV NO already convey the
            # same edge information with spread cost baked in.
            flags = []
            if ya_c is None or na_c is None:
                flags.append("one-sided book")
            if spread_cents is not None and spread_cents > 8:
                flags.append("wide spread")
            if p is not None and 0.40 <= p <= 0.60:
                flags.append("low confidence")
            if volume is not None and volume < 50:
                flags.append("thin volume")

            # My YES / My NO render in default white. Row-level dimming
            # via row-suspect handles "this strike isn't a buy" — once
            # the row is actionable (white), BOTH probabilities render
            # at full opacity so the user can read the model's view of
            # each side cleanly.
            ev_yes_v = v.get("_ev_yes")
            ev_no_v = v.get("_ev_no")
            bot_verdict_pre = v.get("bot_verdict", "SKIP")
            my_yes_cls = ""
            my_no_cls = ""

            # ── Verdict — two states only ──────────────────────────────────
            # Rules:
            #   HOLDING YES / HOLDING NO — bot has an open position on this
            #     strike. Wins over the model's current view so the row
            #     reflects what was actually done, not a contradictory
            #     fresh recommendation. Critical for consistency with the
            #     "Active bet" table above — without this, a row we bought
            #     YES on can show a different state once the market moves.
            #   SKIP — every other row. The model's recommendation (BUY
            #     YES / BUY NO / hold off / blocked-by-gate) shows up in
            #     the Edge / EV / tooltip columns; the Verdict column
            #     itself just reports "have we taken this position or
            #     not". The prior BUY YES / BUY NO / WATCH verdicts were
            #     retired per user request to keep the column to two
            #     stable states.
            # Row highlighting on Model-vs-market keys off REAL Kalshi
            # portfolio state — not the paper sim_state. A row only
            # gets the HOLDING badge + .row-bought styling if the
            # ticker is currently held on the account. Paper positions
            # on the sim dashboard still surface in the Active-bets
            # section above; they just don't paint the model-vs-market
            # row as though it were a real trade.
            held_bet = kalshi_held_by_ticker.get(ticker)
            is_bought = held_bet is not None
            bought_side = ((held_bet.get("side") or "").upper()
                           if held_bet else "")
            bot_verdict = v.get("bot_verdict", "SKIP")
            reason = v.get("rejection_reason") or ""
            best_ev_v = v.get("_best_ev")
            best_side_v = v.get("_best_side")
            tt = f" title='{html.escape(reason)}'" if reason else ""
            if is_bought and bought_side in ("YES", "NO"):
                # HOLDING badge keeps its YES/NO colouring (the badge pill
                # tints itself — not the surrounding row, which now reads
                # in plain white). Tooltip surfaces entry price + the
                # model's current take so the user can audit "is the
                # model still on board with this position?"
                #
                # Badge reads "HOLDING YES" or "HOLDING NO" (user
                # 2026-07-11); the held team's name still surfaces in
                # the tooltip so the row is unambiguous. The team-name
                # variant that used to render inside the badge was
                # dropped because the Side cell already names the
                # player and the badge was echoing the same string.
                _held_tk = (held_bet.get("ticker") or "")
                _held_team = ""
                if _held_tk == (v.get("_yes_ticker") or ""):
                    _held_team = v.get("_yes_label") or ""
                elif _held_tk == (v.get("_no_ticker") or ""):
                    _held_team = v.get("_no_label") or ""
                held_cls = "badge-yes" if bought_side == "YES" else "badge-no"
                entry_c = held_bet.get("entry_price_cents")
                entry_part = f" @ {entry_c}c" if entry_c is not None else ""
                team_part = (f" on {_held_team}" if _held_team else "")
                model_part = ""
                if best_ev_v is not None and best_side_v in ("YES", "NO"):
                    _ev_sign = "+" if best_ev_v > 0 else "−"
                    model_part = (f" · model now: {best_side_v} "
                                  f"(EV {_ev_sign}${abs(best_ev_v):.2f})")
                held_tt = (f"You are holding {bought_side}{team_part}"
                           f"{entry_part}{model_part}")
                badge = (f"<span class='badge {held_cls}' "
                         f"title='{html.escape(held_tt)}'>"
                         f"HOLDING {bought_side}</span>")
            else:
                # Tooltip carries the model's recommendation when there
                # is one, so the user can still see "model would buy YES,
                # EV $0.05" on hover even though the cell says SKIP.
                skip_tt = reason
                if best_ev_v is not None and best_side_v in ("YES", "NO"):
                    _ev_sign = "+" if best_ev_v > 0 else "−"
                    rec = (f"model favours {best_side_v} "
                           f"(EV {_ev_sign}${abs(best_ev_v):.2f})")
                    skip_tt = (f"{rec} · {reason}" if reason else rec)
                tt_attr = (f" title='{html.escape(skip_tt)}'"
                           if skip_tt else "")
                badge = f"<span class='badge badge-skip'{tt_attr}>SKIP</span>"
            # A row is a "good buy opportunity" when the bot would actually
            # take a position on it: BUY_YES/BUY_NO verdict + positive EV
            # + no validator flags. Rows that don't clear all three get
            # greyed out so the user sees only actionable rows in colour.
            is_buyable = (
                bot_verdict_pre in ("BUY_YES", "BUY_NO")
                and best_ev_v is not None and best_ev_v > 0
                and not flags
            )
            classes: List[str] = []
            title_attr = ""
            if is_bought:
                classes.append("row-bought")
                classes.append("bought-yes" if bought_side == "YES"
                               else "bought-no" if bought_side == "NO"
                               else "")
                entry_c = held_bet.get("entry_price_cents")
                contracts = held_bet.get("contracts")
                tip_parts = ["You are holding this strike"]
                if bought_side:
                    tip_parts.append(f"on {bought_side}")
                if contracts is not None:
                    tip_parts.append(f"({contracts} contracts")
                    if entry_c is not None:
                        tip_parts.append(f"@ {entry_c}c)")
                    else:
                        tip_parts[-1] = tip_parts[-1] + ")"
                elif entry_c is not None:
                    tip_parts.append(f"(entry {entry_c}c)")
                title_attr = (" title='"
                              + html.escape(" ".join(tip_parts)) + "'")
            else:
                # Per user request: only HOLDING rows get full-bright
                # white text. Every other row — buyable or not — renders
                # dimmed via .row-suspect so the holdings stand out at
                # a glance against the rest of the watchlist.
                classes.append("row-suspect")
                if flags:
                    reason = "Validator flags: " + ", ".join(flags)
                elif best_ev_v is None or best_ev_v <= 0:
                    reason = "No positive edge"
                elif not is_buyable:
                    reason = "Bot verdict not actionable"
                elif best_side_v in ("YES", "NO"):
                    _ev_sign = "+" if (best_ev_v or 0) > 0 else "−"
                    reason = (f"Model favours {best_side_v} "
                                f"(EV {_ev_sign}${abs(best_ev_v or 0):.2f}) — "
                                f"no position held")
                else:
                    reason = "No position held"
                title_attr = (" title='" + html.escape(reason) + "'")
            row_cls = (f" class='{' '.join(classes)}'" if classes else "") + title_attr

            # Pre-format EV cells. Zero or missing values render as a plain
            # "0" instead of the signed "+$0.00" or "—" dash — both convey
            # the same thing ("no actionable edge") and "0" reads cleaner
            # across a dense table.
            def _ev_cell(ev: float | None) -> tuple[str, str]:
                if ev is None:
                    return "0", "gray"
                if round(float(ev), 2) == 0:
                    return "0", "gray"
                cls_, _ = _ev_status(ev)
                sign = "+" if ev > 0 else "−"
                return f"{sign}${abs(ev):.2f}", cls_
            ev_yes_str, ev_yes_cls = _ev_cell(ev_yes_v)
            ev_no_str, ev_no_cls = _ev_cell(ev_no_v)

            # Edge cells — reference probability for the side minus Kalshi's
            # ask price for the same side. Positive = the reference sharp
            # book disagrees with Kalshi in that side's favour. Half-spread
            # is NOT subtracted here (that's what the EV column is for);
            # Edge is the raw reference-vs-market gap so the user can read
            # the bot's underlying view independent of liquidity cost.
            #
            # Preference order for the reference prob:
            #   1. Pinnacle devigged (sharp global book) — when the sport
            #      bot ships it. This is the same reference the buy gate
            #      uses, so Edge finally lines up with the verdict column.
            #   2. Bot's own model — the legacy behaviour, kept as the
            #      fallback so bots that don't wire in Pinnacle (NBA / WNBA
            #      / World Cup / non-tennis strike bots) still get a
            #      meaningful Edge from their own model.
            edge_ref = v.get("pinnacle_prob_yes")
            if edge_ref is None:
                edge_ref = p
            def _edge(p: float | None, ask_c: int | None) -> float | None:
                if p is None or ask_c is None:
                    return None
                return float(p) - (int(ask_c) / 100.0)
            edge_yes_v = _edge(edge_ref, ya_c)
            edge_no_v = _edge((1.0 - float(edge_ref)) if edge_ref is not None else None,
                               na_c)
            def _edge_cell(e: float | None) -> tuple[str, str]:
                if e is None:
                    return "0", "gray"
                pp = e * 100.0
                if round(pp) == 0:
                    return "0", "gray"
                cls_ = ("green" if e >= 0.05 else
                        "yellow" if e > 0 else
                        "red" if e <= -0.02 else "gray")
                return f"{pp:+.0f}%", cls_
            edge_yes_str, edge_yes_cls = _edge_cell(edge_yes_v)
            edge_no_str, edge_no_cls = _edge_cell(edge_no_v)

            # data-ticker on the row + data-field on each live cell so the
            # snapshot poller can patch them in place without re-rendering.
            # mtc cell isn't tagged because it doesn't refresh on a 30s
            # cadence (advances naturally with wall clock time).
            tt_esc = html.escape(ticker)
            # Kalshi uses lowercased series tickers in its market URLs. The
            # full market ticker has the form "<SERIES>-<EVENT>-<STRIKE>", so
            # the series is everything before the first hyphen. Linking to
            # the series page lands on the same market group the row is
            # describing; Kalshi resolves it to the active event.
            series_lower = (ticker.split("-", 1)[0] if ticker else "").lower()
            ticker_url = (f"https://kalshi.com/markets/{series_lower}"
                          if series_lower else "")
            ticker_cell = (
                f"<a href='{html.escape(ticker_url)}' target='_blank' "
                f"rel='noopener noreferrer' class='ticker-link'>{tt_esc}</a>"
                if ticker_url else tt_esc
            )
            # The "BOUGHT YES/NO" inline pill was retired — the row's
            # side-colored left bar + colored ticker text already convey
            # the bet at a glance.
            # Pass the row's strike value through ``data-strike`` so the
            # JS row-click hook can draw a horizontal threshold line on the
            # chart at this market's strike level (non-sport bots) or at the
            # ticker's YES ask price (sport bots, where strike isn't a
            # meaningful concept).
            sl = v.get("strike_low")
            sh = v.get("strike_high")
            try:
                strike_attr = f" data-strike='{float(sl):.6f}'" if sl is not None else ""
            except (TypeError, ValueError):
                strike_attr = ""
            try:
                yes_attr = f" data-yes-prob='{int(ya_c) / 100.0:.4f}'" if ya_c is not None else ""
            except (TypeError, ValueError):
                yes_attr = ""
            # Sport bots: Title + Side. Non-sport: Title + Question.
            # ``watchlist_title_use_event`` overrides the per-market title
            # with the event-level one (used by the unemployment bot, where
            # every row of the table is the same Initial-Claims week and
            # the Kalshi event title — "Initial jobless claims for the
            # week ending May 9, 2026" — is what the user wants in the
            # Title column instead of the per-strike "200K" repetition).
            if (display or {}).get("watchlist_title_use_event") and event_title:
                title_text = event_title
            else:
                title_text = v.get("title") or ""
            # Title cell is now the click-through to Kalshi's market page —
            # the Ticker column that used to carry the link has been
            # removed. Same series-prefix URL logic as ``ticker_cell_html``,
            # just wrapping the human-readable title instead of the raw
            # ticker string.
            title_link = ticker_link_html(ticker, title_text)
            if is_sport_bot:
                # Tennis-shape rows pre-fill _yes_label / _no_label with the
                # player names (the ticker doesn't carry a parseable tricode
                # the way KXNBAGAME does). Prefer those when set; fall back
                # to the NBA tricode parser for KXNBAGAME tickers.
                yes_team = v.get("_yes_label") or _side_tricode_from_ticker(
                    ticker, "YES")
                opp_team = v.get("_no_label") or _side_tricode_from_ticker(
                    ticker, "NO")
                # Held-side re-orientation on Active bets: swap the
                # players so the held one renders on top (bold),
                # matching the flipped Model % / Kalshi % / Entry %
                # cells below.
                if flip_active:
                    yes_team, opp_team = opp_team, yes_team
                if yes_team:
                    side_cell = (
                        f"<td><strong>{html.escape(str(yes_team))}</strong>"
                        f"<br><span class='small gray'>vs "
                        f"{html.escape(str(opp_team))}</span></td>"
                    )
                else:
                    side_cell = f"<td>{html.escape(qstr)}</td>"
                middle_cells = (
                    f"<td>{title_link}</td>"
                    f"{side_cell}"
                )
            elif is_billboard_bot:
                artist_text = v.get("_artist") or ""
                song_text = v.get("_song") or v.get("direction") or ""
                middle_cells = (
                    f"<td>{title_link}</td>"
                    f"<td>{html.escape(str(song_text))}</td>"
                    f"<td>{html.escape(str(artist_text))}</td>"
                )
            else:
                middle_cells = (
                    f"<td>{title_link}</td>"
                    f"<td>{html.escape(qstr)}</td>"
                )
            # User-requested layout: YES on top in green, NO on bottom in
            # red — across every side-paired column (My %, Kalshi %, Edge,
            # EV). Replaces the previous horizontal "yes | no" rendering.
            # The side is conveyed by vertical position + colour; we drop
            # the per-value green/yellow/red EV-magnitude tinting since
            # the side colour now dominates the cell.
            def _stacked(yes_val: str, no_val: str,
                           field: str, extra_tt: str = "") -> str:
                return (
                    f"<td class='num cell-stack' "
                    f"data-field='{field}'{extra_tt}>"
                    f"<div class='side-yes green' data-side='yes'>{yes_val}</div>"
                    f"<div class='side-no red' data-side='no'>{no_val}</div>"
                    f"</td>"
                )
            # Kalshi % — the live market ask, YES on top / NO on bottom.
            # (Replaces the old "My % + Kalshi entry % + Current %" trio;
            # only the live price stays visible in the column set. My %
            # and Kalshi entry % moved out per user spec — model_prob_yes
            # still populates on every row for JSON consumers + downstream
            # code, just isn't rendered as a table column.)
            if flip_active:
                kalshi_cell = _stacked(kno_str, kyes_str, "kalshi")
            else:
                kalshi_cell = _stacked(kyes_str, kno_str, "kalshi")
            # "Model %" stacked cell. On Model-vs-market it's Pinnacle-
            # only — showing "—" is the right signal (no sharp reference
            # → no buy signal). On Active bets we already own the
            # position, so the user wants the Model % cell to reflect
            # WHATEVER our forecast is right now (Pinnacle if available,
            # else the bot's internal model). That matches the buy path
            # inside the tennis executor, which itself falls back to
            # ``live_prob_a`` / ``live_prob_b`` when Pinnacle doesn't
            # list the match.
            pinn_p = v.get("pinnacle_prob_yes")
            if is_active and pinn_p is None:
                pinn_p = v.get("model_prob_yes")
            if pinn_p is not None:
                if flip_active:
                    pinn_yes_str = f"{int(round((1-float(pinn_p))*100))}%"
                    pinn_no_str = f"{int(round(float(pinn_p)*100))}%"
                else:
                    pinn_yes_str = f"{int(round(float(pinn_p)*100))}%"
                    pinn_no_str = f"{int(round((1-float(pinn_p))*100))}%"
            else:
                pinn_yes_str = "—"
                pinn_no_str = "—"
            pinnacle_cell = _stacked(pinn_yes_str, pinn_no_str, "pinnacle")
            edge_cell   = _stacked(edge_yes_str, edge_no_str, "edge")
            ev_cell     = _stacked(ev_yes_str, ev_no_str, "ev")

            # Closes-in cell. Prefer the row's own ``minutes_to_close`` when
            # the bot supplies it (Kalshi bots do); fall back to parsing the
            # settlement date out of the ticker so sport bots (tennis) that
            # don't compute a live countdown still get a real value.
            mtc = v.get("minutes_to_close")
            if mtc is None:
                mtc = minutes_to_close_from_ticker(ticker)
            closes_in_cell = (
                f"<td class='num' data-field='closes-in'>"
                f"{time_to_close_str(mtc)}</td>"
            )

            # Position cells. Active bets on sport bots: 2 cells (My
            # contracts + Kalshi total cost — Kalshi entry cost dropped
            # per user 2026-07-10). Every other include_position_cols
            # table keeps the legacy 3-cell layout (My contracts +
            # Kalshi entry cost + Total cost).
            if not include_position_cols:
                position_cells = ""
            elif is_bought and held_bet is not None:
                _entry_c = held_bet.get("entry_price_cents")
                _ctr = held_bet.get("contracts") or 0
                _my_contracts_cell = (
                    f"<td class='num' data-field='my-contracts'>"
                    f"{int(_ctr)}</td>"
                ) if _ctr else "<td class='num' data-field='my-contracts'></td>"
                if _entry_c is not None and _ctr:
                    _base = float(_entry_c) * float(_ctr) / 100.0
                    _fee = kalshi_fee_cents(int(_entry_c), int(_ctr)) / 100.0
                    _entry_cost_cell = (
                        f"<td class='num red' title='"
                        f"{int(_entry_c)}¢ × {int(_ctr)} contracts = "
                        f"${_base:.2f} entry cost (before fees).' "
                        f"data-field='entry-cost'>"
                        f"−${_base:.2f}</td>"
                    )
                    _total_cost_cell = (
                        f"<td class='num red' title='"
                        f"{int(_entry_c)}¢ × {int(_ctr)} contracts + "
                        f"${_fee:.2f} entry fee = ${_base + _fee:.2f} total "
                        f"cash out at open.' data-field='total-cost'>"
                        f"−${_base + _fee:.2f}</td>"
                    )
                else:
                    _entry_cost_cell = "<td class='num red' data-field='entry-cost'></td>"
                    _total_cost_cell = "<td class='num red' data-field='total-cost'></td>"
                if is_active:
                    # Investment cell — potential earnings on top
                    # (green), Kalshi total cost beneath (red),
                    # consolidated per user 2026-07-13.
                    if _entry_c is not None and _ctr:
                        _gain = ((100 - int(_entry_c)) * int(_ctr)
                                 - kalshi_fee_cents(int(_entry_c),
                                                    int(_ctr))) / 100.0
                        _g_sign = "+" if _gain >= 0 else "−"
                        _g_neg = " neg" if _gain < 0 else ""
                        _investment_cell = (
                            f"<td class='num cell-stack' title='"
                            f"Top: potential earnings — $1 × {int(_ctr)} "
                            f"contracts paid at settlement − "
                            f"${float(_entry_c) * int(_ctr) / 100.0:.2f} "
                            f"entry − fee = net if this side wins. "
                            f"Bottom: Kalshi total cost — "
                            f"{int(_entry_c)}¢ × {int(_ctr)} contracts + "
                            f"${_fee:.2f} entry fee = total cash out at "
                            f"open.' data-field='investment'>"
                            f"<div class='inv-earn{_g_neg}'>"
                            f"{_g_sign}${abs(_gain):.2f}</div>"
                            f"<div class='inv-cost'>−${_base + _fee:.2f}"
                            f"</div></td>"
                        )
                    else:
                        _investment_cell = (
                            "<td class='num cell-stack' "
                            "data-field='investment'></td>"
                        )
                    position_cells = _my_contracts_cell + _investment_cell
                else:
                    position_cells = _my_contracts_cell + _entry_cost_cell + _total_cost_cell
            else:
                if is_active:
                    position_cells = (
                        "<td class='num' data-field='my-contracts'></td>"
                        "<td class='num cell-stack' "
                        "data-field='investment'></td>"
                    )
                else:
                    position_cells = (
                        "<td class='num' data-field='my-contracts'></td>"
                        "<td class='num red' data-field='entry-cost'></td>"
                        "<td class='num red' data-field='total-cost'></td>"
                    )

            # Kalshi entry % (Active bets only) — stacked YES / NO
            # implied prob at our entry price, on the yes-axis. Static
            # once opened, so no data-field poller patch is needed
            # (still tagged for consistency with other stacked cells).
            if is_active:
                if held_bet is not None:
                    _ep_c = held_bet.get("entry_price_cents")
                    _bs = (held_bet.get("side") or "").upper()
                    if _ep_c is not None and _bs in ("YES", "NO"):
                        yes_ep_pct = (int(_ep_c) if _bs == "YES"
                                       else 100 - int(_ep_c))
                        no_ep_pct = 100 - yes_ep_pct
                        entry_pct_cell = _stacked(
                            f"{yes_ep_pct}%",
                            f"{no_ep_pct}%",
                            "entry-pct",
                        )
                    else:
                        entry_pct_cell = _stacked("—", "—", "entry-pct")
                else:
                    entry_pct_cell = _stacked("—", "—", "entry-pct")
            else:
                entry_pct_cell = ""

            # Rules cell — an info button carrying the row's Kalshi
            # ``rules_primary`` on ``data-rules``. Blank cell when no
            # rule text was published (some brand-new markets don't
            # cache it yet); the popover JS ignores empty payloads.
            _rules_txt = (v.get("rules_primary") or "").strip()
            if _rules_txt:
                rules_cell = (
                    "<td data-field='rules'>"
                    "<button type='button' class='rules-info-btn' "
                    f"data-rules='{html.escape(_rules_txt)}' "
                    "title='Kalshi resolution rule — click to read' "
                    "aria-label='Kalshi resolution rule'>i</button>"
                    "</td>"
                )
            else:
                rules_cell = "<td data-field='rules'></td>"
            # Date cell — right of Rules on every Model-vs-market /
            # single table. Parsed from the ticker's encoded date
            # (rules text as fallback).
            if _show_date_col:
                _dt = _market_date_label(ticker, v.get("rules_primary"))
                date_cell = (
                    f"<td data-field='date'>{html.escape(_dt)}</td>"
                    if _dt else "<td data-field='date'></td>"
                )
            else:
                date_cell = ""
            # Event cell — sits between Date and the row's identifying
            # cells; only rendered on Model-vs-market. Parsed live from
            # ``rules_primary`` (tennis + basketball templates), with
            # the bot's competition label as the fallback.
            if _show_event_col:
                _ev = _sport_event_label(v.get("rules_primary"),
                                          v.get("title"),
                                          v.get("tournament"))
                event_cell = (
                    f"<td data-field='event'>{html.escape(_ev)}</td>"
                    if _ev else "<td data-field='event'></td>"
                )
            else:
                event_cell = ""
            # Row layout mirrors the header branch above:
            #   Active bets: Rules | Date | Event | title/side
            #                | Verdict | Contracts bought
            #                | Kalshi total cost | Closes in
            #                | Model % | Kalshi entry % | Kalshi live %
            #   Model-vs-market / non-sport single: keep legacy order
            #   (Total contracts | Model % | Kalshi % | Edge | EV
            #    | Closes in | Verdict | position cells).
            if is_active:
                # Verdict column removed on Active bets (user
                # 2026-07-11) — a held row IS the verdict; the HOLDING
                # state still shows via the row highlight. Closes in
                # sits at the far right per user 2026-07-11.
                row_body = (
                    f"{rules_cell}{date_cell}{event_cell}{middle_cells}"
                    f"{position_cells}"
                    f"{pinnacle_cell}"
                    f"{entry_pct_cell}"
                    f"{kalshi_cell}"
                    f"{closes_in_cell}"
                )
            else:
                row_body = (
                    f"{rules_cell}{date_cell}{event_cell}{middle_cells}"
                    f"<td class='num' data-field='oi'>{oi_str}</td>"
                    f"{pinnacle_cell}"
                    f"{kalshi_cell}"
                    f"{edge_cell}"
                    f"{ev_cell}"
                    f"{closes_in_cell}"
                    f"<td data-field='verdict'>{badge}</td>"
                    f"{position_cells}"
                )
            out.append(f"<tr{row_cls} data-ticker='{tt_esc}'{strike_attr}{yes_attr}>"
                       f"{row_body}</tr>")
        out.append("</tbody></table></div>")
        # For sport bots, close the per-table `<div class='section'>`
        # wrapper opened at the top of this loop iteration.
        if is_sport_bot:
            out.append("</div></div>")
    # Append the row-click JS hook once — after every table is
    # emitted. The hook globs both tbodies (`watchlist-tbody` and,
    # on sport bots, `watchlist-tbody-active`) via its query
    # selector list.
    out.append(_WATCHLIST_ROW_CLICK_JS)
    # Info-bubble popover next to the EV column header — small
    # click-toggle popup explaining the EV formula.
    out.append(_EV_INFO_POPOVER_HTML_JS)
    # Per-row Rules popover — click the ``i`` button in the Rules
    # column to see that ticker's Kalshi ``rules_primary`` text.
    out.append(_RULES_INFO_POPOVER_HTML_JS)
    # Non-sport bots also need to close the single section wrapper
    # opened at the top of the function. Sport bots already closed
    # theirs at the end of each section-loop iteration.
    if not is_sport_bot:
        out.append("</div></div>")


# Info-bubble popover next to the EV column header. Rendered once per
# watchlist section — the popover element itself is shared via a
# single instance appended to the document body; each ``.ev-info-btn``
# in the DOM toggles it and repositions to the button's coordinates.
_EV_INFO_POPOVER_HTML_JS = """
<div class="ev-info-popover" id="ev-info-popover" hidden>
  <h5>How EV is calculated</h5>
  <p>Per-<code>$1</code> contract, on the side the bot would buy:</p>
  <div class="ev-info-formula">EV = (P<sub>ref</sub> − P<sub>Kalshi</sub>) − ½ · spread − fee</div>
  <p><span class="gray"><b>P<sub>ref</sub></b> is the reference probability — Pinnacle's
  devigged sharp-book prob when available (tennis), else the bot's
  own model.<br>
  <b>P<sub>Kalshi</sub></b> is the ask price in dollars.<br>
  <b>Spread</b> is the YES-ask minus NO-ask on Kalshi (half taken as
  slippage on the fill).<br>
  <b>Fee</b> is Kalshi's per-contract entry fee: <code>ceil(0.07 · p · (1−p))</code>.</span></p>
  <p class="gray">A positive EV means the sharp reference thinks
  Kalshi is under-pricing this side, and the bot expects to make money
  on average after slippage and fees.</p>
</div>
<script>
(function () {
  const pop = document.getElementById('ev-info-popover');
  if (!pop) return;
  let currentBtn = null;
  function position(btn) {
    const rect = btn.getBoundingClientRect();
    // Prefer to the RIGHT of the button; flip to the LEFT if it would
    // clip the viewport. Vertically anchor to the button's top.
    const popW = pop.offsetWidth || 320;
    const spaceRight = window.innerWidth - rect.right;
    const left = (spaceRight >= popW + 12)
      ? rect.right + 8
      : Math.max(8, rect.left - popW - 8);
    const top = Math.min(
      Math.max(8, rect.top - 4),
      window.innerHeight - pop.offsetHeight - 8
    );
    pop.style.left = left + 'px';
    pop.style.top  = top  + 'px';
  }
  document.addEventListener('click', function (ev) {
    const btn = ev.target.closest('.ev-info-btn');
    if (btn) {
      ev.preventDefault();
      ev.stopPropagation();
      if (currentBtn === btn && !pop.hidden) {
        pop.hidden = true; currentBtn = null; return;
      }
      pop.hidden = false;
      position(btn);
      currentBtn = btn;
      return;
    }
    // Any click OUTSIDE the popover dismisses it.
    if (!pop.hidden && !ev.target.closest('.ev-info-popover')) {
      pop.hidden = true; currentBtn = null;
    }
  });
  document.addEventListener('keydown', function (ev) {
    if (ev.key === 'Escape' && !pop.hidden) {
      pop.hidden = true; currentBtn = null;
    }
  });
  // Reposition on scroll / resize so the popover tracks its anchor.
  window.addEventListener('scroll', function () {
    if (currentBtn && !pop.hidden) position(currentBtn);
  }, true);
  window.addEventListener('resize', function () {
    if (currentBtn && !pop.hidden) position(currentBtn);
  });
})();
</script>
"""


# Per-row Kalshi-rules popover. Each ``.rules-info-btn`` carries the
# row's Kalshi ``rules_primary`` on ``data-rules``; click opens the
# shared popover populated with that text, positioned next to the
# clicked button.
_RULES_INFO_POPOVER_HTML_JS = """
<div class="rules-info-popover" id="rules-info-popover" hidden>
  <h5>Kalshi resolution rule</h5>
  <div class="rules-body" id="rules-info-body"></div>
</div>
<script>
(function () {
  const pop  = document.getElementById('rules-info-popover');
  const body = document.getElementById('rules-info-body');
  if (!pop || !body) return;
  let currentBtn = null;
  function position(btn) {
    const rect = btn.getBoundingClientRect();
    const popW = pop.offsetWidth || 420;
    // Prefer to the RIGHT of the button (Rules column is now the
    // leftmost column, so anchoring to the right keeps the popover in
    // view). Flip to the left if there isn't room.
    const spaceRight = window.innerWidth - rect.right;
    const left = (spaceRight >= popW + 12)
      ? rect.right + 8
      : Math.max(8, rect.left - popW - 8);
    const top = Math.min(
      Math.max(8, rect.top - 4),
      window.innerHeight - pop.offsetHeight - 8
    );
    pop.style.left = Math.max(8, left) + 'px';
    pop.style.top  = top + 'px';
  }
  document.addEventListener('click', function (ev) {
    const btn = ev.target.closest('.rules-info-btn');
    if (btn) {
      ev.preventDefault();
      ev.stopPropagation();
      const text = btn.getAttribute('data-rules') || '';
      if (!text) return;
      if (currentBtn === btn && !pop.hidden) {
        pop.hidden = true; currentBtn = null; return;
      }
      body.textContent = text;
      pop.hidden = false;
      position(btn);
      currentBtn = btn;
      return;
    }
    if (!pop.hidden && !ev.target.closest('.rules-info-popover')) {
      pop.hidden = true; currentBtn = null;
    }
  });
  document.addEventListener('keydown', function (ev) {
    if (ev.key === 'Escape' && !pop.hidden) {
      pop.hidden = true; currentBtn = null;
    }
  });
  window.addEventListener('scroll', function () {
    if (currentBtn && !pop.hidden) position(currentBtn);
  }, true);
  window.addEventListener('resize', function () {
    if (currentBtn && !pop.hidden) position(currentBtn);
  });
})();
</script>
"""


# Vanilla-JS hook for the Kalshi watchlist tables. Each watchlist row
# carries ``data-ticker`` plus (for non-sport bots) ``data-strike`` and
# (sport / NBA) ``data-yes-prob``. On click:
#
#   * Highlight the selected row.
#   * Find the existing chart's SVG and overlay a horizontal dashed
#     line at the row's strike value (non-sport: strike on the
#     underlying-value Y axis) or at the YES ask probability (sport:
#     plotted on a 0..1 secondary axis). The line replaces any prior
#     overlay so each click "moves" the threshold rather than stacking.
#
# The chart-coordinate math is delegated to a per-chart ``data-y-min``
# / ``data-y-max`` pair the chart renderer stamps onto its SVG. When
# the chart isn't tagged, the JS bails quietly so it never breaks the
# page. The hero-chart implementation in ``_render_watchlist_hero``
# emits these attributes alongside the existing polyline.
_WATCHLIST_ROW_CLICK_JS = """
<script>
(function() {
  // Both the strike-ladder table (id='watchlist-tbody') and the per-bot
  // active-bets table (id='wl-active-tbody', stamped only on the
  // watchlist tab via ``chart_link=True``) feed the same hero chart.
  // Listing them together lets clicks in either table draw the same
  // overlay line, with rows in both tables clearing each other's
  // selection (so the user always sees one active selection).
  const tbodies = Array.from(document.querySelectorAll(
    \"tbody[data-chart-link], tbody#watchlist-tbody, tbody#watchlist-tbody-active\"
  ));
  if (!tbodies.length) return;

  function findChart() {
    // Look for the kalshi-history hero chart's SVG. It carries
    // ``data-chart='wl-hero'`` so we can find it without grabbing
    // any stray SVG (the favicon is one). Returns null when the
    // page renders the empty-frame placeholder.
    return document.querySelector(\"svg[data-chart='wl-hero']\");
  }

  function clearOverlay(svg) {
    svg.querySelectorAll('.row-overlay').forEach(n => n.remove());
  }

  function drawOverlay(svg, label, color) {
    // Read the chart's plotted Y range from the SVG's data attrs
    // and draw a horizontal line at the requested data value.
    const rangeAttr = svg.getAttribute('data-y-range');
    if (!rangeAttr) return;
    const [yMin, yMax, yPad, padT, padL, padR] = rangeAttr.split(',').map(parseFloat);
    if (!Number.isFinite(yMin) || !Number.isFinite(yMax) || yMax === yMin) return;
    const value = label.value;
    if (!Number.isFinite(value)) return;
    if (value < yMin || value > yMax) return;
    const w = svg.viewBox && svg.viewBox.baseVal && svg.viewBox.baseVal.width || 760;
    const h = svg.viewBox && svg.viewBox.baseVal && svg.viewBox.baseVal.height || 220;
    const innerH = h - padT - yPad;
    const y = padT + (1 - (value - yMin) / (yMax - yMin)) * innerH;
    const xL = padL, xR = w - padR;

    const ns = 'http://www.w3.org/2000/svg';
    const line = document.createElementNS(ns, 'line');
    line.setAttribute('class', 'row-overlay');
    line.setAttribute('x1', xL); line.setAttribute('x2', xR);
    line.setAttribute('y1', y); line.setAttribute('y2', y);
    line.setAttribute('stroke', color);
    line.setAttribute('stroke-width', '1.5');
    line.setAttribute('stroke-dasharray', '6,4');
    svg.appendChild(line);

    const text = document.createElementNS(ns, 'text');
    text.setAttribute('class', 'row-overlay');
    text.setAttribute('x', xR - 4);
    text.setAttribute('y', y - 4);
    text.setAttribute('fill', color);
    text.setAttribute('text-anchor', 'end');
    text.setAttribute('font-size', '11');
    text.setAttribute('font-weight', '600');
    text.appendChild(document.createTextNode(label.text));
    svg.appendChild(text);
  }

  function setSelected(activeTr) {
    // Clear selection on every chart-linked row across all tables so
    // only one row is highlighted at a time.
    tbodies.forEach(function (tb) {
      tb.querySelectorAll('tr').forEach(function (r) {
        r.classList.toggle('row-selected', r === activeTr);
      });
    });
  }

  function onClick(ev) {
    const tr = ev.target.closest('tr');
    if (!tr) return;
    // The criteria-button has its own modal handler — don't hijack it.
    if (ev.target.closest('.criteria-btn')) return;
    if (!tr.dataset || !tr.dataset.ticker) return;
    setSelected(tr);
    const svg = findChart();
    if (!svg) return;
    clearOverlay(svg);
    // Strike overlay removed per user request. Sport bots can still
    // stamp data-yes-prob (0..1); those rows are no-ops on the
    // probability-axis hero chart since the range check rejects
    // values < yMin or > yMax.
    const yesProb = parseFloat(tr.dataset.yesProb);
    if (Number.isFinite(yesProb)) {
      drawOverlay(svg, {
        value: yesProb,
        text: 'YES ' + (yesProb * 100).toFixed(0) + '%',
      }, '#58a6ff');
    }
  }

  tbodies.forEach(function (tb) { tb.addEventListener('click', onClick); });
})();
</script>
<style>
#watchlist-tbody tr,
#wl-active-tbody tr { cursor: pointer; }
#watchlist-tbody tr.row-selected td,
#wl-active-tbody tr.row-selected td { background: #1f2630 !important; }
#watchlist-tbody tr:hover td,
#wl-active-tbody tr:hover td { background: #1c222b; }
/* Held-position rows share the same neutral grey hover / selected
   tint as every other row — the per-row colouring was retired in
   favour of the HOLDING badge in the Verdict column. */
</style>
"""
