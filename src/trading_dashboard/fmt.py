"""Formatting + small computation helpers shared by every renderer."""
from __future__ import annotations

import html
import json
import math
import re
import time
from datetime import datetime
from datetime import timezone
from typing import List
from typing import Tuple


# --------------------------------------------------------------------------- #
# Computations
# --------------------------------------------------------------------------- #

def unrealized_pnl_cents(pos: dict) -> int | None:
    """For a YES position, mark = current yes ask (what we could resell at).
    For a NO position, mark = current no ask. Returns P&L in cents.
    """
    entry = int(pos["entry_price_cents"])
    contracts = int(pos["contracts"])
    side = (pos["side"] or "").upper()
    if side == "YES":
        mark = pos.get("mark_yes_ask")
    else:
        mark = pos.get("mark_no_ask")
    if mark is None:
        return None
    return (int(mark) - entry) * contracts


def _favicon_link() -> str:
    """Return a `<link rel="icon">` tag with an inline SVG data URI.

    The SVG is a stylized chameleon mark in the dashboard's orange +
    teal palette — same shape & colours as static/favicon.svg. Inline
    so the dashboard's BaseHTTPRequestHandler doesn't need a separate
    file-serving route. To swap for a different icon, edit this
    helper or replace static/favicon.svg and copy its contents here.
    """
    # `#` MUST be %23-escaped in data URIs (otherwise it's parsed as a
    # fragment marker). Spaces + < > render fine in modern browsers.
    # Keep this in sync with static/favicon.svg.
    svg = (
        "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'>"
        # Orange teardrop body with a small horn at the top.
        "<path d='M 28 4 L 25 1 L 23 6 C 12 9 4 20 4 32 C 4 46 "
        "14 56 30 56 C 42 56 50 50 52 42 C 56 28 50 12 36 6 "
        "C 33 5 30 4 28 4 Z' fill='%23F5A623'/>"
        # Teal eye dot.
        "<circle cx='46' cy='22' r='5' fill='%231F8B8B'/>"
        # Teal stroked spiral tail (curls back into itself, with
        # the round-cap thickness reading as a solid fill at
        # favicon resolution).
        "<path d='M 30 32 C 14 36 12 56 32 60 C 52 62 60 46 54 34 "
        "C 48 24 34 28 34 42 C 34 50 44 52 46 44' fill='none' "
        "stroke='%231F8B8B' stroke-width='8' stroke-linecap='round'/>"
        "</svg>"
    )
    return (f'<link rel="icon" type="image/svg+xml" '
            f'href="data:image/svg+xml,{svg}"/>')


def kalshi_fee_cents(price_cents: int | None,
                       contracts: int | None) -> int:
    """Kalshi trading fee per their published formula:

        fee = ceil(0.07 × contracts × price × (1 − price))

    where ``price`` is in dollars and the fee is also in dollars.
    Equivalent in cents: ``ceil(0.07 × contracts × p × (100 − p) /
    100)`` where p is the integer-cents price.

    Charged on entry AND on exit (per side). At settlement (price
    is 0¢ or 100¢) the fee is zero — no risk left to fee.

    Returns 0 cents when inputs are missing or out-of-range so the
    caller can safely add this to any cost calculation.
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
    raw = 0.07 * n * p * (100 - p) / 100.0
    return int(math.ceil(raw))


def fmt_signed_cents(c: int | None) -> str:
    if c is None:
        return "—"
    sign = "+" if c >= 0 else "−"
    return f"{sign}${abs(c)/100:.2f}"


def cents_or_dash(c: int | None) -> str:
    return f"{c}c" if c is not None else "—"


def _empty_chart_frame(width: int = 760, height: int = 220,
                        contract_open_ts: float | None = None,
                        contract_close_ts: float | None = None) -> str:
    """Empty-state chart: just the frame (gridlines + day ticks if a
    contract span is known), no polyline. Used when fewer than 2
    snapshots have been recorded — the user wants to see the chart's
    silhouette even when there's nothing to plot yet.
    """
    pad_l, pad_r, pad_t, pad_b = 12, 64, 14, 30
    inner_w = width - pad_l - pad_r
    out: List[str] = [
        f"<div class='wl-chart-wrap'>"
        f"<svg width='100%' height='{height}' viewBox='0 0 {width} {height}' "
        f"preserveAspectRatio='none' style='display:block'>"
    ]
    # Horizontal gridlines removed — the only horizontal line on the
    # chart is the dashed Entry reference (drawn elsewhere when an
    # active bet sets a strike side).
    # Day ticks across [contract_open, now] — chart always ends at
    # the current date, never extends into the future even when the
    # contract isn't closed yet. (contract_close_ts is intentionally
    # ignored here for that reason.)
    if contract_open_ts is not None:
        from datetime import timedelta
        t_min = float(contract_open_ts)
        t_max = max(t_min, time.time())
        dt_min = datetime.fromtimestamp(t_min, tz=timezone.utc)
        dt_max = datetime.fromtimestamp(t_max, tz=timezone.utc)
        day_labels: List[str] = [dt_min.strftime("%b %-d")]
        cur = dt_min.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        while cur < dt_max:
            lbl = cur.strftime("%b %-d")
            if lbl != day_labels[-1]:
                day_labels.append(lbl)
            cur += timedelta(days=1)
        last_label = dt_max.strftime("%b %-d")
        if last_label != day_labels[-1]:
            day_labels.append(last_label)
        n = max(1, len(day_labels) - 1)
        for i, label in enumerate(day_labels):
            frac = i / n if n else 0.5
            x = pad_l + frac * inner_w
            out.append(f"<line x1='{x:.1f}' y1='{pad_t}' x2='{x:.1f}' "
                       f"y2='{height-pad_b}' stroke='#1f2530' stroke-width='1' "
                       f"stroke-dasharray='2,3' opacity='0.7'/>")
            anchor = "start" if i == 0 else (
                "end" if i == len(day_labels) - 1 else "middle")
            out.append(f"<text x='{x:.0f}' y='{height-10}' fill='#8b949e' "
                       f"font-size='10' text-anchor='{anchor}'>"
                       f"{html.escape(label)}</text>")
    out.append("</svg></div>")
    return "".join(out)


def svg_kalshi_chart(history: List[dict], display: dict,
                      reference_strike: float | None = None,
                      strike_side: str | None = None,
                      strike_is_active_bet: bool = False,
                      contract_open_ts: float | None = None,
                      contract_close_ts: float | None = None,
                      total_volume: int | None = None,
                      y_min: float | None = None,
                      y_max: float | None = None,
                      width: int = 760, height: int = 220) -> str:
    """Underlying-price chart, derived from Kalshi's strike ladder.

    Same visual idiom as Kalshi's market-page chart: one line in the
    underlying's native units (USD/MMBtu, USD/gal, K claims), y-axis
    auto-scaled to the data range, optional horizontal strike line
    colored to indicate winning side. Different from the prior 0..100%
    chance chart — the y-axis here is in real-world units so the user
    sees what Kalshi shows on its own market page.
    """
    pts_in: List[Tuple[float, float]] = []
    for r in history:
        ts = r.get("ts")
        v = r.get("value")
        if ts is None or v is None:
            continue
        try:
            pts_in.append((float(ts), float(v)))
        except (TypeError, ValueError):
            continue
    if len(pts_in) < 2:
        return _empty_chart_frame(width=width, height=height,
                                    contract_open_ts=contract_open_ts,
                                    contract_close_ts=contract_close_ts)

    # Y-axis labels go on the right edge (matches Kalshi's market page),
    # so reserve the right padding instead of the left. Bottom padding
    # leaves room for the date-tick row.
    pad_l, pad_r, pad_t, pad_b = 12, 64, 14, 30
    inner_w = width - pad_l - pad_r
    inner_h = height - pad_t - pad_b
    n = len(pts_in)
    # X-axis spans contract open → NOW. Always ends at current date,
    # never extended into the future even when the contract is still
    # open. Future-dated data points (clock skew etc.) are clipped to
    # now since the user's spec is "end at current date".
    now_ts = time.time()
    t_max = now_ts
    t_min = float(contract_open_ts) if contract_open_ts else pts_in[0][0]
    if t_min > pts_in[0][0]:
        t_min = pts_in[0][0]
    t_span = max(1.0, t_max - t_min)

    # The visible polyline plots the raw recorded values — what the
    # underlying actually was at each Kalshi-recorded tick. No smoothing
    # or bucketing: the line and the hover-tooltip values are the same
    # series, so what you see scrubbing matches what you see on the line.
    pts_plot: List[Tuple[float, float]] = list(pts_in)

    # Y-axis: callers can pin the range (``y_min`` / ``y_max``) when
    # the chart represents a value with inherent bounds — e.g. a
    # probability series capped at 0..100¢. When unpinned, auto-scale
    # to the actual data range with 8% padding (default behaviour).
    if y_min is not None and y_max is not None:
        y_lo = float(y_min)
        y_hi = float(y_max)
    else:
        values = [v for _, v in pts_in]
        if strike_is_active_bet and reference_strike is not None:
            values = values + [float(reference_strike)]
        vmin = min(values)
        vmax = max(values)
        if vmax == vmin:
            pad_v = max(0.001, abs(vmax) * 0.005)
        else:
            pad_v = (vmax - vmin) * 0.08
        y_lo = vmin - pad_v
        y_hi = vmax + pad_v

    # With the strike included in the value set, the dotted line is
    # always in range when there's a bet. The flag is kept for
    # completeness (callers can pass an out-of-range strike for the
    # closest-to-money case, which we now skip).
    strike_in_range = (reference_strike is not None
                       and y_lo <= float(reference_strike) <= y_hi)

    def x_at(t: float) -> float:
        return pad_l + (t - t_min) / t_span * inner_w

    def y_at(v: float) -> float:
        return pad_t + (1.0 - (v - y_lo) / (y_hi - y_lo)) * inner_h

    # Wrap the SVG in a positioning context for the hover tooltip and
    # expose the chart geometry as data attrs so the JS can map the
    # cursor's x position back to a timestamp without re-deriving it.
    # Compact JSON of (ts, raw_value) pairs for the hover tooltip's
    # interpolation. Server-side stores the raw value (pre-divisor /
    # pre-format); the JS formats it client-side.
    points_payload = json.dumps([(int(t), v) for t, v in pts_in],
                                  separators=(",", ":"))
    fmt_payload = json.dumps({
        "divisor": float(display.get("divisor", 1.0) or 1.0),
        "decimals": int(display.get("underlying_decimals", 2)),
        "unit": display.get("underlying_unit", ""),
        "unit_position": display.get("unit_position", "prefix"),
    }, separators=(",", ":"))
    # ``data-y-range`` exposes the chart's plotted Y range + padding to
    # the row-click JS hook so it can draw a horizontal threshold line
    # at the clicked row's strike value. Format:
    #   y_min, y_max, pad_b, pad_t, pad_l, pad_r
    y_range_attr = f"{y_lo:.6f},{y_hi:.6f},{pad_b},{pad_t},{pad_l},{pad_r}"
    out: List[str] = [
        f"<div class='wl-chart-wrap' "
        f"data-tmin='{t_min:.0f}' data-tmax='{t_max:.0f}' "
        f"data-padl='{pad_l}' data-innerw='{inner_w}' "
        f"data-padt='{pad_t}' data-padb='{pad_b}' data-h='{height}' "
        f"data-vbw='{width}' "
        f"data-points='{html.escape(points_payload)}' "
        f"data-fmt='{html.escape(fmt_payload)}'>",
        f"<svg data-chart='wl-hero' data-y-range='{y_range_attr}' "
        f"width='100%' height='{height}' viewBox='0 0 {width} {height}' "
        f"preserveAspectRatio='none' style='display:block'>"
    ]

    # 5 evenly-spaced y-axis labels, no horizontal gridlines — the
    # only horizontal line on the chart is the dashed Entry reference
    # drawn below (when there's an active bet). Keeps the visual
    # weight on the plotted polyline.
    for i in range(5):
        v = y_lo + (i / 4.0) * (y_hi - y_lo)
        y = y_at(v)
        out.append(f"<text x='{width-pad_r+6}' y='{y+4}' fill='#8b949e' "
                   f"font-size='10' text-anchor='start'>"
                   f"{html.escape(fmt_underlying(v, display))}</text>")

    # Color the line green where it sits on the winning side of the
    # strike, white on the losing side. Same logic as the prior
    # underlying chart — strike-relative segment splitting.
    side = (strike_side or "").upper()
    if side == "NO":
        above_color, below_color = "#c9d1d9", "#3fb950"
    else:
        above_color, below_color = "#3fb950", "#c9d1d9"

    if not strike_in_range or reference_strike is None:
        path = " ".join(f"{x_at(t):.1f},{y_at(v):.1f}" for t, v in pts_plot)
        out.append(f"<polyline points='{path}' stroke='#c9d1d9' "
                   f"stroke-width='2' fill='none'/>")
    else:
        strike = float(reference_strike)
        runs: List[Tuple[bool, List[Tuple[float, float]]]] = []
        cur_above = pts_plot[0][1] >= strike
        cur_run: List[Tuple[float, float]] = [(x_at(pts_plot[0][0]), y_at(pts_plot[0][1]))]
        for i in range(1, n):
            t_prev, v_prev = pts_plot[i - 1]
            t_curr, v_curr = pts_plot[i]
            new_above = v_curr >= strike
            if new_above == cur_above:
                cur_run.append((x_at(t_curr), y_at(v_curr)))
                continue
            denom = v_curr - v_prev
            t = (strike - v_prev) / denom if denom != 0 else 0.5
            t = max(0.0, min(1.0, t))
            cross_x = x_at(t_prev) + t * (x_at(t_curr) - x_at(t_prev))
            cross_y = y_at(strike)
            cur_run.append((cross_x, cross_y))
            runs.append((cur_above, cur_run))
            cur_run = [(cross_x, cross_y), (x_at(t_curr), y_at(v_curr))]
            cur_above = new_above
        runs.append((cur_above, cur_run))
        for is_above, run in runs:
            if len(run) < 2:
                continue
            color = above_color if is_above else below_color
            pts_str = " ".join(f"{x:.1f},{y:.1f}" for x, y in run)
            out.append(f"<polyline points='{pts_str}' stroke='{color}' "
                       f"stroke-width='2' fill='none'/>")

    # Horizontal strike line — dotted, drawn ONLY for an active bet.
    # YES position → green dotted line, label reads "Above $X"
    # NO  position → red dotted line, label reads "Below $X"
    # The colour communicates "your winning territory": YES bets win
    # when the underlying ends up above the line (green = win), NO
    # bets win when it stays below (red = the threshold you don't
    # want to be above).
    if strike_is_active_bet and strike_in_range and reference_strike is not None:
        ys = y_at(float(reference_strike))
        is_no = (side == "NO")
        line_color = "#f85149" if is_no else "#3fb950"
        label = "Entry"
        out.append(f"<line x1='{pad_l}' y1='{ys}' x2='{width-pad_r}' y2='{ys}' "
                   f"stroke='{line_color}' stroke-width='1.5' "
                   f"stroke-dasharray='4,4' opacity='0.95'/>")
        label_x = pad_l + inner_w * 0.5
        out.append(f"<text x='{label_x:.0f}' y='{ys-6}' fill='{line_color}' "
                   f"font-size='11' text-anchor='middle' opacity='0.95'>"
                   f"{html.escape(label)}</text>")

    # X-axis: one label per UNIQUE day in the contract span, EVENLY
    # SPACED across the chart's width regardless of where each midnight
    # actually falls in time. Visually decouples label position from
    # the time axis (the polyline still uses real time) so the bottom
    # row stays balanced even on lopsided spans.
    from datetime import timedelta
    dt_min = datetime.fromtimestamp(t_min, tz=timezone.utc)
    dt_max = datetime.fromtimestamp(t_max, tz=timezone.utc)
    day_labels: List[str] = [dt_min.strftime("%b %-d")]
    cur = dt_min.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
    while cur < dt_max:
        lbl = cur.strftime("%b %-d")
        if lbl != day_labels[-1]:
            day_labels.append(lbl)
        cur += timedelta(days=1)
    last_label = dt_max.strftime("%b %-d")
    if last_label != day_labels[-1]:
        day_labels.append(last_label)
    n = max(1, len(day_labels) - 1)
    for i, label in enumerate(day_labels):
        frac = i / n if n else 0.5
        x = pad_l + frac * inner_w
        out.append(f"<line x1='{x:.1f}' y1='{pad_t}' x2='{x:.1f}' "
                   f"y2='{height-pad_b}' stroke='#1f2530' stroke-width='1' "
                   f"stroke-dasharray='2,3' opacity='0.7'/>")
        if i == 0:
            anchor, tx = "start", x
        elif i == len(day_labels) - 1:
            anchor, tx = "end", x
        else:
            anchor, tx = "middle", x
        out.append(f"<text x='{tx:.0f}' y='{height-10}' fill='#8b949e' "
                   f"font-size='10' text-anchor='{anchor}'>"
                   f"{html.escape(label)}</text>")

    # Volume moved to the hero header (top-right under "Closes in")
    # per user request — no longer on the chart frame.

    out.append("</svg>")
    # Hover tooltip — JS in the page polyfills this with a vertical
    # line + "May 1 at 9 AM" label as the cursor moves over the chart.
    out.append("<div class='wl-chart-tooltip' hidden></div>")
    out.append("</div>")
    return "".join(out)


def fmt_underlying(value: float | None, display: dict) -> str:
    """Format an underlying value per the bot's display config:
       prefix → '$2.759';   suffix → '189K';   none → '2.759'.
    Applies `divisor` first so bots that store raw counts (e.g. 189000
    claims) can render in thousands.
    """
    if value is None:
        return "—"
    divisor = float(display.get("divisor", 1.0)) or 1.0
    v = float(value) / divisor
    decimals = int(display.get("underlying_decimals", 2))
    unit = display.get("underlying_unit", "")
    pos = display.get("unit_position", "prefix")
    n = f"{v:,.{decimals}f}"
    if pos == "prefix":
        return f"{unit}{n}"
    if pos == "suffix":
        return f"{n}{unit}"
    return n


def time_left_str(minutes: float | None) -> str:
    """Compact 'closes in 3d 4h' / '12h 30m' / '45m' for the hero header."""
    if minutes is None or minutes <= 0:
        return "—"
    total_min = int(minutes)
    days = total_min // (60 * 24)
    rem = total_min - days * 60 * 24
    hours = rem // 60
    mins = rem - hours * 60
    if days >= 1:
        return f"{days}d {hours}h"
    if hours >= 1:
        return f"{hours}h {mins}m"
    return f"{mins}m"


def question_str(direction: str, low: float | None, high: float | None,
                 display: dict | None = None) -> str:
    """Format the watchlist's Question column. Uses the bot's display
    config when given so unemployment renders "above 175K" instead of
    "above $175000.00".

    When ``display['question_format']`` is set, an alternate idiom
    is used. Supported values:
      * ``"at_least_full"`` — "at least 200,000" (raw value, comma-
        separated, no divisor / unit). Used by the unemployment-claims
        bot to surface the full strike count in plain English instead
        of the "above 200K" shorthand fmt_underlying produces.
    """
    if display and display.get("question_format") == "at_least_full":
        if direction == "between" and low is not None and high is not None:
            return f"{int(round(float(low))):,} – {int(round(float(high))):,}"
        if low is not None and direction in ("above", "greater"):
            return f"at least {int(round(float(low))):,}"
        if low is not None and direction in ("below", "less"):
            return f"below {int(round(float(low))):,}"
        if low is not None:
            return f"{direction} {int(round(float(low))):,}"
        return direction or "—"
    if display:
        if direction == "between" and low is not None and high is not None:
            return f"{fmt_underlying(low, display)} – {fmt_underlying(high, display)}"
        if low is not None:
            return f"{direction} {fmt_underlying(low, display)}"
        return direction or "—"
    # Legacy default — gas-prices-style $/gal formatting.
    if direction == "between" and low is not None and high is not None:
        return f"${low:.2f} – ${high:.2f}"
    if low is not None:
        return f"{direction} ${low:.2f}"
    return direction or "—"


def time_to_close_str(minutes: float | None) -> str:
    """Compact "1.7d / 9.2h / 45m" rendering for the Closes-in cell.

    Negative inputs mean the contract's published close time is
    already in the past — typical for tennis paper bets whose match
    settled days ago but whose simulator hasn't received a settle
    signal yet. We surface those as "settled" + how long ago, so
    the user can see the bet is stuck open rather than just a dash.

    Special case: the ±2-minute window around the close time
    renders as "closing" instead of "0m" / "settled 1m ago".
    Avoids the misleading "0m" reading on rows that are actively
    resolving — the hedge daemon will sweep them shortly via the
    ``settled_auto`` path.
    """
    if minutes is None:
        return "—"
    if -2 < minutes < 2:
        return "closing..."
    if minutes < 0:
        ago = -minutes
        if ago > 1440:
            return f"settled {ago/1440:.0f}d ago"
        if ago > 60:
            return f"settled {ago/60:.0f}h ago"
        return f"settled {int(ago)}m ago"
    if minutes > 1440:
        return f"{minutes/1440:.1f}d"
    if minutes > 60:
        return f"{minutes/60:.1f}h"
    return f"{int(minutes)}m"


# Kalshi tickers encode the settlement date as ``YYMMMDD`` after the
# series prefix. ``KXATPMATCH-26MAY12TIRMED`` → 2026-05-12. Tennis
# sim positions don't carry an explicit expected_expiration_time so
# this regex is the universal fallback for the "Closes in" column.
_TICKER_DATE_RE = re.compile(
    r"-(?P<yy>\d{2})(?P<mon>[A-Z]{3})(?P<dd>\d{2})", re.IGNORECASE,
)
_MONTH_MAP = {m: i + 1 for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
     "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
)}


def minutes_to_close_from_ticker(ticker: str | None,
                                    assumed_close_hour_utc: int = 23,
                                    ) -> float | None:
    """Parse the settlement date out of a Kalshi ticker and return the
    signed minutes from now until that day's close window. Settlement
    happens after the event ends, so we anchor at the LAST hour of
    the encoded date (23:59 UTC by default).

    Positive return = minutes remaining until close.
    Negative return = minutes since the contract already settled.
    None = ticker doesn't match the ``-YYMMMDD`` pattern at all.

    The caller's display ``time_to_close_str`` knows how to format
    negative values as "settled Nd ago" so stuck-open paper positions
    on long-finished matches show meaningful state instead of "—".
    """
    if not ticker:
        return None
    m = _TICKER_DATE_RE.search(ticker)
    if not m:
        return None
    mon = _MONTH_MAP.get(m.group("mon").upper())
    if mon is None:
        return None
    try:
        year = 2000 + int(m.group("yy"))
        day = int(m.group("dd"))
        ts = datetime(year, mon, day, assumed_close_hour_utc, 59,
                       tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None
    return (ts - datetime.now(timezone.utc)).total_seconds() / 60.0


def ticker_cell_html(ticker: str | None) -> str:
    """Render a ticker as a Kalshi market-page link.

    Output mirrors the convention already used in the Watchlist table
    (``class='ticker-link'``): the visible text is the full market
    ticker, but the href targets ``kalshi.com/markets/<series>``
    where series is everything before the first hyphen, lowercased.
    Linking to the series page lands on the same market group the
    row is describing; Kalshi resolves it to whichever event is
    currently live.

    Returns "—" for None / empty input so callers can drop it into a
    `<td>` directly.
    """
    if not ticker:
        return "—"
    tt_esc = html.escape(ticker)
    series_lower = ticker.split("-", 1)[0].lower()
    if not series_lower:
        return tt_esc
    url = f"https://kalshi.com/markets/{series_lower}"
    return (f"<a href='{html.escape(url)}' target='_blank' "
            f"rel='noopener noreferrer' class='ticker-link'>{tt_esc}</a>")


def ticker_link_html(ticker: str | None, text: str) -> str:
    """Wrap arbitrary ``text`` in a Kalshi market-page link built from
    ``ticker``. Used by the Watchlist Title cell so clicking the title
    lands on the same market page ``ticker_cell_html`` would have.
    Falls back to the plain (escaped) text when the ticker can't be
    parsed into a series prefix."""
    text_esc = html.escape(text or "")
    if not ticker:
        return text_esc
    series_lower = ticker.split("-", 1)[0].lower()
    if not series_lower:
        return text_esc
    url = f"https://kalshi.com/markets/{series_lower}"
    return (f"<a href='{html.escape(url)}' target='_blank' "
            f"rel='noopener noreferrer' class='ticker-link'>"
            f"{text_esc}</a>")


def _match_text_from_ticker(ticker: str | None) -> str:
    """Parse the matchup string out of a Kalshi NBA ticker.

    Format: ``KXNBAGAME-{YY}{MMM}{DD}{AWAY}{HOME}-{TEAM}``
    Example: ``KXNBAGAME-26MAY08SASMIN-MIN`` → ``"MIN vs SAS (May 8)"``.

    The game date is appended so playoff series with multiple games
    against the same opponent (e.g. CLE vs NYK Games 2 + 3) don't
    collide visually between active bets and history — two rows
    rendered identically as just "NYK vs CLE" can otherwise look
    like the same ticker appearing in both places.

    Returns ``""`` when the ticker doesn't fit the NBA pattern (gas /
    CPI / jobless tickers); the caller renders a ``—`` placeholder.
    """
    if not ticker:
        return ""
    parts = ticker.split("-")
    if len(parts) < 3 or not parts[0].startswith("KXNBAGAME"):
        return ""
    # Middle chunk: 7-char date prefix (YYMMMDD = e.g. ``26MAY08``)
    # then two 3-char tricodes for away + home.
    body = parts[1]
    if len(body) < 13:
        return ""
    away_tri = body[7:10]
    home_tri = body[10:13]
    if not (away_tri.isalpha() and home_tri.isalpha()):
        return ""
    mmm = body[2:5]
    dd = body[5:7]
    month_pretty = mmm.capitalize() if mmm.isalpha() else mmm
    day_pretty = dd.lstrip("0") or dd
    date_suffix = (f" ({month_pretty} {day_pretty})"
                    if month_pretty and day_pretty.isdigit() else "")
    return f"{home_tri.upper()} vs {away_tri.upper()}{date_suffix}"


def _side_tricode_from_ticker(ticker: str | None, side: str) -> str:
    """Return the team tricode the bet is on. The third hyphen-segment
    of an NBA ticker carries the team for which YES = "this team wins".
    On a NO bet, we want the *other* team. Returns "" for non-NBA tickers.
    """
    if not ticker:
        return ""
    parts = ticker.split("-")
    if len(parts) < 3 or not parts[0].startswith("KXNBAGAME"):
        return ""
    yes_team = parts[2].upper()
    if (side or "").upper() == "YES":
        return yes_team
    # NO side → return the other team from the matchup chunk.
    body = parts[1]
    if len(body) < 13:
        return yes_team
    away_tri = body[7:10].upper()
    home_tri = body[10:13].upper()
    return away_tri if yes_team == home_tri else home_tri

def _ev_status(ev: float | None) -> tuple[str, str]:
    """Return (css_class, label) for an EV value. Drives the red/yellow/
    green pill on every EV-bearing card."""
    if ev is None:
        return "gray", "—"
    if ev >= 0.03:
        return "green", "POSITIVE EV"
    if ev > 0:
        return "yellow", "MARGINAL EV"
    return "red", "NEGATIVE EV"

_TENNIS_EVENT_RE = re.compile(
    r"20\d{2}\s+(.+?)"
    r"\s+(?:Round of \d+|Round\b|Semifinal|Quarterfinal"
    r"|Final|Match|R\d+|QF|SF|Grand Final|First Round"
    r"|Second Round|Third Round|Fourth Round)",
    re.IGNORECASE,
)


def _tennis_event_label(rules: str | None, title: str | None = None) -> str:
    """Return a short 'TOUR · Event' label parsed from Kalshi tennis
    ``rules_primary`` text. Empty string when the rules don't match the
    template (billboard / non-tennis contracts) or the parser can't
    identify a tour.

    Examples:
      "…in the 2026 Wimbledon Men Singles Semifinal…"   → "ATP · Wimbledon"
      "…in the 2026 Wimbledon Women Singles Semifinal…" → "WTA · Wimbledon"
      "…in the 2026 M15 Tokyo Round of 16…"             → "ITF · M15 Tokyo"
      "…in the 2026 ATP Miami Open Round of 32…"        → "ATP · Miami Open"
    """
    if not rules:
        return ""
    m = _TENNIS_EVENT_RE.search(rules)
    if not m:
        return ""
    event = m.group(1).strip()
    low = event.lower()

    # Tour detection — precedence: explicit "Women/Men Singles" >
    # explicit "ATP"/"WTA" prefix > Futures M/W tier prefix > Challenger.
    # "Women Singles" must be checked BEFORE "Men Singles" because the
    # substring "men singles" occurs inside "women singles" and would
    # otherwise misclassify every WTA match as ATP.
    tour = ""
    if "women singles" in low or "women's singles" in low:
        tour = "WTA"
    elif "men singles" in low or "men's singles" in low:
        tour = "ATP"
    elif low.startswith("atp "):
        tour = "ATP"
    elif low.startswith("wta "):
        tour = "WTA"
    elif re.match(r"^m\d+\b", low):
        tour = "ITF"          # M-tier Futures (men)
    elif re.match(r"^w\d+\b", low):
        tour = "ITF"          # W-tier Futures (women)
    elif "challenger" in low:
        tour = "ATP-CH"

    # Strip tour-hint tokens from the visible event name so we don't
    # render "ATP · ATP Miami Open".
    event = re.sub(
        r"\b(Men's Singles|Women's Singles|Men Singles|Women Singles"
        r"|ATP|WTA)\b",
        "",
        event,
        flags=re.IGNORECASE,
    ).strip()
    event = re.sub(r"\s{2,}", " ", event)
    if not event:
        return ""
    return f"{tour} · {event}" if tour else event


# Basketball ``rules_primary`` boilerplate: "If X wins the {A} vs {B}
# women's professional basketball game originally scheduled for
# Jul 8, 2026, then the market resolves to Yes." The league (WNBA vs
# NBA) hides in the "women's/men's" qualifier. The game date is NOT
# part of the label — it gets its own Date column.
_BASKETBALL_EVENT_RE = re.compile(
    r"(women'?s|men'?s)\s+professional\s+basketball\s+game",
    re.IGNORECASE,
)


def _basketball_event_label(rules: str | None) -> str:
    """Return 'WNBA' / 'NBA' parsed from Kalshi basketball rules text;
    empty string when the rules don't match the basketball template."""
    if not rules:
        return ""
    m = _BASKETBALL_EVENT_RE.search(rules)
    if not m:
        return ""
    return "WNBA" if m.group(1).lower().startswith("women") else "NBA"


# Fallback for the Date column: an explicit "Jul 9, 2026"-style date
# inside the rules text (some series omit the day from the ticker,
# e.g. monthly CPI).
_RULES_DATE_RE = re.compile(
    r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?"
    r"\s+(\d{1,2}),\s*(\d{4})",
)


def _market_date_label(ticker: str | None, rules: str | None = None) -> str:
    """Human date ("Jul 9, 2026") for the watchlist Date column.
    Prefers the ticker's encoded YYMMMDD (parsed with the module's
    ``_TICKER_DATE_RE`` — the same regex the "Closes in" column uses);
    falls back to the first explicit date in the rules text; empty
    string when neither carries one. Month token must be a real month
    (via ``_MONTH_MAP``) so strike suffixes can't false-positive."""
    m = _TICKER_DATE_RE.search(ticker or "")
    if m:
        mon_token = m.group("mon").upper()
        if mon_token in _MONTH_MAP:
            return (f"{mon_token.capitalize()} {int(m.group('dd'))}, "
                    f"20{m.group('yy')}")
    if rules:
        m = _RULES_DATE_RE.search(rules)
        if m:
            return f"{m.group(1)} {int(m.group(2))}, {m.group(3)}"
    return ""


def _sport_event_label(rules: str | None, title: str | None,
                        tournament: str | None) -> str:
    """Event-cell label for a sport watchlist row. Tries the known
    rules templates (tennis, then basketball); falls back to the
    bot's own competition label (``tournament`` — e.g. "WNBA") so the
    cell never goes blank just because Kalshi rewords its boilerplate.
    """
    return (_tennis_event_label(rules, title)
            or _basketball_event_label(rules)
            or (tournament or "").strip())
