"""Shared chrome + formatters for the Bitcoin pages.

Tab navigation, page header, and small helpers (cents → dollars, etc.)
live here so ``bitcoin_watchlist`` and ``bitcoin_performance`` can stay
focused on their own table renderers.
"""
from __future__ import annotations

import html
from datetime import datetime, timezone
from typing import Iterable, List, Optional, Tuple

BITCOIN_TABS: List[Tuple[str, str]] = [
    ("home",        "Home"),
    ("watchlist",   "Bitcoin Watchlist"),
    ("performance", "Bitcoin Performance"),
    ("history",     "History"),
]

_EXTRA_CSS = (
    ".pill { display:inline-block; padding:2px 8px; "
    "border-radius:10px; font-size:11px; font-weight:600; "
    "font-variant-numeric: tabular-nums; }"
    ".pill.green { background: rgba(63,185,80,0.18); "
    "color:#3fb950; border:1px solid rgba(63,185,80,0.35); }"
    ".pill.red { background: rgba(248,81,73,0.15); "
    "color:#f85149; border:1px solid rgba(248,81,73,0.30); }"
    ".pill.yellow { background: rgba(212,153,0,0.18); "
    "color:#d49900; border:1px solid rgba(212,153,0,0.35); }"
    ".pill.gray { background: rgba(139,148,158,0.15); "
    "color:#8b949e; border:1px solid rgba(139,148,158,0.30); }"
    ".side-yes { color:#3fb950; font-weight:600; }"
    ".side-no  { color:#f85149; font-weight:600; }"
    ".pos { color:#3fb950; }"
    ".neg { color:#f85149; }"
    ".gray { color:#8b949e; }"
    ".muted { color:#8b949e; }"
    ".btc-question { max-width: 320px; overflow: hidden; "
    "text-overflow: ellipsis; white-space: nowrap; color: #c9d1d9; }"
)


_BOT_SELECT_NAVIGATE_JS = """<script>
(function () {
  const sel = document.getElementById("bot-select-top");
  if (!sel) return;
  sel.addEventListener("change", function () {
    if (sel.value) window.location.href = sel.value;
  });
})();
</script>"""


def render_chrome(out: List[str], title: str,
                  active_tab: str, available_bots: List[dict],
                  current_bot_key: str) -> None:
    """Page header + tab navigation + bot selector. Modifies ``out``."""
    from ..dashboard import CSS, _favicon_link, _render_bot_filter
    out.append("<!doctype html><html><head>")
    out.append("<meta charset='utf-8'>")
    out.append("<meta http-equiv='refresh' content='30'>")
    out.append(f"<title>{html.escape(title)} — Kalshi simulation dashboard</title>")
    out.append(_favicon_link())
    out.append(f"<style>{CSS}</style>")
    out.append(f"<style>{_EXTRA_CSS}</style>")
    out.append("</head><body>")
    out.append("<h1>Kalshi simulation dashboard</h1>")
    out.append(
        "<div class='meta'>Loaded "
        f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%SZ')}"
        " · refreshes every 30s · PAPER-TRADE mode (no real orders)</div>"
    )

    bot_qs = html.escape(current_bot_key)
    main_tabs = [
        ("home",        "Home",                "?tab=home"),
        ("watchlist",   "Bitcoin Watchlist",   f"?tab=watchlist&bot={bot_qs}"),
        ("performance", "Bitcoin Performance", f"?tab=performance&bot={bot_qs}"),
        ("history",     "History",             "?tab=history"),
    ]
    _render_bot_filter(out, available_bots, current_bot_key,
                       select_id="bot-select-top",
                       include_all_option=True,
                       tab_key=active_tab if active_tab != "performance"
                       else "watchlist")
    out.append("<div class='tab-bar'>")
    for key, label, href in main_tabs:
        cls = "tab-pill" + (" tab-pill-active" if key == active_tab else "")
        out.append(f"<a class='{cls}' href='{href}'>{html.escape(label)}</a>")
    out.append("</div>")


def render_footer(out: List[str]) -> None:
    out.append(_BOT_SELECT_NAVIGATE_JS)
    out.append("</body></html>")


# ----------------------------------------------------------------------- #
# Formatters
# ----------------------------------------------------------------------- #

def fmt_cents(v) -> str:
    if v is None:
        return "—"
    try:
        return f"{int(round(float(v)))}c"
    except (TypeError, ValueError):
        return "—"


def fmt_signed_dollars(cents) -> str:
    if cents is None:
        return "—"
    try:
        d = float(cents) / 100.0
    except (TypeError, ValueError):
        return "—"
    sign = "+" if d >= 0 else "−"
    return f"{sign}${abs(d):,.2f}"


def fmt_pct(p, decimals: int = 0) -> str:
    if p is None:
        return "—"
    try:
        return f"{float(p) * 100:.{decimals}f}%"
    except (TypeError, ValueError):
        return "—"


def fmt_btc(price) -> str:
    if price is None:
        return "—"
    try:
        return f"${float(price):,.0f}"
    except (TypeError, ValueError):
        return "—"


def fmt_minutes(mins) -> str:
    if mins is None:
        return "—"
    try:
        m = float(mins)
    except (TypeError, ValueError):
        return "—"
    if m < 60:
        return f"{m:.0f}m"
    if m < 60 * 24:
        return f"{m/60:.1f}h"
    return f"{m/(60*24):.1f}d"


def fmt_ts(ts) -> str:
    if not ts:
        return "—"
    s = str(ts)
    # ISO-8601 truncated to seconds — easier to scan in a table.
    return s.replace("T", " ").split(".")[0]


def fmt_distance(d) -> str:
    if d is None:
        return "—"
    try:
        v = float(d)
    except (TypeError, ValueError):
        return "—"
    sign = "+" if v >= 0 else "−"
    return f"{sign}${abs(v):,.0f}"


def signal_pill(signal: Optional[str]) -> str:
    if not signal:
        return "<span class='pill gray'>—</span>"
    s = str(signal).upper()
    cls = {
        "BUY_YES": "green",
        "BUY_NO": "red",
        "HOLD": "yellow",
        "AVOID": "gray",
    }.get(s, "gray")
    return f"<span class='pill {cls}'>{html.escape(s)}</span>"


def pnl_class(cents) -> str:
    if cents is None:
        return ""
    try:
        v = float(cents)
    except (TypeError, ValueError):
        return ""
    if v > 0:
        return "pos"
    if v < 0:
        return "neg"
    return ""


def empty_state_card(out: List[str], message: str) -> None:
    out.append("<div class='section'><div class='body'>")
    out.append(f"<div class='empty'>{html.escape(message)}</div>")
    out.append("</div></div>")
