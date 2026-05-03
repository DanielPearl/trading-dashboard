"""Whale-watcher dashboard view.

Different shape than the gas-bot-style dashboard:

  - Source is JSONL, not SQLite. Whale-watcher writes
    `data/signal_tracking.jsonl` (one row per detected whale event,
    accepted or rejected) and `data/orders.jsonl` (one row per entry
    the bot placed).
  - There are no "users" — Kalshi's public API does not expose trader
    identity on trades. The "user" abstraction here is the *whale event*
    itself. We surface two views:
      1. Per-event list of standout signals (sortable).
      2. Pseudo-identity cohorts: bucket events by ticker prefix × size
         × direction and rank cohorts that historically pay.

Reads are cheap (small JSONL files) and best-effort — missing files
render the empty state instead of raising.
"""
from __future__ import annotations

import html
import json
import logging
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

log = logging.getLogger("dashboard.whale")


# --------------------------------------------------------------------------- #
# Data loaders
# --------------------------------------------------------------------------- #

def load_events(path: str | None, limit: int = 1000) -> List[dict]:
    """Read up to `limit` most-recent rows from signal_tracking.jsonl.

    Each row already has all checkpoints captured (signal_tracker only
    flushes a row once every checkpoint resolves). So we can trust
    `checkpoints[-1].favorable_cents` as the +30m signal-quality value.
    """
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        return []
    out: List[dict] = []
    try:
        with p.open() as f:
            # JSONL is append-only; reading the whole file is fine for
            # the kind of volumes we expect (tens of thousands of rows
            # max). If this ever gets hot, switch to seek-from-end.
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return []
    if limit and len(out) > limit:
        out = out[-limit:]
    return out


def load_orders(path: str | None, limit: int = 500) -> List[dict]:
    """Read recent entries from orders.jsonl.

    Schema (from kalshi_whale_bot/main.py::_persist_order):
      {
        "ts": iso8601,
        "signal": {"ticker", "zscore", "direction", "reason"},
        "order":  {"client_order_id", "side", "count",
                   "limit_price_cents", "dry_run"}
      }
    Exits aren't currently logged here — only entries.
    """
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        return []
    out: List[dict] = []
    try:
        with p.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return []
    if limit and len(out) > limit:
        out = out[-limit:]
    return out


# --------------------------------------------------------------------------- #
# Derived metrics
# --------------------------------------------------------------------------- #

def _last_favorable(event: dict) -> Optional[float]:
    """+30m (last) favorable_cents, or None if not captured."""
    ckpts = event.get("checkpoints") or []
    if not ckpts:
        return None
    fav = ckpts[-1].get("favorable_cents")
    return None if fav is None else float(fav)


def _favorable_at(event: dict, idx: int) -> Optional[float]:
    ckpts = event.get("checkpoints") or []
    if idx >= len(ckpts):
        return None
    fav = ckpts[idx].get("favorable_cents")
    return None if fav is None else float(fav)


def _ticker_prefix(ticker: str) -> str:
    """Kalshi tickers look like 'KXNFLGAME-26APR18SEAWAS-SEA'. We bucket
    by the part before the first '-' so events on the same series cluster
    together."""
    if "-" not in ticker:
        return ticker
    return ticker.split("-", 1)[0]


def _size_bucket(notional_cents: int) -> str:
    """Log-bucket bet size into human labels: <$5, $5-$20, $20-$100, $100+."""
    dollars = notional_cents / 100.0
    if dollars < 5:
        return "<$5"
    if dollars < 20:
        return "$5-$20"
    if dollars < 100:
        return "$20-$100"
    if dollars < 500:
        return "$100-$500"
    return "$500+"


def summarize(events: List[dict]) -> dict:
    """Top-line aggregate stats for the summary card row."""
    n = len(events)
    if n == 0:
        return {
            "n_signals": 0, "n_entered": 0, "pct_entered": 0.0,
            "mean_fav_30m": None, "win_rate_30m": None,
            "first_ts": None, "last_ts": None, "verdict": "no signals yet",
        }
    n_entered = sum(1 for e in events if e.get("entered"))
    favs: List[float] = [
        f for f in (_last_favorable(e) for e in events) if f is not None
    ]
    n_with_fav = len(favs)
    mean_fav = (sum(favs) / n_with_fav) if favs else None
    n_pos = sum(1 for f in favs if f > 0)
    win_rate = (n_pos / n_with_fav) if n_with_fav else None
    timestamps = [e.get("signal_ts") for e in events if e.get("signal_ts") is not None]
    first_ts = min(timestamps) if timestamps else None
    last_ts = max(timestamps) if timestamps else None

    # Verdict: too-few / noise / clear-edge. Sample sizes for whale
    # signals are usually small at the start; be honest about it.
    if n < 50:
        verdict = "too few signals to judge"
    elif mean_fav is None:
        verdict = "no checkpoint data"
    elif mean_fav > 1.5 and win_rate and win_rate > 0.55:
        verdict = "looks like an edge"
    elif mean_fav < -1.0 or (win_rate and win_rate < 0.45):
        verdict = "fading edge — review"
    else:
        verdict = "noise — keep collecting"

    return {
        "n_signals": n, "n_entered": n_entered,
        "pct_entered": (n_entered / n) if n else 0.0,
        "mean_fav_30m": mean_fav, "win_rate_30m": win_rate,
        "first_ts": first_ts, "last_ts": last_ts, "verdict": verdict,
    }


def standouts(events: List[dict], sort_by: str = "recent",
              limit: int = 50) -> List[dict]:
    """Rank standout events. Recognised sorts:
      - recent     (default; most recent first)
      - size       (largest notional first)
      - zscore     (most-anomalous size-vs-history first)
      - favorable  (best +30m payoff first)
      - rejected   (only show non-entered, most recent first)
      - entered    (only show entered, most recent first)
    """
    pool = list(events)
    if sort_by == "rejected":
        pool = [e for e in pool if not e.get("entered")]
        pool.sort(key=lambda e: e.get("signal_ts") or 0, reverse=True)
    elif sort_by == "entered":
        pool = [e for e in pool if e.get("entered")]
        pool.sort(key=lambda e: e.get("signal_ts") or 0, reverse=True)
    elif sort_by == "size":
        pool.sort(key=lambda e: e.get("whale_notional_cents") or 0, reverse=True)
    elif sort_by == "zscore":
        pool.sort(key=lambda e: e.get("zscore") or 0, reverse=True)
    elif sort_by == "favorable":
        pool.sort(key=lambda e: _last_favorable(e) if _last_favorable(e) is not None else -1e9,
                  reverse=True)
    else:  # recent
        pool.sort(key=lambda e: e.get("signal_ts") or 0, reverse=True)
    return pool[:limit]


def compute_cohorts(events: List[dict], min_n: int = 3) -> List[dict]:
    """Bucket events by (ticker_prefix, size_bucket, direction) and
    rank by mean favorable@30m. Cohorts with fewer than `min_n`
    samples are filtered out — small N is noise.

    Returns list of dicts:
      { "key": str, "n": int, "n_entered": int, "mean_fav_30m": float,
        "win_rate": float, "median_size_dollars": float,
        "ticker_prefix": str, "size_bucket": str, "direction": str }
    """
    buckets: Dict[Tuple[str, str, str], List[dict]] = defaultdict(list)
    for e in events:
        ticker = e.get("ticker") or ""
        notional = e.get("whale_notional_cents") or 0
        direction = e.get("direction") or "?"
        key = (_ticker_prefix(ticker), _size_bucket(int(notional)), direction)
        buckets[key].append(e)

    out: List[dict] = []
    for (prefix, size, direction), members in buckets.items():
        if len(members) < min_n:
            continue
        favs = [f for f in (_last_favorable(e) for e in members) if f is not None]
        if not favs:
            continue
        mean_fav = sum(favs) / len(favs)
        wins = sum(1 for f in favs if f > 0)
        win_rate = wins / len(favs)
        sizes = sorted(int(e.get("whale_notional_cents") or 0) / 100.0
                       for e in members)
        median = sizes[len(sizes) // 2] if sizes else 0.0
        out.append({
            "key": f"{prefix} · {size} · {direction.upper()}",
            "ticker_prefix": prefix,
            "size_bucket": size,
            "direction": direction,
            "n": len(members),
            "n_entered": sum(1 for e in members if e.get("entered")),
            "mean_fav_30m": mean_fav,
            "win_rate": win_rate,
            "median_size_dollars": median,
        })
    # Rank by mean_fav, but down-weight small N a bit so a 3/3 win rate
    # doesn't outrank a 8/15 cohort with bigger absolute payoff.
    out.sort(key=lambda c: c["mean_fav_30m"] * math.log(c["n"] + 1),
             reverse=True)
    return out


def rejection_histogram(events: List[dict]) -> List[Tuple[str, int]]:
    counts: Counter = Counter()
    for e in events:
        if e.get("entered"):
            continue
        reason = e.get("rejection_reason") or "(no reason recorded)"
        counts[reason] += 1
    return counts.most_common(15)


# --------------------------------------------------------------------------- #
# Render helpers — mirror the standard dashboard's CSS classes so the
# whale page looks at home in the shared shell.
# --------------------------------------------------------------------------- #

def _fmt_dollars(cents: int | float | None) -> str:
    if cents is None:
        return "—"
    return f"${float(cents) / 100:,.2f}"


def _fmt_signed_cents(cents: float | None) -> str:
    if cents is None:
        return "—"
    sign = "+" if cents > 0 else ("−" if cents < 0 else "")
    return f"{sign}{abs(cents):.1f}c"


def _fmt_ts(ts: float | str | None) -> str:
    if ts is None:
        return "—"
    if isinstance(ts, str):
        try:
            t = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except ValueError:
            return ts
    else:
        t = datetime.fromtimestamp(float(ts), tz=timezone.utc)
    return t.strftime("%Y-%m-%d %H:%M:%SZ")


def _fmt_pct(p: float | None) -> str:
    if p is None:
        return "—"
    return f"{p * 100:.1f}%"


def _verdict_class(verdict: str) -> str:
    if verdict.startswith("looks like"):
        return "good"
    if verdict.startswith("fading"):
        return "bad"
    if verdict.startswith("too few") or verdict.startswith("no"):
        return "gray"
    return ""


def render_page(
    *,
    events: List[dict],
    orders: List[dict],
    available_bots: List[dict],
    current_bot_key: str,
    sort_by: str = "recent",
) -> str:
    """Whole HTML page for the whale-watcher view.

    Reuses the standard dashboard's CSS via a <link> import isn't possible
    (CSS is embedded), so we re-import the CSS string at call site.
    """
    # Imported lazily to avoid a circular import at module load time.
    from .dashboard import CSS, _favicon_link, _render_bot_filter

    summary = summarize(events)
    rows = standouts(events, sort_by=sort_by, limit=50)
    cohorts = compute_cohorts(events, min_n=3)
    rej = rejection_histogram(events)

    out: List[str] = []
    out.append("<!doctype html><html><head>")
    out.append("<meta charset='utf-8'>")
    out.append("<meta http-equiv='refresh' content='30'>")
    out.append("<title>Whale Watcher — signal analysis</title>")
    out.append(_favicon_link())
    out.append(f"<style>{CSS}</style>")
    out.append("<style>"
               ".whale-stats { display:grid; grid-template-columns: repeat(4, 1fr);"
               " gap: 14px; margin: 8px 0 22px 0; }"
               ".whale-stats .card { padding: 14px 16px; }"
               ".whale-stats .label { color:#8b949e; font-size:11px; "
               "text-transform:uppercase; letter-spacing:0.06em; }"
               ".whale-stats .value { font-size:22px; font-weight:600; "
               "color:#f0f6fc; margin-top:4px; }"
               ".verdict.good { color:#3fb950; }"
               ".verdict.bad  { color:#f85149; }"
               ".verdict.gray { color:#8b949e; }"
               ".sort-bar { display:flex; gap:8px; margin: 6px 0 14px 0;"
               " flex-wrap: wrap; }"
               ".sort-pill { padding: 4px 10px; border:1px solid #30363d;"
               " border-radius: 999px; color:#8b949e; text-decoration:none;"
               " font-size: 12px; }"
               ".sort-pill.active { background:#1f6feb22; color:#58a6ff;"
               " border-color:#1f6feb55; }"
               ".dim { color: #8b949e; }"
               ".side-yes { color:#3fb950; font-weight:600; }"
               ".side-no  { color:#f85149; font-weight:600; }"
               ".pos { color:#3fb950; }"
               ".neg { color:#f85149; }"
               ".bench { padding: 2px 6px; background:#21262d; border-radius:4px;"
               " font-family: ui-monospace, SFMono-Regular, Consolas, monospace;"
               " font-size: 11px; }"
               "</style>")
    out.append("</head><body>")
    out.append("<h1>Whale Watcher — signal analysis</h1>")
    out.append("<div class='meta'>"
               f"Loaded {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%SZ')}"
               " · auto-refresh every 30s · "
               "Kalshi public API does not expose trader identity, so each "
               "row below is one anonymous large trade flagged by the bot's "
               "rolling z-score detector."
               "</div>")

    _render_bot_filter(out, available_bots, current_bot_key)

    # ------- Aggregate stats card row -------
    _render_summary_cards(out, summary)

    if summary["n_signals"] == 0:
        _render_empty_state(out)
        out.append("</body></html>")
        return "".join(out)

    # ------- Standout events table -------
    _render_standouts(out, rows, sort_by)

    # ------- Bot's reactive entries -------
    _render_orders(out, orders)

    # ------- Cohorts (pseudo-identity) -------
    _render_cohorts(out, cohorts)

    # ------- Rejection reason histogram -------
    _render_rejections(out, rej)

    out.append("</body></html>")
    return "".join(out)


def _render_summary_cards(out: List[str], summary: dict) -> None:
    out.append("<div class='whale-stats'>")
    out.append("<div class='card'><div class='label'>Signals captured</div>"
               f"<div class='value'>{summary['n_signals']:,}</div>"
               f"<div class='dim small'>"
               f"{_fmt_ts(summary['first_ts'])} → {_fmt_ts(summary['last_ts'])}"
               "</div></div>")
    out.append("<div class='card'><div class='label'>Bot entered</div>"
               f"<div class='value'>{summary['n_entered']:,} "
               f"<span class='dim small'>({_fmt_pct(summary['pct_entered'])})</span>"
               "</div><div class='dim small'>signals that survived all gates</div></div>")
    fav = summary["mean_fav_30m"]
    fav_cls = "pos" if (fav is not None and fav > 0) else ("neg" if fav is not None and fav < 0 else "")
    out.append("<div class='card'><div class='label'>Mean favorable @ +30m</div>"
               f"<div class='value {fav_cls}'>{_fmt_signed_cents(fav)}</div>"
               "<div class='dim small'>direction-signed mid drift after the whale hit</div>"
               "</div>")
    wr = summary["win_rate_30m"]
    out.append("<div class='card'><div class='label'>Signal win rate @ +30m</div>"
               f"<div class='value'>{_fmt_pct(wr)}</div>"
               f"<div class='dim small verdict {_verdict_class(summary['verdict'])}'>"
               f"{html.escape(summary['verdict'])}</div></div>")
    out.append("</div>")


def _render_empty_state(out: List[str]) -> None:
    out.append("<div class='section'><h2>No whale events captured yet</h2>"
               "<div class='body'><div class='empty'>"
               "Start the whale-watcher bot (<code>systemctl start "
               "kalshi-whale-bot</code>) and let it run for a few hours. "
               "Detected whale events are flushed to "
               "<code>data/signal_tracking.jsonl</code> only after every "
               "checkpoint resolves (default +1m / +5m / +15m / +30m), "
               "so the first row takes ~30 minutes to appear."
               "</div></div></div>")


def _render_standouts(out: List[str], rows: List[dict], sort_by: str) -> None:
    sorts = [
        ("recent", "Recent"),
        ("size", "Biggest"),
        ("zscore", "Most anomalous"),
        ("favorable", "Best payoff"),
        ("entered", "Bot entered"),
        ("rejected", "Bot rejected"),
    ]
    out.append("<div class='section'>")
    out.append("<h2>Standout whale events</h2>")
    out.append("<div class='body'>")
    out.append("<div class='sort-bar'>")
    for key, label in sorts:
        cls = "sort-pill active" if key == sort_by else "sort-pill"
        out.append(f"<a class='{cls}' "
                   f"href='?bot=whale-watcher&sort={key}'>{label}</a>")
    out.append("</div>")
    if not rows:
        out.append("<div class='empty'>No matches.</div></div></div>")
        return

    out.append("<table>")
    out.append("<thead><tr>"
               "<th>When</th><th>Ticker</th><th>Side</th>"
               "<th class='right'>Size</th>"
               "<th class='right'>Z</th>"
               "<th class='right'>Dir conf</th>"
               "<th>Bot</th>"
               "<th class='right'>Fav +1m</th>"
               "<th class='right'>+5m</th>"
               "<th class='right'>+15m</th>"
               "<th class='right'>+30m</th>"
               "</tr></thead><tbody>")
    for e in rows:
        side = (e.get("direction") or "?").lower()
        side_html = (f"<span class='side-{side}'>{side.upper()}</span>"
                     if side in ("yes", "no") else html.escape(side))
        if e.get("entered"):
            bot_cell = "<span class='pos'>ENTERED</span>"
        else:
            reason = html.escape(e.get("rejection_reason") or "—")
            bot_cell = f"<span class='dim' title='{reason}'>skip · {reason}</span>"
        f1 = _favorable_at(e, 0)
        f5 = _favorable_at(e, 1)
        f15 = _favorable_at(e, 2)
        f30 = _favorable_at(e, 3)
        out.append(
            "<tr>"
            f"<td class='dim small'>{_fmt_ts(e.get('signal_ts'))}</td>"
            f"<td><span class='bench'>{html.escape(e.get('ticker') or '—')}</span></td>"
            f"<td>{side_html}</td>"
            f"<td class='right'>{_fmt_dollars(e.get('whale_notional_cents'))}</td>"
            f"<td class='right'>{(e.get('zscore') or 0):.1f}</td>"
            f"<td class='right'>{(e.get('direction_confidence') or 0):.2f}</td>"
            f"<td>{bot_cell}</td>"
            f"<td class='right {_pos_neg(f1)}'>{_fmt_signed_cents(f1)}</td>"
            f"<td class='right {_pos_neg(f5)}'>{_fmt_signed_cents(f5)}</td>"
            f"<td class='right {_pos_neg(f15)}'>{_fmt_signed_cents(f15)}</td>"
            f"<td class='right {_pos_neg(f30)}'>{_fmt_signed_cents(f30)}</td>"
            "</tr>"
        )
    out.append("</tbody></table>")
    out.append("</div></div>")


def _pos_neg(v: Optional[float]) -> str:
    if v is None:
        return ""
    return "pos" if v > 0 else ("neg" if v < 0 else "")


def _render_orders(out: List[str], orders: List[dict]) -> None:
    out.append("<div class='section'><h2>Bot's reactive entries</h2>"
               "<div class='body'>")
    if not orders:
        out.append("<div class='empty'>The bot hasn't placed any orders. "
                   "<code>data/orders.jsonl</code> is missing or empty.</div>"
                   "</div></div>")
        return
    # Most recent first.
    rows = sorted(orders, key=lambda o: o.get("ts") or "", reverse=True)[:30]
    out.append("<table>")
    out.append("<thead><tr>"
               "<th>When</th><th>Ticker</th><th>Side</th>"
               "<th class='right'>Count</th>"
               "<th class='right'>Limit</th>"
               "<th class='right'>Notional</th>"
               "<th class='right'>Z</th>"
               "<th>Reason</th>"
               "<th>Mode</th>"
               "</tr></thead><tbody>")
    for o in rows:
        order = o.get("order") or {}
        sig = o.get("signal") or {}
        side = (order.get("side") or "?").lower()
        side_html = (f"<span class='side-{side}'>{side.upper()}</span>"
                     if side in ("yes", "no") else html.escape(side))
        count = int(order.get("count") or 0)
        limit_c = int(order.get("limit_price_cents") or 0)
        notional = count * limit_c
        mode = "DRY" if order.get("dry_run") else "LIVE"
        mode_cls = "dim" if order.get("dry_run") else "pos"
        out.append(
            "<tr>"
            f"<td class='dim small'>{_fmt_ts(o.get('ts'))}</td>"
            f"<td><span class='bench'>{html.escape(sig.get('ticker') or '—')}</span></td>"
            f"<td>{side_html}</td>"
            f"<td class='right'>{count}</td>"
            f"<td class='right'>{limit_c}c</td>"
            f"<td class='right'>{_fmt_dollars(notional)}</td>"
            f"<td class='right'>{(sig.get('zscore') or 0):.1f}</td>"
            f"<td class='dim small'>{html.escape(sig.get('reason') or '—')}</td>"
            f"<td class='{mode_cls}'>{mode}</td>"
            "</tr>"
        )
    out.append("</tbody></table>")
    out.append("<div class='dim small' style='margin-top:8px'>"
               "Realized P&amp;L is not yet plumbed: the whale bot's "
               "position manager logs exits to journalctl but doesn't "
               "append them to <code>orders.jsonl</code>. Add an exit "
               "writer in <code>main.py</code> to surface fills here."
               "</div>")
    out.append("</div></div>")


def _render_cohorts(out: List[str], cohorts: List[dict]) -> None:
    out.append("<div class='section'><h2>Cohorts &mdash; pseudo-identity by pattern</h2>")
    out.append("<div class='body'>")
    out.append("<div class='dim small' style='margin-bottom:10px'>"
               "Bucket every whale event by ticker prefix &times; size band &times; "
               "direction, then rank by mean favorable-cents at +30m, log-weighted "
               "by sample size. Cohorts with N&lt;3 are filtered out as noise. "
               "If the same anonymous trader keeps hitting the same kind of market "
               "with the same kind of size, their pattern bubbles to the top here."
               "</div>")
    if not cohorts:
        out.append("<div class='empty'>No cohort has accumulated 3+ samples yet.</div>"
                   "</div></div>")
        return
    out.append("<table>")
    out.append("<thead><tr>"
               "<th>Cohort</th>"
               "<th class='right'>N</th>"
               "<th class='right'>Bot took</th>"
               "<th class='right'>Median size</th>"
               "<th class='right'>Mean fav +30m</th>"
               "<th class='right'>Win rate</th>"
               "</tr></thead><tbody>")
    for c in cohorts[:30]:
        fav = c["mean_fav_30m"]
        out.append(
            "<tr>"
            f"<td><span class='bench'>{html.escape(c['key'])}</span></td>"
            f"<td class='right'>{c['n']}</td>"
            f"<td class='right'>{c['n_entered']}</td>"
            f"<td class='right'>${c['median_size_dollars']:,.0f}</td>"
            f"<td class='right {_pos_neg(fav)}'>{_fmt_signed_cents(fav)}</td>"
            f"<td class='right'>{_fmt_pct(c['win_rate'])}</td>"
            "</tr>"
        )
    out.append("</tbody></table>")
    out.append("</div></div>")


def _render_rejections(out: List[str], hist: List[Tuple[str, int]]) -> None:
    out.append("<div class='section'><h2>Why signals were rejected</h2>"
               "<div class='body'>")
    if not hist:
        out.append("<div class='empty'>No rejections recorded — every whale "
                   "signal cleared every gate. Either the bot hasn't run "
                   "long enough or your gates are very loose.</div>"
                   "</div></div>")
        return
    out.append("<table style='max-width:640px'>")
    out.append("<thead><tr><th>Reason</th><th class='right'>Count</th></tr></thead>"
               "<tbody>")
    for reason, n in hist:
        out.append(f"<tr><td>{html.escape(reason)}</td>"
                   f"<td class='right'>{n}</td></tr>")
    out.append("</tbody></table>")
    out.append("<div class='dim small' style='margin-top:8px'>"
               "Top-of-funnel rejections tell you which validator is "
               "throttling the strategy. If <code>spread_too_wide</code> "
               "dominates, you're chasing illiquid markets; tighten "
               "<code>max_spread_cents</code> or add a market_profile "
               "override."
               "</div>")
    out.append("</div></div>")
