#!/usr/bin/env python3
"""Post-deploy pane check — catch a filter change blanking a bot.

For every bot in the Contracts dropdown, fetch the rendered watchlist
page AND the /api/snapshot payload, then compare:

  * page_rows      verdict cells rendered in Model-vs-market + Active
  * eligible_rows  snapshot rows that plausibly should render (have a
                   displayable model %, open interest, and a live
                   two-sided-ish quote)

A pane with eligible_rows > 0 and page_rows == 0 is flagged — that is
the signature of every filter regression this dashboard has had
(no-model-% rule vs bots without benchmarks, zero-OI rule vs bots
that don't stamp OI, ticker-date settled rule vs hormuz).

Usage:
    python3 scripts/check_panes.py http://localhost:8080
    python3 scripts/check_panes.py http://178.128.145.111:8081
Exit 1 when any pane is flagged, so it can gate a deploy script.
"""
from __future__ import annotations

import json
import re
import sys
import urllib.request

BOTS = ["mlb", "nba", "wnba", "tennis", "table-tennis", "darts",
        "world-cup", "billboard", "reality-leaks", "hormuz", "rain", "temp",
        "cpi"]


def _get(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent":
                                                "pane-check"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode("utf-8", "replace")


def page_rows(base: str, bot: str) -> int:
    raw = _get(f"{base}/?bot={bot}&tab=watchlist")
    body = re.sub(r"<script.*?</script>", "", raw, flags=re.S)
    i = body.rfind("Model vs market")
    seg = body[i:] if i >= 0 else body
    j = seg.find("How EV is calculated")
    if j > 0:
        seg = seg[:j]
    return len(re.findall(r"data-field='verdict'", seg))


def eligible_rows(base: str, bot: str) -> int:
    try:
        payload = json.loads(_get(f"{base}/api/snapshot?bot={bot}"))
    except Exception:  # noqa: BLE001
        return 0
    n = 0
    for r in payload.get("watchlist") or []:
        model = (r.get("pinnacle_prob_yes")
                 if r.get("pinnacle_prob_yes") is not None
                 else r.get("model_prob_yes"))
        ask = r.get("kalshi_yes")
        live_quote = isinstance(ask, (int, float)) and 1 < ask < 99
        oi_ok = (r.get("open_interest") or 0) > 0
        if model is not None and live_quote and oi_ok:
            n += 1
    return n


def main() -> int:
    base = (sys.argv[1] if len(sys.argv) > 1
            else "http://localhost:8080").rstrip("/")
    failures = 0
    print(f"{'bot':<14} {'page':>5} {'eligible':>9}")
    for bot in BOTS:
        try:
            p = page_rows(base, bot)
        except Exception as e:  # noqa: BLE001
            print(f"{bot:<14} fetch failed: {e}")
            failures += 1
            continue
        e_ = eligible_rows(base, bot)
        flag = ""
        if p == 0 and e_ > 0:
            flag = "  <-- BLANK PANE with eligible rows"
            failures += 1
        print(f"{bot:<14} {p:>5} {e_:>9}{flag}")
    if failures:
        print(f"\n{failures} pane(s) flagged")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
