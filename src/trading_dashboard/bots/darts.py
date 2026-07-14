"""Darts bot (PDC matches, Pinnacle guest benchmark) — thin spec over the shared Shape-B loop.

All tick mechanics (fetch → collapse → rows → paper sim / live
executor → watchlist mirror) live in bots/_sport_bot.py; this module
only names what differs for darts. See _sport_bot's docstring for the
sim/live process split and the arming rules.
"""
from __future__ import annotations

from ._sport_bot import SportSpec, make_start_daemon

BOT_KEY = "darts"

SPEC = SportSpec(
    bot_key=BOT_KEY,
    name="darts",
    alias="darts_src",
    fetch_fn="fetch_darts_markets",
    repo_default="/root/darts-forecast",
    interval_default=60,
    noun="matches",
    executor_kwargs={"tournament": "PDC", "surface": "Indoor",
                     "win_verb": "winning"},
    benchmark_guest_sport="darts",
    benchmark_kwargs={"win_verb": "winning"},
    records_to_builder=False,
)

start_daemon = make_start_daemon(SPEC)
