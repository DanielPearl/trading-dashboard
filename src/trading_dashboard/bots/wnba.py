"""WNBA bot (KXWNBAGAME) — thin spec over the shared Shape-B loop.

All tick mechanics (fetch → collapse → rows → paper sim / live
executor → watchlist mirror) live in bots/_sport_bot.py; this module
only names what differs for wnba. See _sport_bot's docstring for the
sim/live process split and the arming rules.
"""
from __future__ import annotations

from ._sport_bot import SportSpec, make_start_daemon

BOT_KEY = "wnba"

SPEC = SportSpec(
    bot_key=BOT_KEY,
    name="wnba",
    alias="wnba_src",
    fetch_fn="fetch_wnba_markets",
    repo_default="/root/wnba",
    interval_default=120,
    noun="games",
    executor_kwargs={"tournament": "WNBA", "surface": "Basketball",
                     "win_verb": "winning"},
)

start_daemon = make_start_daemon(SPEC)
