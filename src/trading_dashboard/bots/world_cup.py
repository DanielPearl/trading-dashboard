"""World Cup bot (KXWCGAME / KXWCADVANCE) — thin spec over the shared Shape-B loop.

All tick mechanics (fetch → collapse → rows → paper sim / live
executor → watchlist mirror) live in bots/_sport_bot.py; this module
only names what differs for world-cup. See _sport_bot's docstring for the
sim/live process split and the arming rules.
"""
from __future__ import annotations

from ._sport_bot import SportSpec, make_start_daemon

BOT_KEY = "world-cup"

SPEC = SportSpec(
    bot_key=BOT_KEY,
    name="world-cup",
    alias="world_cup_src",
    fetch_fn="fetch_world_cup_markets",
    repo_default="/root/world-cup",
    interval_default=300,
    noun="matches",
    executor_kwargs={"tournament": "FIFA World Cup 2026",
                     "surface": "Soccer", "win_verb": "advancing"},
)

start_daemon = make_start_daemon(SPEC)
