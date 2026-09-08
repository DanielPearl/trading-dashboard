"""NBA bot (KXNBAGAME + Summer League) — thin spec over the shared Shape-B loop.

All tick mechanics (fetch → collapse → rows → paper sim / live
executor → watchlist mirror) live in bots/_sport_bot.py; this module
only names what differs for nba. See _sport_bot's docstring for the
sim/live process split and the arming rules.
"""
from __future__ import annotations

from ._sport_bot import SportSpec, make_start_daemon

# Merged basketball bot (user 2026-09-08): the /root/nba repo now
# discovers KXNBAGAME + KXNBASUMMERGAME + KXWNBAGAME and benchmarks
# both leagues; the old separate wnba daemon is retired (its card,
# filter entry and toggle fold into this one).
BOT_KEY = "basketball"

SPEC = SportSpec(
    bot_key=BOT_KEY,
    name="basketball",
    alias="nba_src",
    fetch_fn="fetch_nba_markets",
    repo_default="/root/nba",
    interval_default=120,
    noun="games",
    executor_kwargs={"tournament": "Basketball", "surface": "Basketball",
                     "win_verb": "winning"},
)

start_daemon = make_start_daemon(SPEC)
