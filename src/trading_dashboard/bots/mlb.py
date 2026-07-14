"""MLB bot (KXMLBGAME + KXNPBGAME + KXKBOGAME) — thin spec over the shared Shape-B loop.

All tick mechanics (fetch → collapse → rows → paper sim / live
executor → watchlist mirror) live in bots/_sport_bot.py; this module
only names what differs for mlb. See _sport_bot's docstring for the
sim/live process split and the arming rules.
"""
from __future__ import annotations

from ._sport_bot import SportSpec, make_start_daemon

BOT_KEY = "mlb"

SPEC = SportSpec(
    bot_key=BOT_KEY,
    name="mlb",
    alias="baseball_src",
    fetch_fn="fetch_mlb_markets",
    repo_default="/root/baseball",
    interval_default=120,
    noun="games",
    executor_kwargs={"tournament": "MLB", "surface": "Baseball",
                     "win_verb": "winning"},
)

start_daemon = make_start_daemon(SPEC)
