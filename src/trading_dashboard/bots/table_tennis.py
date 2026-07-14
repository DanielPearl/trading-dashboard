"""Table-tennis bot (TT Elite, Pinnacle guest benchmark) — thin spec over the shared Shape-B loop.

All tick mechanics (fetch → collapse → rows → paper sim / live
executor → watchlist mirror) live in bots/_sport_bot.py; this module
only names what differs for table-tennis. See _sport_bot's docstring for the
sim/live process split and the arming rules.
"""
from __future__ import annotations

from ._sport_bot import SportSpec, make_start_daemon

BOT_KEY = "table-tennis"

SPEC = SportSpec(
    bot_key=BOT_KEY,
    name="table-tennis",
    alias="table_tennis_src",
    fetch_fn="fetch_table_tennis_markets",
    repo_default="/root/table-tennis-forecast",
    interval_default=60,
    noun="matches",
    executor_kwargs={"tournament": "TT Elite", "surface": "Indoor",
                     "win_verb": "winning"},
    benchmark_guest_sport="table_tennis",
    # TT rows without a Pinnacle line keep the upstream
    # Elo model probs (display-only; never buy-eligible).
    benchmark_kwargs={"win_verb": "winning",
                      "keep_model_probs": True},
    records_to_builder=False,
)

start_daemon = make_start_daemon(SPEC)
