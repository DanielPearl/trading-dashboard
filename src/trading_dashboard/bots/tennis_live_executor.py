"""Tennis live executor — thin subclass of the shared ``SportLiveExecutor``.

Every behaviour that used to live in this file's 1500-line monolith is
now provided by ``_sport_live_executor.SportLiveExecutor`` (buy path,
profit-lock, mark-to-market, dry-run void, arming from the Home-tab
toggle, stop-loss, gain-based profit-lock, hours-to-close gate) and
``_live_core`` (Kalshi session, atomic writes, closed-record shape).

The subclass exists purely to stamp the tennis-specific ticker
prefixes onto the orphan-adoption path — a shared Kalshi account
lists positions from every bot, and re-adopting the wrong one would
silently profit-lock another bot's win. ``KXATPMATCH`` / ``KXWTAMATCH``
/ ``KXITFMATCH`` cover every match ticker the upstream exporter
generates; if the tennis series changes, add the prefix here.
"""
from __future__ import annotations

from ._sport_live_executor import SportLiveExecutor


TENNIS_TICKER_PREFIXES = ("KXATPMATCH", "KXWTAMATCH", "KXITFMATCH")


class TennisLiveExecutor(SportLiveExecutor):
    def __init__(self, cfg: dict, state_path: str) -> None:
        super().__init__(
            cfg, state_path,
            bot_key="tennis",
            tournament="Tennis",
            surface="Unknown",
            win_verb="winning",
            orphan_ticker_prefixes=TENNIS_TICKER_PREFIXES,
        )
