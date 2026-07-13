"""WNBA live executor — thin subclass of the shared ``SportLiveExecutor``.

Everything from the old 380-line monolith is now provided by
``_sport_live_executor.SportLiveExecutor``. The WNBA-specific
``require_pinnacle`` gate is deliberately dropped: the upstream
watchlist row already marks ``buy_eligible: False`` with a
``no_pinnacle_reference`` blocker when Pinnacle doesn't quote, so the
gate was a duplicate check on top of the same signal.

Orphan-adoption prefix is ``KXWNBAGAME`` — every match ticker the WNBA
exporter generates uses that series.
"""
from __future__ import annotations

from ._sport_live_executor import SportLiveExecutor


WNBA_TICKER_PREFIXES = ("KXWNBAGAME",)


class WNBALiveExecutor(SportLiveExecutor):
    def __init__(self, cfg: dict, state_path: str) -> None:
        super().__init__(
            cfg, state_path,
            bot_key="wnba",
            tournament="WNBA",
            surface="Court",
            win_verb="winning",
            hard={
                # Pre-refactor hard ceilings for the WNBA-tuned caps.
                "max_open_positions": 8,
                "max_orders_per_day": 10,
                "max_entry_price_cents": 85,
                "min_entry_price_cents": 10,
            },
            defaults={
                "max_open_positions": 4,
                "max_orders_per_day": 6,
                "min_edge_pp": 0.09,
                "max_entry_price_cents": 60,
                "min_entry_price_cents": 30,
            },
            orphan_ticker_prefixes=WNBA_TICKER_PREFIXES,
        )
