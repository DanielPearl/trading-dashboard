"""World Cup live executor — thin subclass of the shared
``SportLiveExecutor``.

Everything from the old 390-line monolith is now provided by
``_sport_live_executor.SportLiveExecutor``. World Cup rows carry
``kickoff`` (not ``expected_expiration_time``) which the base already
respects via its ``prematch_buffer_minutes`` gate.

Orphan-adoption prefix is ``KXWCADVANCE`` — every World Cup match
ticker the exporter generates uses that series.
"""
from __future__ import annotations

from ._sport_live_executor import SportLiveExecutor


WORLD_CUP_TICKER_PREFIXES = ("KXWCADVANCE",)


class WorldCupLiveExecutor(SportLiveExecutor):
    def __init__(self, cfg: dict, state_path: str) -> None:
        super().__init__(
            cfg, state_path,
            bot_key="world-cup",
            tournament="FIFA World Cup",
            surface="Grass",
            win_verb="advancing",
            hard={
                "max_open_positions": 8,
                "max_orders_per_day": 10,
                "max_entry_price_cents": 90,
                "min_entry_price_cents": 5,
            },
            defaults={
                "max_open_positions": 4,
                "max_orders_per_day": 6,
                "min_edge_pp": 0.08,
                "max_entry_price_cents": 85,
                "min_entry_price_cents": 10,
                "prematch_buffer_minutes": 15,
            },
            orphan_ticker_prefixes=WORLD_CUP_TICKER_PREFIXES,
        )
