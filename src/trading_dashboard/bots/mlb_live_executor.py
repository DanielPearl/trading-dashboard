"""MLB live executor — thin parameterization of the shared
SportLiveExecutor (this module was the template the generic class was
extracted from; see bots/_sport_live_executor.py for the mechanics and
the safety model). Kept as its own module so bots/mlb.py's import and
the config docs stay stable."""
from __future__ import annotations

from ._sport_live_executor import SportLiveExecutor


class MLBLiveExecutor(SportLiveExecutor):
    def __init__(self, cfg: dict, state_path: str) -> None:
        super().__init__(
            cfg, state_path,
            bot_key="mlb",
            tournament="MLB",
            surface="Baseball",
            win_verb="winning",
        )
