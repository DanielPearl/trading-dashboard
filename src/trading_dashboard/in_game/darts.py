"""Darts in-game model.

Darts paper-trade through the same tennis-shape adapter (watchlist.
json + sim_state.json), so the live signal lives in the bot's own
``live_prob_a`` / ``live_prob_b`` fields exactly like tennis. This
module delegates to ``tennis.predict`` with ``sport="darts"`` and
applies darts-specific tweaks where useful.

Darts-specific notes (and TODO for a richer future model)
---------------------------------------------------------
- Leg/set scoring is different from tennis sets — a player can
  "break throw" without ever serving, so the within-set tempo is
  faster. The confidence ladder in ``tennis.predict`` is tuned for
  tennis sets; darts settles much quicker per "set". The override
  here bumps confidence to high after a single set.
- Checkout-percentage proxies and leg-streak persistence are
  reachable from the bot's per-tick data but not yet exposed in
  watchlist.json. When the darts bot starts persisting them, plug
  them in as additional features and we can train an actual model.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from .base import LivePrediction
from . import tennis as _tennis


def predict(bot: Dict[str, Any], position: Dict[str, Any],
              market_view: Optional[Dict[str, Any]] = None,
              ) -> Optional[LivePrediction]:
    base = _tennis.predict(bot, position, market_view, sport="darts")
    if base is None:
        return base
    # Darts settles faster within a set than tennis does — bump the
    # confidence ladder so the model is "trusted" earlier.
    if base.features.get("sets_completed", 0) >= 1:
        bumped = min(0.85, base.confidence + 0.1)
        return LivePrediction(
            live_prob_yes=base.live_prob_yes,
            confidence=bumped,
            recommended_action=base.recommended_action,
            reason=base.reason,
            features=base.features,
        )
    return base
