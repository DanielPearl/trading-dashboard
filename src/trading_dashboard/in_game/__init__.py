"""In-game prediction layer for the sport bots.

The pre-game models the bots ship with run before the match starts —
classifier on player/team/contract priors, written to each bot's
sim.db (or sim_state.json for tennis-shape). Those are untouched by
this package.

This layer fires *after* the match has gone live (gated by
``sport_game_state.is_hedge_allowed``) and produces a second
estimate: a live win probability for the bet's side, plus a
recommended action that the hedge monitor consults.

Why two models?
---------------
Sports markets are noisy pre-game (injury news, lineup leaks) and
then become signal-rich the moment the ball tips. A single model
trying to span both regimes either over-reacts to pre-game noise or
under-reacts to in-game shifts. Two models — one for each regime,
gated by the match-start signal — gives each one a coherent input
distribution to be calibrated against.

Boundary
--------
The pre-game model writes to the bot's existing storage. The
in-game model is read-only against the bot's pre-game outputs (uses
``model_yes_prob_at_entry`` as a prior) and never writes back to
sim.db / sim_state.json. Pre-game training data is never mixed with
in-game training data — they're separate populations.

Public API
----------
``predict(bot, position, market_view)`` returns ``LivePrediction``
or ``None`` (model can't speak — fall back to pre-game-only logic).
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from .base import LivePrediction
from . import nba as _nba
from . import tennis as _tennis
from . import darts as _darts


__all__ = ["LivePrediction", "predict"]


def predict(bot: Dict[str, Any], position: Dict[str, Any],
             market_view: Dict[str, Any] | None = None,
             ) -> Optional[LivePrediction]:
    """Dispatch to the sport-specific in-game model. Returns ``None``
    when the bot has no in-game model registered or when the model
    can't produce a confident estimate from available data.

    Side-effect: every confident *transition* (the model's
    recommended action flipping from its prior value for the same
    ticker) is appended to ``data/in_game_predictions.jsonl`` via
    the logger module. See ``in_game/logger.py``.
    """
    bot_key = (bot.get("key") or "").lower()
    dt = (bot.get("dashboard_type") or "standard").lower()
    pred: Optional[LivePrediction] = None
    try:
        if bot_key == "nba":
            pred = _nba.predict(bot, position, market_view)
        elif bot_key == "tennis" or (dt == "tennis" and bot_key == "tennis"):
            pred = _tennis.predict(bot, position, market_view,
                                      sport="tennis")
        elif bot_key == "table-tennis":
            pred = _tennis.predict(bot, position, market_view,
                                      sport="table-tennis")
        elif bot_key == "darts":
            pred = _darts.predict(bot, position, market_view)
    except Exception:  # noqa: BLE001
        # In-game model errors must NEVER take down the hedge tick.
        # A failed live prediction returns None — caller falls back
        # to pre-game thresholds.
        import logging
        logging.getLogger("dashboard.in_game").exception(
            "in_game.predict failed for bot=%s", bot_key,
        )
        return None
    if pred is not None:
        try:
            from . import logger as _ig_logger
            _ig_logger.maybe_log(bot, position, pred)
        except Exception:  # noqa: BLE001
            import logging
            logging.getLogger("dashboard.in_game").exception(
                "in_game.logger.maybe_log failed for bot=%s", bot_key,
            )
    return pred
