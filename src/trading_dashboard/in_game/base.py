"""Shared types for the in-game prediction layer."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


# Recommendation values the hedge monitor understands.
ACTION_HOLD = "hold"            # Keep position; ignore threshold hit if any.
ACTION_EXIT_NOW = "exit_now"    # Close the position immediately.
ACTION_LET_RUN = "let_run"      # Don't close on threshold — model thinks
                                # the move will reverse / position will win.
ACTION_NEUTRAL = "neutral"      # No strong opinion — defer to thresholds.


@dataclass
class LivePrediction:
    """Output of an in-game model for one bet on one position.

    Fields
    ------
    live_prob_yes : float
        The in-game model's estimate of P(YES resolves) given live
        game state. Always 0..1, never raw cents.
    confidence : float
        0..1 — how strongly the model trusts its own output. Low
        confidence ⇒ hedge monitor falls back to its standard
        thresholds.
    recommended_action : str
        One of ``ACTION_*``. The hedge monitor uses this to decide
        whether to fire on threshold, suppress, or pre-empt.
    reason : str
        Short human-readable string. Logged with every hedge action
        the in-game model influences.
    features : dict
        Raw per-feature values the model considered. Useful for the
        watchlist UI's hover state and for later debugging /
        regression analysis. Never load-bearing.
    """
    live_prob_yes: float
    confidence: float
    recommended_action: str
    reason: str
    features: Dict[str, float] = field(default_factory=dict)


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def softclip_prob(p: float, floor: float = 0.02, ceil: float = 0.98) -> float:
    """Pull extreme probability estimates back inside [floor, ceil].

    The in-game models we ship are heuristic — they should never
    claim certainty. Soft-clipping prevents a single fluky feature
    from blasting the output to 1.0 and overriding the hedge gate
    on weak evidence.
    """
    return max(floor, min(ceil, p))
