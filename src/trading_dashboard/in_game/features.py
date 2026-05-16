"""Cross-sport features derived from Kalshi market data alone.

These are the most universally applicable in-game signals — they
don't need a sport-specific data feed, just the per-ticker history
the bot has been recording. All values are designed to be combined
linearly in the per-sport scorers, so they're all roughly on a
[-1, 1] or [0, 1] scale even when the underlying quantity isn't.

Features implemented
--------------------
``market_velocity``
    Recent rate of change in the YES price. Positive = market
    moving toward YES. Captures the "live odds movement velocity"
    feature on the user's wish list.

``volatility``
    Standard deviation of recent price changes. High volatility
    means the live state is genuinely uncertain — hedge monitor
    should defer to thresholds, not let the in-game model bully
    its way to a close.

``divergence``
    |pre_game_prob − current_market_prob|. Captures "market
    overreaction" — when the live state hasn't justified the gap.
    Always >= 0.
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple


def market_velocity(history: List[Tuple[float, float]],
                      window_seconds: int = 600) -> float:
    """Average price change per minute over the last ``window_seconds``.

    ``history`` is ``[(unix_ts, yes_cents), ...]`` in chronological
    order. Returns 0.0 when fewer than 2 points fall inside the
    window — not enough data to differentiate movement from noise.

    Output range: roughly [-1, 1] (capped). +1.0 ≈ +1¢/minute or
    faster YES move; -1.0 ≈ -1¢/minute or faster NO move.
    """
    if not history or len(history) < 2:
        return 0.0
    latest_ts, _ = history[-1]
    cutoff = latest_ts - window_seconds
    window = [(ts, v) for ts, v in history if ts >= cutoff]
    if len(window) < 2:
        return 0.0
    t0, p0 = window[0]
    t1, p1 = window[-1]
    minutes = max(1 / 60.0, (t1 - t0) / 60.0)
    cents_per_min = (p1 - p0) / minutes
    # Cap at ±1.0 so a single fluky jump doesn't dominate downstream
    # linear combinations. Real moves rarely exceed 1¢/min for very
    # long; sustained > 1¢/min is "fast" and clipped.
    return max(-1.0, min(1.0, cents_per_min))


def volatility(history: List[Tuple[float, float]],
                 window_seconds: int = 600) -> float:
    """Stdev of YES-price first-differences in the recent window.

    Range: 0 (perfectly stable) to ~5 (frantic). The hedge monitor
    treats anything > ~2 as "too noisy to act on the in-game model
    alone".
    """
    if not history or len(history) < 3:
        return 0.0
    latest_ts, _ = history[-1]
    cutoff = latest_ts - window_seconds
    window = [v for ts, v in history if ts >= cutoff]
    if len(window) < 3:
        return 0.0
    diffs = [window[i + 1] - window[i] for i in range(len(window) - 1)]
    mean = sum(diffs) / len(diffs)
    var = sum((d - mean) ** 2 for d in diffs) / len(diffs)
    return math.sqrt(var)


def divergence(pre_game_prob: Optional[float],
                 current_market_prob: Optional[float]) -> Optional[float]:
    """Magnitude of disagreement between the pre-game model and the
    market right now. Returns ``None`` when either is unknown.

    Output range: [0, 1]. Large divergence with low live-volatility
    is the textbook "market overreaction" pattern.
    """
    if pre_game_prob is None or current_market_prob is None:
        return None
    try:
        return abs(float(pre_game_prob) - float(current_market_prob))
    except (TypeError, ValueError):
        return None


def expected_reversion_pull(pre_game_prob: float,
                              current_market_prob: float,
                              live_volatility: float,
                              divergence_floor: float = 0.10,
                              ) -> float:
    """How strongly should the model expect the market to revert
    toward the pre-game prior? Positive = expect YES to drop (NO
    rally); negative = expect YES to rise.

    Heuristic: large divergence with low volatility = strong
    reversion expectation. Low divergence or high volatility = no
    pull.
    """
    div = current_market_prob - pre_game_prob
    if abs(div) < divergence_floor:
        return 0.0
    # Volatility damp: high vol means whatever moved this much could
    # keep moving in the same direction. Damp the reversion pull.
    damp = max(0.0, 1.0 - (live_volatility / 4.0))
    return div * damp
