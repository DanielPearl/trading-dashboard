"""NBA in-game prediction model.

Two-input heuristic baseline:

1. **Score / time component**. Live win probability for the bet's
   team, derived from the standard basketball "lead / sqrt(time)"
   logistic. Captures the wish-list features ``score differential``
   and ``time remaining`` directly.

2. **Market component**. Three cross-sport features built from the
   per-ticker market history we already record: velocity (recent
   price drift), volatility (noise), divergence from pre-game
   prior (overreaction signal).

The two are combined into a final live YES probability + an action
recommendation that the hedge monitor consults. Output is intentionally
conservative: ``confidence`` is high only when both the game-state
and market-state components agree.

What this model does NOT yet use (TODO / future training)
---------------------------------------------------------
- play-by-play stats (foul trouble, lineup combos, pace, shooting %
  vs expected, turnover rate, rebound dominance, FT rate, bench vs
  starter splits, etc.)
- injury / minutes-restriction flags
- timeout patterns and clutch-time historical performance
- news / social sentiment

Each of those needs a separate live data feed and a training pipeline
on historical play-by-play. The README in this package lists the
specific data sources and steps required.
"""
from __future__ import annotations

import json
import logging
import math
import re
import time
import urllib.request
from typing import Any, Dict, Optional, Tuple

from .base import LivePrediction, softclip_prob, clamp01
from .base import (
    ACTION_HOLD, ACTION_EXIT_NOW, ACTION_LET_RUN, ACTION_NEUTRAL,
)
from . import features as _features
from . import market_state as _market_state

log = logging.getLogger("dashboard.in_game.nba")


# ESPN cache mirrors the one in sport_game_state but with a longer
# TTL since this module reads richer fields. 30 s keeps the data
# fresh enough for a 30 s hedge tick without thrashing the API.
_ESPN_CACHE: Dict[str, Tuple[float, Optional[dict]]] = {}
_ESPN_TTL_SECONDS = 30.0


def _fetch_espn_nba() -> Optional[dict]:
    now = time.time()
    cached = _ESPN_CACHE.get("nba")
    if cached and (now - cached[0]) < _ESPN_TTL_SECONDS:
        return cached[1]
    url = ("https://site.api.espn.com/apis/site/v2/sports/"
            "basketball/nba/scoreboard")
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "kalshi-dashboard/1.0"},
        )
        with urllib.request.urlopen(req, timeout=5) as r:
            data = json.load(r)
    except Exception as exc:  # noqa: BLE001
        log.debug("ESPN NBA fetch failed: %s", exc)
        _ESPN_CACHE["nba"] = (now, None)
        return None
    _ESPN_CACHE["nba"] = (now, data)
    return data


_NBA_MID_RE = re.compile(r"^\d{2}[A-Z]{3}\d{1,2}(?P<teams>[A-Z]+)$")


def _parse_ticker(ticker: str) -> Optional[Tuple[str, str, str]]:
    """``KXNBAGAME-26MAY18SASOKC-OKC`` → ``("SAS", "OKC", "OKC")``.

    Returns ``(team_a, team_b, our_team)``. ``our_team`` is the team
    the YES side of this market is betting on.
    """
    if not ticker:
        return None
    parts = ticker.split("-")
    if len(parts) < 3:
        return None
    m = _NBA_MID_RE.match(parts[1])
    if not m:
        return None
    teams = m.group("teams")
    # ESPN abbreviations are typically 3 chars. Some legacy tickers
    # use 2 — try both splits if 3+3 doesn't add up.
    for split_at in (3, 2, 4):
        if 2 * split_at == len(teams):
            return (teams[:split_at], teams[split_at:], parts[2])
    if len(teams) >= 6:
        return (teams[:3], teams[3:], parts[2])
    return None


# NBA period clocks: 12-min quarters, 5-min overtimes.
_PERIOD_SECONDS = {1: 720, 2: 720, 3: 720, 4: 720}


def _seconds_remaining_in_game(period: int, clock_seconds: int) -> int:
    """Approximate remaining game seconds. Treats anything past Q4
    as overtime (5min each) — we won't know the exact OT count ahead
    of time, but for a "time remaining" feature that's fine."""
    if period <= 0:
        return 48 * 60
    if period >= 5:
        # In overtime — just the visible clock.
        return max(0, int(clock_seconds))
    later_periods = max(0, 4 - period)
    return int(clock_seconds) + later_periods * _PERIOD_SECONDS.get(1, 720)


def _parse_clock_seconds(display_clock: str) -> int:
    """ESPN ``displayClock`` is ``"M:SS"`` or ``"M:SS.t"``."""
    if not display_clock:
        return 0
    try:
        parts = display_clock.strip().split(":")
        if len(parts) != 2:
            return 0
        m = int(parts[0])
        s = float(parts[1])
        return int(m * 60 + s)
    except (TypeError, ValueError):
        return 0


def _live_state(ticker: str) -> Optional[Dict[str, Any]]:
    """Returns ESPN-derived live game state for the matchup encoded
    in this ticker, or ``None`` if we can't match it.
    """
    parsed = _parse_ticker(ticker)
    if not parsed:
        return None
    team_a, team_b, our_team = parsed
    data = _fetch_espn_nba()
    if not data:
        return None
    for ev in (data.get("events") or []):
        for comp in (ev.get("competitions") or []):
            competitors = comp.get("competitors") or []
            abbr_to_score: Dict[str, int] = {}
            for c in competitors:
                ab = ((c.get("team") or {}).get("abbreviation") or "").upper()
                try:
                    abbr_to_score[ab] = int(c.get("score") or 0)
                except (TypeError, ValueError):
                    abbr_to_score[ab] = 0
            if team_a.upper() not in abbr_to_score:
                continue
            if team_b.upper() not in abbr_to_score:
                continue
            status = (comp.get("status") or {}).get("type") or {}
            state = (status.get("state") or "").lower()
            period = int((comp.get("status") or {}).get("period") or 0)
            clock = _parse_clock_seconds(
                (comp.get("status") or {}).get("displayClock") or "",
            )
            our_score = abbr_to_score.get(our_team.upper(), 0)
            opp = (team_b if our_team.upper() == team_a.upper()
                    else team_a)
            opp_score = abbr_to_score.get(opp.upper(), 0)
            return {
                "state": state,
                "period": period,
                "clock_seconds": clock,
                "seconds_remaining": _seconds_remaining_in_game(period, clock),
                "our_team": our_team,
                "opp_team": opp,
                "our_score": our_score,
                "opp_score": opp_score,
                "our_lead": our_score - opp_score,
            }
    return None


# Logistic coefficient used in the score/time win-prob model. The
# 0.045 factor is the canonical basketball figure for "lead per
# sqrt(seconds remaining)" — see Brian Burke / basketball-reference
# write-ups. Tunable later when we have a training set.
_LEAD_TIME_COEF = 0.045


def _win_prob_from_state(state: Dict[str, Any]) -> float:
    """Live win prob for ``our_team`` given the score / time state."""
    lead = float(state.get("our_lead", 0))
    secs = float(state.get("seconds_remaining", 1))
    if secs <= 0:
        # Game over — prob is just sign(lead) with a small floor.
        if lead > 0:
            return 0.99
        if lead < 0:
            return 0.01
        return 0.5
    z = _LEAD_TIME_COEF * lead / math.sqrt(secs)
    return 1.0 / (1.0 + math.exp(-z))


def predict(bot: Dict[str, Any], position: Dict[str, Any],
              market_view: Optional[Dict[str, Any]] = None,
              ) -> Optional[LivePrediction]:
    ticker = position.get("ticker") or ""
    state = _live_state(ticker)
    if not state or state.get("state") != "in":
        return None  # pre or post — no in-game prediction.

    # Pre-game prior used for the divergence / reversion features.
    pre_game = None
    try:
        pre_game = position.get("model_yes_prob_at_entry")
        pre_game = float(pre_game) if pre_game is not None else None
    except (TypeError, ValueError):
        pre_game = None

    db_path = bot.get("db_path") or ""
    history = _market_state.recent_market_history(
        db_path, ticker, hours=6, limit=400,
    )
    current_market_cents = (
        (market_view or {}).get("yes_ask_cents")
        or _market_state.latest_yes_cents(db_path, ticker)
    )
    try:
        current_market_prob = (float(current_market_cents) / 100.0
                                if current_market_cents is not None
                                else None)
    except (TypeError, ValueError):
        current_market_prob = None

    state_prob = _win_prob_from_state(state)
    velocity = _features.market_velocity(history, window_seconds=300)
    volat = _features.volatility(history, window_seconds=600)
    div = _features.divergence(pre_game, current_market_prob) if pre_game is not None else None
    reversion_pull = (
        _features.expected_reversion_pull(
            pre_game or state_prob, current_market_prob or state_prob,
            volat,
        )
        if (pre_game is not None and current_market_prob is not None)
        else 0.0
    )

    # Blend: the state-derived probability is the anchor. Reversion
    # pull nudges us back toward the pre-game prior when divergence
    # is high but live volatility is low (overreaction). High
    # volatility damps the in-game model's confidence — the market
    # is in flux and we shouldn't pretend to know better.
    blended = state_prob - 0.5 * reversion_pull
    blended = softclip_prob(blended)

    # Confidence rules: high when game is past the first quarter
    # AND market volatility is below 1.5. Otherwise medium / low.
    secs_left = state.get("seconds_remaining", 2880)
    past_q1 = secs_left < 36 * 60  # under 36min remaining ≈ past Q1
    confidence = 0.0
    if past_q1 and volat < 1.5:
        confidence = 0.7
    elif past_q1:
        confidence = 0.45
    else:
        confidence = 0.25

    # Recommended action — interpret with respect to the side bet.
    side = (position.get("side") or "").upper()
    # The "live YES" from blended is "our team wins"; if we're on
    # the YES side of this ticker that equals our win prob.
    our_side_prob = blended if side == "YES" else (1.0 - blended)
    entry_cents = position.get("entry_price_cents")
    try:
        entry_prob = (float(entry_cents) / 100.0
                       if entry_cents is not None else None)
    except (TypeError, ValueError):
        entry_prob = None

    action = ACTION_NEUTRAL
    reason_bits = [
        f"lead={state.get('our_lead', 0):+d}",
        f"left={secs_left//60}m",
        f"state_prob={state_prob:.2f}",
        f"vol={volat:.2f}",
    ]
    if confidence >= 0.6 and our_side_prob < 0.30:
        action = ACTION_EXIT_NOW
        reason_bits.append("model expects loss")
    elif confidence >= 0.5 and entry_prob is not None and our_side_prob > entry_prob + 0.10:
        action = ACTION_LET_RUN
        reason_bits.append("model expects gain vs entry")
    elif div is not None and div > 0.20 and volat < 1.0:
        action = ACTION_HOLD
        reason_bits.append("likely market overreaction")

    return LivePrediction(
        live_prob_yes=clamp01(blended),
        confidence=confidence,
        recommended_action=action,
        reason=" · ".join(reason_bits),
        features={
            "state_prob": state_prob,
            "our_side_prob": our_side_prob,
            "market_velocity": velocity,
            "volatility": volat,
            "divergence": div if div is not None else 0.0,
            "reversion_pull": reversion_pull,
            "lead": float(state.get("our_lead", 0)),
            "seconds_remaining": float(secs_left),
        },
    )
