"""Tennis-shape in-game model (covers tennis, table-tennis).

The tennis-shape bots already produce a live probability estimate
inside their own watchlist.json (``live_prob_a`` / ``live_prob_b``).
That field is computed by the bot's own model from the live data
feed it subscribes to — it already captures most of the wish-list
features for the tennis bots:

- first/second serve %, breakpoint conversion, unforced errors,
  ace rate, rally length trends — fed into the bot's
  Baseline-Break model on every live update.

This module's job is to take that live estimate and overlay the
market-state features (velocity / volatility / divergence) so we
can recognize overreactions and avoid hedging into noise. We do
NOT recompute the per-point live model from scratch — the bot is
the authoritative live source.

What this layer adds on top of the bot's live model
---------------------------------------------------
1. Cross-sport market overreaction detection — flag positions
   where the market has moved sharply away from a stable live
   estimate.
2. Confidence rules tuned for tennis: hold off when the match is
   very early (first few games) or when live data is stale.
3. Recommended action that the hedge monitor can consult.

What we don't yet do (TODO for a future trained layer)
-----------------------------------------------------
- Body-language / tilt proxies (no public feed).
- Medical-timeout signals (some watchlists carry an
  ``injury_news_flag``; surfaced as a feature but not yet
  weighted because we lack a training set).
- Historical comeback probability from set/game score state
  (would need ATP/WTA point-by-point history; doable as a
  separate pipeline).
"""
from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .base import LivePrediction, softclip_prob, clamp01
from .base import (
    ACTION_HOLD, ACTION_EXIT_NOW, ACTION_LET_RUN, ACTION_NEUTRAL,
)
from . import features as _features
from . import news_signals as _news

log = logging.getLogger("dashboard.in_game.tennis")


# Watchlist cache so a 30s hedge tick doesn't re-read the file
# four times for the same poll. 15s TTL — fresh enough that a
# live point change won't be hidden.
_WL_CACHE: Dict[str, Tuple[float, Optional[dict]]] = {}
_WL_TTL_SECONDS = 15.0


def _load_watchlist(path: str | None) -> Optional[dict]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    now = time.time()
    cached = _WL_CACHE.get(path)
    if cached and (now - cached[0]) < _WL_TTL_SECONDS:
        return cached[1]
    try:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        log.debug("watchlist read failed (%s): %s", path, exc)
        _WL_CACHE[path] = (now, None)
        return None
    _WL_CACHE[path] = (now, data)
    return data


def _find_match_row(watchlist: dict, match_id: str) -> Optional[dict]:
    if not watchlist or not match_id:
        return None
    for row in (watchlist.get("rows") or []):
        if row.get("match_id") == match_id:
            return row
    return None


# Tennis current_score patterns we recognize. Examples:
#   "6-4 3-2"     → set 1: 6-4 done, set 2: 3-2 in progress
#   "0-0"         → first game just started
#   "6-4 6-7 4-3" → 3rd set in progress
# Table tennis: similar (sets to 11, best of 5/7).
# Darts: leg/set scores; we treat any digit pair as "started".
_SCORE_PAIR_RE = re.compile(r"(\d+)-(\d+)")


def _parse_score_state(score_str: str) -> Dict[str, int]:
    """Light parse of a current_score string. Returns rough state:
    ``{"sets_completed": int, "set_score_a": int, "set_score_b": int}``.
    Empty when we can't read it.
    """
    if not score_str:
        return {}
    pairs = _SCORE_PAIR_RE.findall(score_str)
    if not pairs:
        return {}
    completed = len(pairs) - 1
    try:
        a, b = pairs[-1]
        return {
            "sets_completed": completed,
            "set_score_a": int(a),
            "set_score_b": int(b),
        }
    except (TypeError, ValueError):
        return {}


def _our_side_player_label(ticker: str, row: dict) -> str:
    """The tennis adapter encodes side in the ticker tail as
    ``PLAYER_A`` or ``PLAYER_B``. The watchlist row tells us which
    label each side corresponds to.
    """
    tail = (ticker or "").rsplit("-", 1)[-1].upper()
    if tail in ("PLAYER_A", "A"):
        return "A"
    if tail in ("PLAYER_B", "B"):
        return "B"
    return ""


def predict(bot: Dict[str, Any], position: Dict[str, Any],
              market_view: Optional[Dict[str, Any]] = None,
              sport: str = "tennis",
              ) -> Optional[LivePrediction]:
    ticker = (position.get("ticker") or position.get("match_id") or "")
    if not ticker:
        return None
    watchlist = _load_watchlist(bot.get("watchlist_json_path"))
    if not watchlist:
        return None

    # Tennis-shape positions store the match_id on the position
    # itself; the hedge_monitor passes ticker = match_id here.
    match_id = position.get("match_id") or ticker
    row = _find_match_row(watchlist, match_id)
    if not row:
        return None

    score_str = (row.get("current_score") or "").strip()
    if not score_str or score_str in {"-", "—", "0-0", "0:0"}:
        return None  # match not actually live yet

    # The bot's own live estimate is the authoritative live signal.
    side_label = _our_side_player_label(ticker, row)
    if not side_label:
        return None
    live_prob_a = row.get("live_prob_a")
    live_prob_b = row.get("live_prob_b")
    try:
        live_prob_a = float(live_prob_a) if live_prob_a is not None else None
        live_prob_b = float(live_prob_b) if live_prob_b is not None else None
    except (TypeError, ValueError):
        live_prob_a, live_prob_b = None, None
    if live_prob_a is None and live_prob_b is None:
        return None
    if live_prob_a is None and live_prob_b is not None:
        live_prob_a = 1.0 - live_prob_b
    if live_prob_b is None and live_prob_a is not None:
        live_prob_b = 1.0 - live_prob_a

    our_team_live = live_prob_a if side_label == "A" else live_prob_b

    # Pre-game prior — for the divergence feature.
    pre_match_prob_a = row.get("pre_match_prob_a")
    try:
        pre_match_prob_a = (float(pre_match_prob_a)
                             if pre_match_prob_a is not None else None)
    except (TypeError, ValueError):
        pre_match_prob_a = None
    pre_game_our = (pre_match_prob_a if side_label == "A"
                     else (1.0 - pre_match_prob_a)
                          if pre_match_prob_a is not None else None)

    # Market state. Current price from the watchlist row; price
    # history from the dashboard's tennis_snapshotter daemon
    # (writes per-match JSONL every 60s to data/tennis_history/).
    market_yes_cents = row.get("yes_ask_cents_a" if side_label == "A"
                                else "yes_ask_cents_b")
    try:
        market_yes_cents = (int(market_yes_cents)
                              if market_yes_cents is not None else None)
    except (TypeError, ValueError):
        market_yes_cents = None
    current_market_prob = (market_yes_cents / 100.0
                            if market_yes_cents is not None else None)

    div = _features.divergence(pre_game_our, current_market_prob)
    # Cross-sport velocity / volatility from the snapshotter's
    # per-match history. Same window the NBA model uses.
    try:
        from . import tennis_snapshotter as _snap
        history = _snap.yes_cents_history(
            bot.get("key") or "tennis", match_id, side_label, hours=6,
        )
    except Exception:  # noqa: BLE001
        history = []
    velocity = _features.market_velocity(history, window_seconds=300)
    volat = _features.volatility(history, window_seconds=600)

    state = _parse_score_state(score_str)
    sets_completed = state.get("sets_completed", 0)

    # Confidence: low while still in set 1 (highly noisy), medium
    # during set 2, high in set 3 onwards.
    if sets_completed == 0:
        confidence = 0.35
    elif sets_completed == 1:
        confidence = 0.55
    else:
        confidence = 0.75

    # Tennis-shape rows also expose an ``injury_news_flag`` — when
    # set, drop confidence sharply. Surfaced as a feature so the UI
    # can show it; not yet weighted as a decision input because we
    # lack a calibration history.
    injury_flag = bool(row.get("injury_news_flag"))
    if injury_flag:
        confidence = max(0.1, confidence - 0.25)

    # The tennis bot publishes its own per-match confidence and
    # volatility scores on the watchlist row. Use them to modulate
    # the in-game model's confidence: when the bot itself isn't
    # sure, we shouldn't be either.
    bot_confidence_score = None
    bot_volatility_score = None
    try:
        v = row.get("confidence_score")
        bot_confidence_score = float(v) if v is not None else None
    except (TypeError, ValueError):
        pass
    try:
        v = row.get("volatility_score")
        bot_volatility_score = float(v) if v is not None else None
    except (TypeError, ValueError):
        pass
    if bot_confidence_score is not None:
        # Take a soft minimum: in-game confidence can't exceed the
        # bot's own confidence by more than 10pp. Caps our claims
        # when the bot is unsure of its live estimate.
        cap = min(0.95, bot_confidence_score + 0.10)
        confidence = min(confidence, cap)
    if (bot_volatility_score is not None
            and bot_volatility_score > 0.5):
        # High volatility from the bot — drop our confidence
        # proportionally. 0.5 is the calibrated "things are
        # moving fast" threshold the bot itself uses.
        haircut = min(0.30, (bot_volatility_score - 0.5) * 0.40)
        confidence = max(0.10, confidence - haircut)

    # The bot also publishes its own recommended action. We surface
    # it as a feature but DON'T let it drive our action — that
    # would just echo the bot to itself. The user can see both
    # on the In-game predictions panel and judge agreement.
    bot_recommendation = (row.get("recommended_action") or "").strip()

    # Cross-sport news scanner — pull last-name keywords from the
    # match's player labels and ask ESPN's tennis news feed if
    # there's an injury-related article mentioning either player
    # in the last 12 hours. Only meaningful for the `tennis`
    # sport endpoint; table-tennis / darts don't have an ESPN
    # news feed and return 0.
    news_count = 0
    try:
        sport_path = _news.NEWS_PATHS.get(sport)
        if sport_path:
            # Try a few keyword forms — the bot stores the side
            # player names under _yes_label / _no_label / title.
            kws: list = []
            for k in ("player_a", "player_b", "title_a", "title_b"):
                val = row.get(k)
                if val:
                    # Take the last name (last whitespace-delim
                    # token) for a more specific match.
                    parts = str(val).split()
                    if parts:
                        kws.append(parts[-1])
            news_count = _news.injury_signal_count(
                sport_path, kws, max_age_hours=12,
            )
    except Exception:  # noqa: BLE001
        news_count = 0
    if news_count > 0:
        # Drop confidence on injury news — symmetric across sides
        # since we can't always tell which player the article is
        # about without proper NLP.
        confidence = max(0.10, confidence - 0.10 * min(3, news_count))

    # Volatility damp — but we don't have it from history yet.
    # Reversion pull stays modest.
    reversion_pull = 0.0
    if (pre_game_our is not None and current_market_prob is not None
            and div is not None and div > 0.15):
        reversion_pull = (current_market_prob - pre_game_our) * 0.3

    blended = our_team_live - reversion_pull
    blended = softclip_prob(blended)

    side = (position.get("side") or "").upper()
    # YES bet on a tennis ticker pays out when the named player wins.
    # The blended live prob is for our team already (we picked the
    # right column above), so:
    our_bet_prob = blended if side in ("YES", "PLAYER_A", "PLAYER_B") else (1.0 - blended)
    # Tennis paper-trades record entry_market_prob in 0..1 (not cents).
    entry_prob = None
    raw_entry = position.get("entry_market_prob")
    if raw_entry is None:
        raw_entry = position.get("entry_price_cents")
        try:
            entry_prob = (float(raw_entry) / 100.0
                           if raw_entry is not None else None)
        except (TypeError, ValueError):
            entry_prob = None
    else:
        try:
            entry_prob = float(raw_entry)
        except (TypeError, ValueError):
            entry_prob = None

    # Historical comeback probability from current score state.
    # Heuristic baseline from published tennis serve-dominance models
    # (Klaassen-Magnus style): when down a set, a 60% baseline server
    # has roughly a ~28% comeback probability in best-of-3, ~38% in
    # best-of-5. We use a simplified table keyed on (our_sets_won,
    # opp_sets_won) for best-of-3 (default for tennis/table-tennis).
    # Negative comeback prob ⇒ we're not behind; surface as None.
    comeback_prob: Optional[float] = None
    our_sets = (sets_completed if side_label == "A" else 0)  # placeholder
    # Properly compute from score parses — re-walk pairs to count
    # set wins on each side.
    set_pairs = _SCORE_PAIR_RE.findall(score_str or "")
    our_sets_won = 0
    opp_sets_won = 0
    if set_pairs:
        # All pairs except the last are completed sets.
        for pair in set_pairs[:-1]:
            try:
                a, b = int(pair[0]), int(pair[1])
            except (TypeError, ValueError):
                continue
            # Standard tennis set: first to 6 with 2-game lead, or
            # tiebreak to 7. Table tennis sets go to 11. We don't
            # know which is which from score format alone, so:
            # whichever side has more games "won" the set.
            if side_label == "A":
                our, opp = a, b
            else:
                our, opp = b, a
            if our > opp:
                our_sets_won += 1
            elif opp > our:
                opp_sets_won += 1
    # Comeback table (we trail). Best-of-3 default.
    COMEBACK_BO3 = {
        (0, 1): 0.30,  # down a set in BO3, normal player
        (0, 2): 0.05,  # match point essentially
        (1, 2): 0.05,
    }
    if opp_sets_won > our_sets_won:
        comeback_prob = COMEBACK_BO3.get(
            (our_sets_won, opp_sets_won),
        )

    action = ACTION_NEUTRAL
    reason_bits = [
        f"sets={sets_completed}",
        f"set_score={state.get('set_score_a', 0)}-{state.get('set_score_b', 0)}",
        f"live_prob={our_team_live:.2f}",
    ]
    if injury_flag:
        reason_bits.append("injury_flag")
    if confidence >= 0.6 and our_bet_prob < 0.30:
        action = ACTION_EXIT_NOW
        reason_bits.append("model expects loss")
    elif (comeback_prob is not None and comeback_prob < 0.10
            and confidence >= 0.5):
        # Match is functionally over and we're losing — exit if we
        # haven't been told otherwise by the bot's live model.
        action = ACTION_EXIT_NOW
        reason_bits.append(
            f"comeback prob ~{comeback_prob*100:.0f}% — match "
            "essentially decided")
    elif confidence >= 0.5 and entry_prob is not None and our_bet_prob > entry_prob + 0.10:
        action = ACTION_LET_RUN
        reason_bits.append("model expects gain vs entry")
    elif div is not None and div > 0.20:
        action = ACTION_HOLD
        reason_bits.append("possible market overreaction")

    features_out = {
        "live_prob_a": float(live_prob_a) if live_prob_a is not None else 0.0,
        "live_prob_b": float(live_prob_b) if live_prob_b is not None else 0.0,
        "our_team_live": our_team_live,
        "our_bet_prob": our_bet_prob,
        "divergence": div if div is not None else 0.0,
        "sets_completed": float(sets_completed),
        "injury_flag": 1.0 if injury_flag else 0.0,
        "market_velocity": velocity,
        "volatility": volat,
    }
    if bot_confidence_score is not None:
        features_out["bot_confidence_score"] = bot_confidence_score
    if bot_volatility_score is not None:
        features_out["bot_volatility_score"] = bot_volatility_score
    # Comeback context — surfaced even when 0 so the UI can show
    # "0% comeback prob" for hopeless states rather than blanking.
    if comeback_prob is not None:
        features_out["comeback_prob"] = comeback_prob
    if news_count > 0:
        features_out["news_injury_count"] = float(news_count)

    if bot_recommendation:
        reason_bits.append(f"bot_rec={bot_recommendation}")
    if comeback_prob is not None and comeback_prob < 0.15:
        reason_bits.append(f"comeback_p={comeback_prob:.2f}")

    return LivePrediction(
        live_prob_yes=clamp01(blended),
        confidence=confidence,
        recommended_action=action,
        reason=f"[{sport}] " + " · ".join(reason_bits),
        features=features_out,
    )
