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
from . import news_signals as _news

log = logging.getLogger("dashboard.in_game.nba")


# ESPN cache mirrors the one in sport_game_state but with a longer
# TTL since this module reads richer fields. 30 s keeps the data
# fresh enough for a 30 s hedge tick without thrashing the API.
_ESPN_CACHE: Dict[str, Tuple[float, Optional[dict]]] = {}
_ESPN_TTL_SECONDS = 30.0
# /summary per-event response is bigger; cache 60s per event.
_ESPN_SUMMARY_CACHE: Dict[str, Tuple[float, Optional[dict]]] = {}
_ESPN_SUMMARY_TTL = 60.0


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


def _fetch_espn_summary(event_id: str) -> Optional[dict]:
    """ESPN's per-event /summary endpoint. Returns predictor (their
    own win projection), injuries list, and — for live games —
    per-team statistics + per-player box-score stats. Cached 60s.
    """
    if not event_id:
        return None
    now = time.time()
    cached = _ESPN_SUMMARY_CACHE.get(event_id)
    if cached and (now - cached[0]) < _ESPN_SUMMARY_TTL:
        return cached[1]
    url = (f"https://site.api.espn.com/apis/site/v2/sports/"
            f"basketball/nba/summary?event={event_id}")
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "kalshi-dashboard/1.0"},
        )
        with urllib.request.urlopen(req, timeout=5) as r:
            data = json.load(r)
    except Exception as exc:  # noqa: BLE001
        log.debug("ESPN /summary fetch failed (%s): %s", event_id, exc)
        _ESPN_SUMMARY_CACHE[event_id] = (now, None)
        return None
    _ESPN_SUMMARY_CACHE[event_id] = (now, data)
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
        event_id = ev.get("id") or ""
        for comp in (ev.get("competitions") or []):
            competitors = comp.get("competitors") or []
            abbr_to_score: Dict[str, int] = {}
            abbr_to_home_away: Dict[str, str] = {}
            for c in competitors:
                ab = ((c.get("team") or {}).get("abbreviation") or "").upper()
                try:
                    abbr_to_score[ab] = int(c.get("score") or 0)
                except (TypeError, ValueError):
                    abbr_to_score[ab] = 0
                abbr_to_home_away[ab] = (c.get("homeAway") or "").lower()
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
                "event_id": str(event_id),
                "period": period,
                "clock_seconds": clock,
                "seconds_remaining": _seconds_remaining_in_game(period, clock),
                "our_team": our_team,
                "opp_team": opp,
                "our_score": our_score,
                "opp_score": opp_score,
                "our_lead": our_score - opp_score,
                "our_home_away": abbr_to_home_away.get(our_team.upper(), ""),
            }
    return None


def _extract_summary_features(event_id: str, our_team: str,
                                 opp_team: str) -> Dict[str, Any]:
    """Pull the rich features ESPN's /summary endpoint exposes:

    Returns a dict (possibly partial) with:
      - ``espn_win_proj_our``  : ESPN's predicted win % for our team
                                  (0..1). Useful as a second baseline.
      - ``our_injury_count``   : count of our-team injuries reported
      - ``opp_injury_count``   : count of opponent injuries reported
      - ``our_critical_injuries`` : injuries with status Out / DTD
      - ``team_stat_gap_*``    : per-stat (FG%, FT%, etc.) live
                                  gap between our team and opponent
                                  (only present when game is live)
      - ``foul_trouble_count`` : count of our-team players with ≥4
                                  personal fouls (live games only)

    All fields are optional — missing data simply omits the key.
    """
    out: Dict[str, Any] = {}
    summary = _fetch_espn_summary(event_id)
    if not summary:
        return out

    # ── ESPN's own win projection (predictor.homeTeam / awayTeam)
    pred = summary.get("predictor") or {}
    header = summary.get("header") or {}
    # The predictor uses team ids; map id -> abbreviation via header.
    abbr_by_id: Dict[str, str] = {}
    for comp in ((header.get("competitions") or [{}])[0]
                  .get("competitors") or []):
        team = comp.get("team") or {}
        tid = str(team.get("id") or "")
        ab = (team.get("abbreviation") or "").upper()
        if tid and ab:
            abbr_by_id[tid] = ab
    for side in ("homeTeam", "awayTeam"):
        block = pred.get(side) or {}
        tid = str(block.get("id") or "")
        ab = abbr_by_id.get(tid, "")
        try:
            proj = float(block.get("gameProjection") or 0) / 100.0
        except (TypeError, ValueError):
            continue
        if ab == our_team.upper():
            out["espn_win_proj_our"] = proj
        elif ab == opp_team.upper():
            out["espn_win_proj_opp"] = proj

    # ── Injuries per team
    crit_statuses = {"OUT", "DOUBTFUL", "DAYTODAY", "DAY-TO-DAY"}
    for team_block in (summary.get("injuries") or []):
        ab = ((team_block.get("team") or {})
              .get("abbreviation") or "").upper()
        inj_list = team_block.get("injuries") or []
        n_total = len(inj_list)
        n_crit = 0
        for i in inj_list:
            status_label = (i.get("status") or "").upper().replace(" ", "")
            type_block = i.get("type") or {}
            abbr = (type_block.get("abbreviation") or "").upper()
            if status_label in crit_statuses or abbr in {"O", "DD", "D"}:
                n_crit += 1
        if ab == our_team.upper():
            out["our_injury_count"] = n_total
            out["our_critical_injuries"] = n_crit
        elif ab == opp_team.upper():
            out["opp_injury_count"] = n_total
            out["opp_critical_injuries"] = n_crit

    # ── Live box-score deltas (only meaningful when state == "in")
    boxscore = summary.get("boxscore") or {}
    teams = boxscore.get("teams") or []
    stat_by_team: Dict[str, Dict[str, float]] = {}
    for t in teams:
        team = t.get("team") or {}
        ab = (team.get("abbreviation") or "").upper()
        stats = t.get("statistics") or []
        d: Dict[str, float] = {}
        for s in stats:
            key = (s.get("abbreviation") or s.get("name") or "").upper()
            try:
                v = float(s.get("displayValue") or 0)
            except (TypeError, ValueError):
                continue
            d[key] = v
        stat_by_team[ab] = d
    our_stats = stat_by_team.get(our_team.upper(), {})
    opp_stats = stat_by_team.get(opp_team.upper(), {})
    # Pull a handful of recognizable stat keys. ESPN uses
    # abbreviations like FG%, FT%, REB, TO, AST, 3P%. We compute
    # the gap (our − opp) for each available stat.
    for key, out_key in [
        ("FG%", "team_fg_pct_gap"),
        ("FT%", "team_ft_pct_gap"),
        ("3P%", "team_3pt_pct_gap"),
        ("REB", "team_reb_gap"),
        ("TO", "team_to_gap"),  # turnovers: gap is negative-favorable
        ("AST", "team_ast_gap"),
    ]:
        if key in our_stats and key in opp_stats:
            out[out_key] = our_stats[key] - opp_stats[key]

    # ── Foul trouble: per-player fouls on our team. ESPN exposes
    # them in boxscore.players[*].statistics[*].athletes[*].stats
    # where each athlete's stats list is indexed by the same labels
    # the parent .labels array carries. Defensive: bail on any
    # shape mismatch.
    players_block = boxscore.get("players") or []
    foul_trouble = 0
    for pteam in players_block:
        ab = ((pteam.get("team") or {})
              .get("abbreviation") or "").upper()
        if ab != our_team.upper():
            continue
        for stat_block in (pteam.get("statistics") or []):
            labels = [(l or "").upper()
                       for l in (stat_block.get("labels") or [])]
            try:
                pf_idx = labels.index("PF")
            except ValueError:
                continue
            for ath in (stat_block.get("athletes") or []):
                stats = ath.get("stats") or []
                if pf_idx >= len(stats):
                    continue
                try:
                    if int(stats[pf_idx]) >= 4:
                        foul_trouble += 1
                except (TypeError, ValueError):
                    continue
    if players_block:
        out["foul_trouble_count"] = foul_trouble

    return out


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

    # ESPN /summary features — second-opinion win projection,
    # injury counts, live box-score gaps, foul-trouble count.
    summary_features = _extract_summary_features(
        state.get("event_id") or "",
        state.get("our_team") or "",
        state.get("opp_team") or "",
    )

    # Cross-sport news scanner — count recent injury-related ESPN
    # articles that mention either team's abbreviation. Folded as
    # a small per-article nudge: news that mentions OUR team in
    # an injury context tends to be bad for us.
    try:
        our_news_count = _news.injury_signal_count(
            "basketball/nba",
            [state.get("our_team") or ""],
            max_age_hours=12,
        )
        opp_news_count = _news.injury_signal_count(
            "basketball/nba",
            [state.get("opp_team") or ""],
            max_age_hours=12,
        )
    except Exception:  # noqa: BLE001
        our_news_count, opp_news_count = 0, 0
    summary_features["news_injury_ours"] = our_news_count
    summary_features["news_injury_opp"] = opp_news_count

    # Blend: the state-derived probability is the anchor. Reversion
    # pull nudges us back toward the pre-game prior when divergence
    # is high but live volatility is low (overreaction). High
    # volatility damps the in-game model's confidence — the market
    # is in flux and we shouldn't pretend to know better.
    blended = state_prob - 0.5 * reversion_pull

    # ESPN's own win projection gets a small weight when present —
    # treat it as a third independent opinion, average it in at a
    # 25% weight so it nudges without overriding.
    espn_proj = summary_features.get("espn_win_proj_our")
    if espn_proj is not None:
        blended = 0.75 * blended + 0.25 * float(espn_proj)

    # Home-court advantage. Standard NBA analytics put home edge
    # around +3pp on win probability for a neutral matchup. We
    # apply a small nudge proportional to time remaining — early
    # game the crowd factor is at its peak, late game player
    # adjustments dominate.
    HOME_NUDGE_MAX = 0.025
    ha = (state.get("our_home_away") or "").lower()
    if ha in ("home", "away"):
        secs_for_decay = float(state.get("seconds_remaining", 2880))
        # Decay home advantage as the game progresses (1.0 at tip,
        # 0.3 in the final minute). Game length ~48 min = 2880 s.
        decay = 0.3 + 0.7 * min(1.0, secs_for_decay / 2880.0)
        sign = 1 if ha == "home" else -1
        blended += sign * HOME_NUDGE_MAX * decay

    # Injury / foul-trouble nudges. Each significant injury or
    # foul-troubled key player on OUR team drops our_team_live_prob
    # by 1.5pp; same magnitude on the opp team raises it.
    INJURY_NUDGE = 0.015
    our_inj = summary_features.get("our_critical_injuries", 0) or 0
    opp_inj = summary_features.get("opp_critical_injuries", 0) or 0
    foul_trouble = summary_features.get("foul_trouble_count", 0) or 0
    blended -= INJURY_NUDGE * (our_inj + foul_trouble)
    blended += INJURY_NUDGE * opp_inj

    # Recent injury-news mentions: smaller signal than the
    # structured ESPN injury list (1pp vs 1.5pp). Only counts
    # articles from the last 12 hours.
    NEWS_NUDGE = 0.010
    blended -= NEWS_NUDGE * (summary_features.get("news_injury_ours", 0) or 0)
    blended += NEWS_NUDGE * (summary_features.get("news_injury_opp", 0) or 0)

    # Live-game box-score deltas: each pp of FG% / FT% gap nudges
    # the live prob by a small amount. Direction: positive gap for
    # our team raises our prob.
    BOX_NUDGE = 0.001  # per pp of stat gap
    for stat_key in ("team_fg_pct_gap", "team_ft_pct_gap",
                       "team_3pt_pct_gap", "team_ast_gap"):
        v = summary_features.get(stat_key)
        if v is None:
            continue
        try:
            blended += BOX_NUDGE * float(v)
        except (TypeError, ValueError):
            continue
    # Turnover gap: negative is favorable (we turn it over less).
    to_gap = summary_features.get("team_to_gap")
    if to_gap is not None:
        try:
            blended -= BOX_NUDGE * 3.0 * float(to_gap)
        except (TypeError, ValueError):
            pass
    # Rebound gap: positive favorable.
    reb_gap = summary_features.get("team_reb_gap")
    if reb_gap is not None:
        try:
            blended += BOX_NUDGE * 1.5 * float(reb_gap)
        except (TypeError, ValueError):
            pass

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

    features_out: Dict[str, float] = {
        "state_prob": state_prob,
        "our_side_prob": our_side_prob,
        "market_velocity": velocity,
        "volatility": volat,
        "divergence": div if div is not None else 0.0,
        "reversion_pull": reversion_pull,
        "lead": float(state.get("our_lead", 0)),
        "seconds_remaining": float(secs_left),
    }
    # Home/away flag — 1.0 home, 0.0 away, -1.0 unknown.
    ha_val = (state.get("our_home_away") or "").lower()
    if ha_val == "home":
        features_out["our_home_away"] = 1.0
    elif ha_val == "away":
        features_out["our_home_away"] = 0.0
    # ESPN /summary derived features — only emitted when present.
    for k in ("espn_win_proj_our", "our_critical_injuries",
              "opp_critical_injuries", "foul_trouble_count",
              "team_fg_pct_gap", "team_ft_pct_gap",
              "team_3pt_pct_gap", "team_to_gap",
              "team_reb_gap", "team_ast_gap",
              "news_injury_ours", "news_injury_opp"):
        v = summary_features.get(k)
        if v is None:
            continue
        try:
            features_out[k] = float(v)
        except (TypeError, ValueError):
            continue

    # Note in reason when summary features influenced the output.
    if summary_features.get("espn_win_proj_our") is not None:
        reason_bits.append(
            f"espn_proj={summary_features['espn_win_proj_our']:.2f}"
        )
    if (summary_features.get("our_critical_injuries", 0) or 0) > 0:
        reason_bits.append(
            f"our_inj={summary_features['our_critical_injuries']}"
        )
    if (summary_features.get("foul_trouble_count", 0) or 0) > 0:
        reason_bits.append(
            f"foul_trouble={summary_features['foul_trouble_count']}"
        )

    return LivePrediction(
        live_prob_yes=clamp01(blended),
        confidence=confidence,
        recommended_action=action,
        reason=" · ".join(reason_bits),
        features=features_out,
    )
