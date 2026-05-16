"""NBA live-data adapter (NBA.com CDN endpoints).

Wraps three public CDN endpoints that NBA.com itself uses to power
its live-scoreboard pages:

    scoreboard  : /liveData/scoreboard/todaysScoreboard_00.json
    boxscore    : /liveData/boxscore/boxscore_<GAME_ID>.json
    playbyplay  : /liveData/playbyplay/playbyplay_<GAME_ID>.json

Why this instead of the ``nba_api`` package?
  - No external dependency. The dashboard is stdlib-only and we
    want to keep it that way.
  - The CDN endpoints don't need API keys; they're the same JSON
    NBA.com hits from the browser. As long as we send a real
    User-Agent + Referer + Origin, they return clean JSON.
  - Rate limits are generous since this is the CDN that serves
    millions of NBA.com page loads. A 15-30s polling cadence
    from one client is well below noise.

Returns ``{}`` (empty) on any failure — caller falls back to
ESPN data or the heuristic baseline. The dashboard should never
break because the CDN is having a moment.

Public API
----------
``advanced_features(our_team, opp_team)``
    Look up today's game by team tricode, pull live boxscore,
    return a dict of features the in-game model can nudge with.

Cached: 60s for scoreboard, 15s per boxscore (game-id keyed).
"""
from __future__ import annotations

import json
import logging
import time
import urllib.request
from typing import Any, Dict, Optional, Tuple

log = logging.getLogger("dashboard.in_game.nba_cdn")


_CDN_BASE = "https://cdn.nba.com/static/json/liveData"
_BROWSER_HEADERS = {
    # NBA's CDN edge config seems to whitelist requests that look
    # like the same JS fetches NBA.com itself makes. A bare
    # urllib User-Agent gets a 403.
    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0 Safari/537.36"),
    "Referer": "https://www.nba.com/",
    "Origin": "https://www.nba.com",
    "Accept": "application/json",
}

# Scoreboard cache: single key. (ts, parsed_json).
_SB_CACHE: Tuple[float, Optional[dict]] = (0.0, None)
_SB_TTL = 60.0

# Boxscore cache: keyed by game_id.
_BS_CACHE: Dict[str, Tuple[float, Optional[dict]]] = {}
_BS_TTL = 15.0


def _fetch_json(url: str, timeout: int = 5) -> Optional[dict]:
    """Generic HTTP-GET-JSON with the browser-style header set the
    CDN requires. Returns ``None`` on any error (logged at DEBUG).
    """
    req = urllib.request.Request(url, headers=_BROWSER_HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)
    except Exception as exc:  # noqa: BLE001
        log.debug("NBA CDN fetch failed (%s): %s", url, exc)
        return None


def _fetch_scoreboard() -> Optional[dict]:
    global _SB_CACHE
    now = time.time()
    ts, data = _SB_CACHE
    if (now - ts) < _SB_TTL and data is not None:
        return data
    data = _fetch_json(f"{_CDN_BASE}/scoreboard/todaysScoreboard_00.json")
    _SB_CACHE = (now, data)
    return data


def _fetch_boxscore(game_id: str) -> Optional[dict]:
    if not game_id:
        return None
    now = time.time()
    cached = _BS_CACHE.get(game_id)
    if cached and (now - cached[0]) < _BS_TTL:
        return cached[1]
    data = _fetch_json(f"{_CDN_BASE}/boxscore/boxscore_{game_id}.json")
    _BS_CACHE[game_id] = (now, data)
    return data


def _game_id_for_matchup(our_team: str,
                           opp_team: str) -> Optional[str]:
    """Walk today's scoreboard and return the gameId for the
    matchup with our two team tricodes (in either home/away
    arrangement).
    """
    if not our_team or not opp_team:
        return None
    our_u = our_team.upper()
    opp_u = opp_team.upper()
    sb = _fetch_scoreboard()
    if not sb:
        return None
    games = ((sb.get("scoreboard") or {}).get("games") or [])
    for g in games:
        away = ((g.get("awayTeam") or {}).get("teamTricode") or "").upper()
        home = ((g.get("homeTeam") or {}).get("teamTricode") or "").upper()
        names = {away, home}
        if our_u in names and opp_u in names:
            return str(g.get("gameId") or "")
    return None


def _team_block(bs: dict, side: str) -> Optional[dict]:
    """``side`` is "home" or "away". Returns the boxScore team
    block or None when the CDN response is missing the structure.
    """
    boxscore = bs.get("boxScore") if isinstance(bs, dict) else None
    if not boxscore:
        return None
    return boxscore.get("homeTeam") if side == "home" else boxscore.get("awayTeam")


def advanced_features(our_team: str, opp_team: str) -> Dict[str, Any]:
    """Top-level helper. Returns the merged feature dict for the
    in-game model. Empty when no game found, CDN unreachable, or
    box-score not yet live.

    Keys returned (all optional):
      - ``cdn_live_pace``               : possessions/48 minutes
      - ``cdn_fg_pct_gap``              : (our FG% − opp FG%) in pp
      - ``cdn_3pt_pct_gap``             : (our 3P% − opp 3P%) in pp
      - ``cdn_ft_rate_gap``             : (our FT/FGA − opp FT/FGA)
                                            in pp
      - ``cdn_to_gap``                  : (our TOs − opp TOs)
      - ``cdn_reb_gap``                 : (our REB − opp REB)
      - ``cdn_fouls_out_ours``          : count of our players
                                            who have fouled out (PF≥6)
      - ``cdn_foul_trouble_ours``       : count of our players w/ 4-5 PF
      - ``cdn_starter_plusminus_avg``   : mean plus-minus across our
                                            starters
      - ``cdn_bench_plusminus_avg``     : mean plus-minus across our
                                            bench players who played
      - ``cdn_bench_minutes_share``     : our bench minutes / total
    """
    out: Dict[str, Any] = {}
    game_id = _game_id_for_matchup(our_team, opp_team)
    if not game_id:
        return out
    bs = _fetch_boxscore(game_id)
    if not bs:
        return out

    # Identify which side is our team.
    boxscore = bs.get("boxScore") or {}
    home_tri = ((boxscore.get("homeTeam") or {})
                .get("teamTricode") or "").upper()
    if not home_tri:
        return out
    our_side = "home" if our_team.upper() == home_tri else "away"
    opp_side = "away" if our_side == "home" else "home"
    our_block = _team_block(bs, our_side) or {}
    opp_block = _team_block(bs, opp_side) or {}

    def _team_stat(block: dict, *keys, default=None):
        s = block.get("statistics") or {}
        for k in keys:
            if k in s and s[k] is not None:
                return s[k]
        return default

    # ── Live pace (possessions per 48 min) — single value per team
    pace = _team_stat(our_block, "pace")
    if pace is not None:
        try:
            out["cdn_live_pace"] = float(pace)
        except (TypeError, ValueError):
            pass

    # ── Team stat gaps (in percentage-points). The CDN reports
    # percentages as 0-100 floats (47.8 = 47.8%), not 0-1 decimals.
    def _gap(stat_key: str) -> Optional[float]:
        ours = _team_stat(our_block, stat_key)
        opps = _team_stat(opp_block, stat_key)
        if ours is None or opps is None:
            return None
        try:
            return float(ours) - float(opps)
        except (TypeError, ValueError):
            return None

    for stat_key, out_key in [
        ("fieldGoalsPercentage", "cdn_fg_pct_gap"),
        ("threePointersPercentage", "cdn_3pt_pct_gap"),
        ("turnoversTeam", "cdn_to_gap"),
        ("reboundsTotal", "cdn_reb_gap"),
    ]:
        v = _gap(stat_key)
        if v is not None:
            out[out_key] = v

    # Free-throw rate (FT attempted / FG attempted) — compute per-team
    # since CDN exposes raw attempts.
    def _ft_rate(block: dict) -> Optional[float]:
        s = block.get("statistics") or {}
        try:
            fta = float(s.get("freeThrowsAttempted") or 0)
            fga = float(s.get("fieldGoalsAttempted") or 0)
            if fga <= 0:
                return None
            return fta / fga
        except (TypeError, ValueError):
            return None

    our_ftr = _ft_rate(our_block)
    opp_ftr = _ft_rate(opp_block)
    if our_ftr is not None and opp_ftr is not None:
        out["cdn_ft_rate_gap"] = (our_ftr - opp_ftr) * 100.0  # in pp

    # ── Per-player digestion: foul trouble + starter/bench +/-.
    players = our_block.get("players") or []
    fouled_out = 0
    foul_trouble = 0
    starter_pm: list = []
    bench_pm: list = []
    bench_min = 0.0
    total_min = 0.0
    for p in players:
        s = p.get("statistics") or {}
        try:
            pf = int(s.get("foulsPersonal") or 0)
            pm = int(s.get("plusMinusPoints") or 0)
        except (TypeError, ValueError):
            pf, pm = 0, 0
        # minutes is "PT24M30.00S" ISO duration; parse out the minutes int.
        mins_str = s.get("minutesCalculated") or s.get("minutes") or ""
        try:
            # Look for "PT<X>M" pattern.
            mm = 0.0
            if "M" in mins_str:
                head = mins_str.split("M", 1)[0]
                head = head.replace("PT", "")
                mm = float(head)
            total_min += mm
        except (TypeError, ValueError):
            mm = 0.0
        if pf >= 6:
            fouled_out += 1
        elif pf >= 4:
            foul_trouble += 1
        # NBA boxscore marks starters with `starter == "1"` in the
        # public CDN payload.
        starter_flag = str(p.get("starter") or "").strip()
        is_starter = starter_flag == "1"
        if mm <= 0:
            continue  # didn't play; +/- noisy
        if is_starter:
            starter_pm.append(pm)
        else:
            bench_pm.append(pm)
            bench_min += mm

    out["cdn_fouls_out_ours"] = fouled_out
    out["cdn_foul_trouble_ours"] = foul_trouble
    if starter_pm:
        out["cdn_starter_plusminus_avg"] = sum(starter_pm) / len(starter_pm)
    if bench_pm:
        out["cdn_bench_plusminus_avg"] = sum(bench_pm) / len(bench_pm)
    if total_min > 0:
        out["cdn_bench_minutes_share"] = bench_min / total_min

    return out
